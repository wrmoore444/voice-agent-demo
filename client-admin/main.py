from fastapi import FastAPI, Depends, WebSocket, HTTPException, Form, File, UploadFile
from fastapi.responses import Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
import os
from contextlib import asynccontextmanager
from typing import Optional
from sqlalchemy.orm import selectinload
from loguru import logger
from models import User, Agent, Conversation, MenuItem, EventBooking, Base
from voice_agent import run_voice_bot
from fastapi.middleware.cors import CORSMiddleware  
from uuid import uuid4
from datetime import datetime

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=1800,
    pool_timeout=30,
)

AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting...")
    
    # 🔥 AUTOMATIC TABLE CREATION ON STARTUP 🔥
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("✅ Database tables created/verified successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to create database tables: {e}")
        raise
    
    yield
    
    logger.info("Application shutting down, closing database connections...")
    await engine.dispose()
    logger.info("Database connections closed")

app = FastAPI(title="Agents + Conversations API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


# ========== USERS ==========

@app.post("/login")
async def login(username: str, password: str, db: AsyncSession = Depends(get_db)):
    """Simple login - returns user id and username"""
    
    from sqlalchemy import select
    
    # Find user (async way)
    result = await db.execute(
        select(User).filter(
            User.username == username,
            User.password == password
        )
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    return {"id": user.id, "username": user.username}


@app.post("/register")
async def register(username: str, password: str, db: AsyncSession = Depends(get_db)):
    """Create new user"""
    
    from sqlalchemy import select
    
    # Check if user already exists
    result = await db.execute(select(User).filter(User.username == username))
    existing_user = result.scalar_one_or_none()
    
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Create new user
    new_user = User(username=username, password=password)
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    
    return {"id": new_user.id, "username": new_user.username}

@app.post("/users")
async def create_user(username: str = Form(...), password: str = Form(...), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).filter(User.username == username))
    existing = result.scalar_one_or_none()
    if existing:
        raise HTTPException(400, "Username already exists")
    user = User(username=username)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return {"id": user.id, "username": user.username}

@app.get("/users/{user_id}")
async def get_user(user_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).filter(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(404, "User not found")
    return {"id": user.id, "username": user.username}

# ========== AGENTS ==========
@app.post("/agents")
async def create_agent(
    user_id: int = Form(...),
    name: str = Form(...),
    prompt: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(User).filter(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(404, "User not found")
    agent = Agent(name=name, prompt=prompt, user_id=user_id)
    db.add(agent)
    await db.commit()
    await db.refresh(agent)
    return {"id": agent.id, "name": agent.name, "prompt": agent.prompt}

@app.get("/agents/{agent_id}")
async def get_agent(agent_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Agent).filter(Agent.id == agent_id))
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(404, "Agent not found")
    return {"id": agent.id, "name": agent.name, "prompt": agent.prompt}

@app.put("/agents/{agent_id}")
async def update_agent(
    agent_id: int,
    name: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(Agent).filter(Agent.id == agent_id))
    agent = result.scalar_one_or_none()
    if not agent:
        raise HTTPException(404, "Agent not found")
    if name is not None:
        agent.name = name
    if prompt is not None:
        agent.prompt = prompt
    await db.commit()
    await db.refresh(agent)
    return {
        "id": agent.id,
        "name": agent.name,
        "prompt": agent.prompt,
        "message": "Agent updated successfully"
    }

@app.get("/users/{user_id}/agents")
async def get_user_agents(user_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Agent).filter(Agent.user_id == user_id))
    agents = result.scalars().all()
    return [{"id": a.id, "name": a.name, "prompt": a.prompt} for a in agents]

# ========== CONVERSATIONS ==========
@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: int, db: AsyncSession = Depends(get_db)):
    """Get conversation with all its transcriptions"""
    result = await db.execute(
        select(Conversation)
        .options(selectinload(Conversation.transcriptions))
        .filter(Conversation.id == conversation_id)
    )
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(404, "Conversation not found")
    
    return {
        "id": conv.id,
        "user_id": conv.user_id,
        "agent_id": conv.agent_id,
        "has_audio": conv.audio_file is not None,
        "audio_filename": conv.audio_filename,
        "datapoints": conv.datapoints,
        "transcriptions": [
            {
                "id": t.id,
                "text": t.text,
                "speaker": t.speaker,
                "timestamp": t.timestamp.isoformat()
            }
            for t in conv.transcriptions
        ]
    }

@app.get("/users/{user_id}/conversations")
async def get_user_conversations(user_id: int, agent_id: Optional[int] = None, db: AsyncSession = Depends(get_db)):
    query = (
        select(Conversation)
        .options(
            selectinload(Conversation.agent), 
            selectinload(Conversation.transcriptions)
        )
        .filter(Conversation.user_id == user_id)
    )
    if agent_id:
        query = query.filter(Conversation.agent_id == agent_id)
    result = await db.execute(query)
    convs = result.scalars().all()
    return [
        {
            "id": c.id,
            "agent_id": c.agent_id,
            "agent_name": c.agent.name,
            "has_audio": c.audio_file is not None,
            "transcription_count": len(c.transcriptions),
            "created_at": c.created_at.isoformat()
        }
        for c in convs
    ]

@app.get("/conversations/user/{user_id}/agent/{agent_id}")
async def get_conversations_by_user_and_agent(user_id: int, agent_id: int, db: AsyncSession = Depends(get_db)):
    """Get all conversations between a specific user and agent"""
    result = await db.execute(
        select(Conversation)
        .options(
            selectinload(Conversation.agent),
            selectinload(Conversation.transcriptions)
        )
        .filter(
            Conversation.user_id == user_id,
            Conversation.agent_id == agent_id
        )
    )
    convs = result.scalars().all()
    if not convs:
        return []
    return [
        {
            "id": c.id,
            "user_id": c.user_id,
            "agent_id": c.agent_id,
            "agent_name": c.agent.name if c.agent else None,
            "has_audio": c.audio_file is not None,
            "audio_filename": c.audio_filename,
            "datapoints": c.datapoints,
            "transcription_count": len(c.transcriptions),
            "created_at": c.created_at.isoformat(),
            "transcriptions": [
                {
                    "id": t.id,
                    "text": t.text,
                    "speaker": t.speaker,
                    "timestamp": t.timestamp.isoformat()
                }
                for t in c.transcriptions
            ]
        }
        for c in convs
    ]

# ========== MENU ITEMS ==========
@app.post("/menu-items")
async def create_menu_item(
    category: str = Form(...),
    title: str = Form(...),
    description: Optional[str] = Form(None),
    price: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    db: AsyncSession = Depends(get_db)
):
    """Create a new menu item with optional image upload"""
    menu_item = MenuItem(
        category=category,
        title=title,
        description=description,
        price=price
    )
    
    # Handle image upload
    if image:
        image_data = await image.read()
        menu_item.image = image_data
        menu_item.image_filename = image.filename
        menu_item.image_mime = image.content_type
    
    db.add(menu_item)
    await db.commit()
    await db.refresh(menu_item)
    
    return {
        "id": menu_item.id,
        "category": menu_item.category,
        "title": menu_item.title,
        "description": menu_item.description,
        "price": menu_item.price,
        "has_image": menu_item.image is not None,
        "created_at": menu_item.created_at.isoformat()
    }

@app.get("/menu-items")
async def get_menu_items(
    category: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """Get all menu items, optionally filtered by category"""
    query = select(MenuItem)
    
    if category:
        query = query.filter(MenuItem.category == category)
    
    result = await db.execute(query)
    items = result.scalars().all()
    
    return [
        {
            "id": item.id,
            "category": item.category,
            "title": item.title,
            "description": item.description,
            "price": item.price,
            "has_image": item.image is not None,
            "image_url": f"/menu-items/{item.id}/image" if item.image else None,
            "created_at": item.created_at.isoformat()
        }
        for item in items
    ]

@app.get("/menu-items/{item_id}")
async def get_menu_item(item_id: int, db: AsyncSession = Depends(get_db)):
    """Get a specific menu item by ID"""
    result = await db.execute(select(MenuItem).filter(MenuItem.id == item_id))
    item = result.scalar_one_or_none()
    
    if not item:
        raise HTTPException(404, "Menu item not found")
    
    return {
        "id": item.id,
        "category": item.category,
        "title": item.title,
        "description": item.description,
        "price": item.price,
        "has_image": item.image is not None,
        "image_url": f"/menu-items/{item.id}/image" if item.image else None,
        "created_at": item.created_at.isoformat(),
        "updated_at": item.updated_at.isoformat()
    }

@app.get("/menu-items/{item_id}/image")
async def get_menu_item_image(item_id: int, db: AsyncSession = Depends(get_db)):
    """Get the image for a specific menu item"""
    result = await db.execute(select(MenuItem).filter(MenuItem.id == item_id))
    item = result.scalar_one_or_none()
    
    if not item:
        raise HTTPException(404, "Menu item not found")
    
    if not item.image:
        raise HTTPException(404, "Menu item has no image")
    
    return Response(
        content=item.image,
        media_type=item.image_mime or "image/jpeg"
    )

@app.get("/menu")
async def get_full_menu(db: AsyncSession = Depends(get_db)):
    """Get complete menu grouped by categories"""
    result = await db.execute(select(MenuItem).order_by(MenuItem.category, MenuItem.id))
    items = result.scalars().all()
    
    # Group items by category
    menu_by_category = {}
    for item in items:
        if item.category not in menu_by_category:
            menu_by_category[item.category] = []
        
        menu_by_category[item.category].append({
            "id": item.id,
            "title": item.title,
            "description": item.description,
            "price": item.price,
            "has_image": item.image is not None,
            "image_url": f"/menu-items/{item.id}/image" if item.image else None,
        })
    
    return {
        "categories": [
            {
                "category": category,
                "items": items
            }
            for category, items in menu_by_category.items()
        ]
    }

@app.put("/menu-items/{item_id}")
async def update_menu_item(
    item_id: int,
    category: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    price: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    db: AsyncSession = Depends(get_db)
):
    """Update a menu item"""
    result = await db.execute(select(MenuItem).filter(MenuItem.id == item_id))
    item = result.scalar_one_or_none()
    
    if not item:
        raise HTTPException(404, "Menu item not found")
    
    # Update fields
    if category is not None:
        item.category = category
    if title is not None:
        item.title = title
    if description is not None:
        item.description = description
    if price is not None:
        item.price = price
    
    # Update image if provided
    if image:
        image_data = await image.read()
        item.image = image_data
        item.image_filename = image.filename
        item.image_mime = image.content_type
    
    item.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(item)
    
    return {
        "id": item.id,
        "category": item.category,
        "title": item.title,
        "description": item.description,
        "price": item.price,
        "has_image": item.image is not None,
        "message": "Menu item updated successfully"
    }

@app.delete("/menu-items/{item_id}")
async def delete_menu_item(item_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a menu item"""
    result = await db.execute(select(MenuItem).filter(MenuItem.id == item_id))
    item = result.scalar_one_or_none()
    
    if not item:
        raise HTTPException(404, "Menu item not found")
    
    await db.delete(item)
    await db.commit()
    
    return {"message": "Menu item deleted successfully"}

@app.get("/menu-categories")
async def get_menu_categories(db: AsyncSession = Depends(get_db)):
    """Get all unique menu categories"""
    result = await db.execute(select(MenuItem.category).distinct())
    categories = result.scalars().all()
    return {"categories": categories}

# ========== EVENT BOOKINGS ==========
@app.post("/event-bookings")
async def create_event_booking(
    # Contact Information
    full_name: str = Form(...),
    email: str = Form(...),
    phone: str = Form(...),
    contact_method: str = Form(...),
    contact_consent: str = Form(...),
    company: Optional[str] = Form(None),
    
    # Event Details
    event_type: str = Form(...),
    event_type_other: Optional[str] = Form(None),
    event_date: str = Form(...),  # Will be converted to datetime
    date_flexible: str = Form(...),
    start_time: str = Form(...),
    end_time: Optional[str] = Form(None),
    guest_count: str = Form(...),
    
    # Venue Preference
    preferred_space: str = Form(...),
    event_vibe: Optional[str] = Form(None),
    
    # Food & Beverage (comma-separated string)
    food_style: Optional[str] = Form(None),
    
    db: AsyncSession = Depends(get_db)
):
    """Create a new event booking"""
    from datetime import datetime as dt
    
    # Convert date string to datetime
    try:
        event_date_obj = dt.strptime(event_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(400, "Invalid date format. Use YYYY-MM-DD")
    
    booking = EventBooking(
        full_name=full_name,
        email=email,
        phone=phone,
        contact_method=contact_method,
        contact_consent=contact_consent,
        company=company,
        event_type=event_type,
        event_type_other=event_type_other,
        event_date=event_date_obj,
        date_flexible=date_flexible,
        start_time=start_time,
        end_time=end_time,
        guest_count=guest_count,
        preferred_space=preferred_space,
        event_vibe=event_vibe,
        food_style=food_style,
        status="pending"
    )
    
    db.add(booking)
    await db.commit()
    await db.refresh(booking)
    
    return {
        "id": booking.id,
        "full_name": booking.full_name,
        "email": booking.email,
        "event_type": booking.event_type,
        "event_date": booking.event_date.isoformat(),
        "guest_count": booking.guest_count,
        "preferred_space": booking.preferred_space,
        "status": booking.status,
        "created_at": booking.created_at.isoformat(),
        "message": "Event booking created successfully"
    }

@app.get("/event-bookings")
async def get_all_event_bookings(
    status: Optional[str] = None,
    event_type: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """Get all event bookings with optional filters"""
    
    query = select(EventBooking)
    
    if status:
        query = query.filter(EventBooking.status == status)
    if event_type:
        query = query.filter(EventBooking.event_type == event_type)
    
    query = query.order_by(EventBooking.created_at.desc())
    
    result = await db.execute(query)
    bookings = result.scalars().all()
    
    return [
        {
            "id": booking.id,
            "full_name": booking.full_name,
            "email": booking.email,
            "phone": booking.phone,
            "contact_method": booking.contact_method,
            "contact_consent": booking.contact_consent,
            "company": booking.company,
            "event_type": booking.event_type,
            "event_type_other": booking.event_type_other,
            "event_date": booking.event_date.isoformat(),
            "date_flexible": booking.date_flexible,
            "start_time": booking.start_time,
            "end_time": booking.end_time,
            "guest_count": booking.guest_count,
            "preferred_space": booking.preferred_space,
            "event_vibe": booking.event_vibe,
            "food_style": booking.food_style,
            "status": booking.status,
            "created_at": booking.created_at.isoformat(),
            "updated_at": booking.updated_at.isoformat()
        }
        for booking in bookings
    ]

@app.get("/event-bookings/{booking_id}")
async def get_event_booking(booking_id: int, db: AsyncSession = Depends(get_db)):
    """Get a specific event booking by ID"""
    
    result = await db.execute(select(EventBooking).filter(EventBooking.id == booking_id))
    booking = result.scalar_one_or_none()
    
    if not booking:
        raise HTTPException(404, "Event booking not found")
    
    return {
        "id": booking.id,
        "full_name": booking.full_name,
        "email": booking.email,
        "phone": booking.phone,
        "contact_method": booking.contact_method,
        "contact_consent": booking.contact_consent,
        "company": booking.company,
        "event_type": booking.event_type,
        "event_type_other": booking.event_type_other,
        "event_date": booking.event_date.isoformat(),
        "date_flexible": booking.date_flexible,
        "start_time": booking.start_time,
        "end_time": booking.end_time,
        "guest_count": booking.guest_count,
        "preferred_space": booking.preferred_space,
        "event_vibe": booking.event_vibe,
        "food_style": booking.food_style,
        "status": booking.status,
        "created_at": booking.created_at.isoformat(),
        "updated_at": booking.updated_at.isoformat()
    }

@app.put("/event-bookings/{booking_id}")
async def update_event_booking(
    booking_id: int,
    # Contact Information
    full_name: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    phone: Optional[str] = Form(None),
    contact_method: Optional[str] = Form(None),
    contact_consent: Optional[str] = Form(None),
    company: Optional[str] = Form(None),
    
    # Event Details
    event_type: Optional[str] = Form(None),
    event_type_other: Optional[str] = Form(None),
    event_date: Optional[str] = Form(None),
    date_flexible: Optional[str] = Form(None),
    start_time: Optional[str] = Form(None),
    end_time: Optional[str] = Form(None),
    guest_count: Optional[str] = Form(None),
    
    # Venue Preference
    preferred_space: Optional[str] = Form(None),
    event_vibe: Optional[str] = Form(None),
    
    # Food & Beverage
    food_style: Optional[str] = Form(None),
    
    # Status
    status: Optional[str] = Form(None),
    
    db: AsyncSession = Depends(get_db)
):
    """Update an event booking"""
    from datetime import datetime as dt
    
    result = await db.execute(select(EventBooking).filter(EventBooking.id == booking_id))
    booking = result.scalar_one_or_none()
    
    if not booking:
        raise HTTPException(404, "Event booking not found")
    
    # Update fields
    if full_name is not None:
        booking.full_name = full_name
    if email is not None:
        booking.email = email
    if phone is not None:
        booking.phone = phone
    if contact_method is not None:
        booking.contact_method = contact_method
    if contact_consent is not None:
        booking.contact_consent = contact_consent
    if company is not None:
        booking.company = company
    if event_type is not None:
        booking.event_type = event_type
    if event_type_other is not None:
        booking.event_type_other = event_type_other
    if event_date is not None:
        try:
            booking.event_date = dt.strptime(event_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(400, "Invalid date format. Use YYYY-MM-DD")
    if date_flexible is not None:
        booking.date_flexible = date_flexible
    if start_time is not None:
        booking.start_time = start_time
    if end_time is not None:
        booking.end_time = end_time
    if guest_count is not None:
        booking.guest_count = guest_count
    if preferred_space is not None:
        booking.preferred_space = preferred_space
    if event_vibe is not None:
        booking.event_vibe = event_vibe
    if food_style is not None:
        booking.food_style = food_style
    if status is not None:
        booking.status = status
    
    booking.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(booking)
    
    return {
        "id": booking.id,
        "full_name": booking.full_name,
        "email": booking.email,
        "event_type": booking.event_type,
        "event_date": booking.event_date.isoformat(),
        "status": booking.status,
        "message": "Event booking updated successfully"
    }

@app.delete("/event-bookings/{booking_id}")
async def delete_event_booking(booking_id: int, db: AsyncSession = Depends(get_db)):
    """Delete an event booking"""
    
    result = await db.execute(select(EventBooking).filter(EventBooking.id == booking_id))
    booking = result.scalar_one_or_none()
    
    if not booking:
        raise HTTPException(404, "Event booking not found")
    
    await db.delete(booking)
    await db.commit()
    
    return {"message": "Event booking deleted successfully"}

@app.get("/event-bookings/stats/summary")
async def get_booking_stats(db: AsyncSession = Depends(get_db)):
    """Get booking statistics"""
    from sqlalchemy import func
    
    # Count by status
    status_result = await db.execute(
        select(EventBooking.status, func.count(EventBooking.id))
        .group_by(EventBooking.status)
    )
    status_counts = dict(status_result.all())
    
    # Count by event type
    type_result = await db.execute(
        select(EventBooking.event_type, func.count(EventBooking.id))
        .group_by(EventBooking.event_type)
    )
    type_counts = dict(type_result.all())
    
    # Total bookings
    total_result = await db.execute(select(func.count(EventBooking.id)))
    total = total_result.scalar()
    
    return {
        "total_bookings": total,
        "by_status": status_counts,
        "by_event_type": type_counts
    }

# ========== WEBSOCKET ==========
active_websockets = set()

@app.websocket("/ws/{user_id}/{agent_id}/{current_conversation_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    user_id: int,
    agent_id: int,
    current_conversation_id: str = None,
    db: AsyncSession = Depends(get_db)
):
    await websocket.accept()
    voice_session = None
    try:
        # Get user
        user_result = await db.execute(select(User).filter(User.id == user_id))
        user = user_result.scalar_one_or_none()

        # Get agent
        agent_result = await db.execute(
            select(Agent).filter(Agent.id == agent_id, Agent.user_id == user_id)
        )
        agent = agent_result.scalar_one_or_none()

        if not user or not agent:
            logger.error(f"User {user_id} or Agent {agent_id} not found")
            await websocket.close(code=1008, reason="User or Agent not found")
            return

        logger.info(f"Voice bot started: user={user_id}, agent={agent_id}")

        # Create dedicated session for voice bot to avoid conflicts
        async with AsyncSessionLocal() as voice_session:
            try:
                await run_voice_bot(
                    websocket_client=websocket,
                    agent_prompt=agent.prompt,
                    agent_name=agent.name,
                    user_id=user_id,
                    agent_id=agent_id,
                    db_session=voice_session,
                    current_conversation_id=current_conversation_id
                )
            finally:
                # Ensure session cleanup
                if voice_session and voice_session.is_active:
                    voice_session.expunge_all()
                    await voice_session.close()
                    logger.info("Voice session closed and cleaned up")

    except Exception as e:
        logger.exception(f"WebSocket error: {e}")
    finally:
        # Ensure WebSocket is closed
        try:
            if websocket.client_state.name != "DISCONNECTED":
                await websocket.close()
        except:
            pass
        logger.info(f"WebSocket connection closed for user={user_id}, agent={agent_id}")
    
        
@app.post("/connect/{user_id}/{agent_id}")
async def get_websocket_url(
    user_id: int,
    agent_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get WebSocket URL and create conversation if needed"""
    
    # Get user
    user_result = await db.execute(select(User).filter(User.id == user_id))
    user = user_result.scalar_one_or_none()

    # Get agent
    agent_result = await db.execute(
        select(Agent).filter(Agent.id == agent_id, Agent.user_id == user_id)
    )
    agent = agent_result.scalar_one_or_none()

    if not user or not agent:
        raise HTTPException(404, "User or Agent not found")

    conv_uuid = str(uuid4())

    conversation = Conversation(
        agent_id=agent_id,
        user_id=user_id,
        uuid=conv_uuid,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )

    db.add(conversation)
    await db.commit()
    await db.refresh(conversation)

    ws_url = (
        f"wss://client-admin-f4be5302a1c2.herokuapp.com/ws/"
        f"{user_id}/{agent_id}/{conversation.uuid}"
    )

    print(ws_url)

    return {
        "ws_url": ws_url,
    }