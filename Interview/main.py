from fastapi import FastAPI, Depends, WebSocket, HTTPException, Form, File, UploadFile, Request, Query
from fastapi.responses import Response, HTMLResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
import os
import asyncio
from contextlib import asynccontextmanager
from typing import Optional
from sqlalchemy.orm import selectinload
from loguru import logger
from models import User, Agent, Conversation, MenuItem, EventBooking, Base
from voice_agent import run_voice_bot
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
from datetime import datetime
import json

# Bot demo imports
from bot_demo import DualBotService, list_personas

# Pipecat bot demo imports
from bot_demo_pipecat import PipecatDualBotService, list_personas as pipecat_list_personas

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
    request: Request,
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

    # ws_url = f"ws://localhost:8000/ws/{user_id}/{agent_id}/{conversation.uuid}"
    
    base = str(request.base_url).rstrip("/")  # e.g. http://localhost:8000
    ws_base = base.replace("http://", "ws://").replace("https://", "wss://")

    ws_url = f"{ws_base}/ws/{user_id}/{agent_id}/{conversation.uuid}"

    print(ws_url)

    return {
        "ws_url": ws_url,
    }


# ========== BOT-TO-BOT DEMO ==========

# Global demo service instance
demo_service = DualBotService(max_turns=20, turn_delay_ms=300)  # Longer for stage testing


@app.get("/demo/personas")
async def get_personas():
    """Get list of available personas."""
    return list_personas()


@app.post("/demo/start")
async def start_demo(
    topic: str = Query(default="", description="Conversation topic (optional)"),
    alice: str = Query(default=None, description="Alice persona filename"),
    bob: str = Query(default=None, description="Bob persona filename")
):
    """Start a bot-to-bot conversation demo with selected personas."""
    success = await demo_service.start(
        topic=topic,
        alice_persona=alice,
        bob_persona=bob
    )
    if not success:
        raise HTTPException(400, "Demo already running or failed to start")

    state = demo_service.get_state()
    return {
        "status": "started",
        "alice_persona": state.get("alice_persona"),
        "bob_persona": state.get("bob_persona"),
        "topic": state.get("topic"),
        "message": "Bot conversation started"
    }


@app.post("/demo/stop")
async def stop_demo():
    """Stop the bot-to-bot conversation demo."""
    success = await demo_service.stop()
    if not success:
        raise HTTPException(400, "No demo running")
    return {
        "status": "stopped",
        "message": "Bot conversation stopped"
    }


@app.get("/demo/status")
async def get_demo_status():
    """Get current demo status."""
    return demo_service.get_state()


@app.websocket("/demo/viewer/ws")
async def demo_viewer_websocket(websocket: WebSocket):
    """WebSocket endpoint for viewing the bot-to-bot conversation."""
    await websocket.accept()
    viewer_queue = demo_service.register_viewer()

    try:
        # Send current conversation history on connect
        state = demo_service.get_state()
        await websocket.send_json({
            "type": "history",
            "data": state.get("conversation_history", [])
        })

        # Stream new messages and audio
        while True:
            try:
                message = await asyncio.wait_for(viewer_queue.get(), timeout=30.0)
                # Message is already formatted with type from DualBotService
                await websocket.send_json(message)
            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_json({"type": "ping"})
            except Exception as e:
                logger.error(f"Error sending to viewer: {e}")
                break
    except Exception as e:
        logger.error(f"Viewer WebSocket error: {e}")
    finally:
        demo_service.unregister_viewer(viewer_queue)
        try:
            await websocket.close()
        except:
            pass


@app.get("/demo/viewer", response_class=HTMLResponse)
async def demo_viewer_page():
    """Serve the demo viewer HTML page."""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bot-to-Bot Conversation Viewer</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e4e4e7;
            padding: 20px;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            margin-bottom: 20px;
            color: #a78bfa;
        }
        .controls {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
            justify-content: center;
            align-items: center;
        }
        input[type="text"] {
            padding: 12px 16px;
            border: 2px solid #3f3f46;
            border-radius: 8px;
            background: #27272a;
            color: #e4e4e7;
            font-size: 16px;
            flex: 1;
            min-width: 200px;
        }
        input[type="text"]:focus {
            outline: none;
            border-color: #a78bfa;
        }
        button {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn-start {
            background: #22c55e;
            color: white;
        }
        .btn-start:hover {
            background: #16a34a;
        }
        .btn-stop {
            background: #ef4444;
            color: white;
        }
        .btn-stop:hover {
            background: #dc2626;
        }
        .btn-clear {
            background: #3f3f46;
            color: #e4e4e7;
        }
        .btn-clear:hover {
            background: #52525b;
        }
        .audio-toggle {
            display: flex;
            align-items: center;
            gap: 8px;
            background: #27272a;
            padding: 8px 16px;
            border-radius: 8px;
        }
        .audio-toggle input {
            width: 18px;
            height: 18px;
            cursor: pointer;
        }
        .audio-toggle label {
            cursor: pointer;
            font-size: 14px;
        }
        .status {
            text-align: center;
            margin-bottom: 20px;
            padding: 10px;
            border-radius: 8px;
            background: #27272a;
        }
        .status.connected {
            border-left: 4px solid #22c55e;
        }
        .status.disconnected {
            border-left: 4px solid #ef4444;
        }
        .speaking-indicator {
            display: inline-block;
            margin-left: 10px;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            background: #7c3aed;
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .conversation {
            background: #27272a;
            border-radius: 12px;
            padding: 20px;
            min-height: 400px;
            max-height: 600px;
            overflow-y: auto;
        }
        .message {
            margin-bottom: 16px;
            padding: 12px 16px;
            border-radius: 12px;
            max-width: 85%;
            animation: fadeIn 0.3s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .message.alice {
            background: linear-gradient(135deg, #7c3aed 0%, #a78bfa 100%);
            margin-right: auto;
        }
        .message.bob {
            background: linear-gradient(135deg, #0891b2 0%, #22d3d1 100%);
            margin-left: auto;
        }
        .message.system {
            background: #3f3f46;
            margin: 0 auto;
            text-align: center;
            font-style: italic;
        }
        .message.speaking {
            box-shadow: 0 0 20px rgba(167, 139, 250, 0.5);
        }
        .message .speaker {
            font-weight: 700;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 4px;
            opacity: 0.9;
        }
        .message .text {
            line-height: 1.5;
        }
        .message .time {
            font-size: 11px;
            opacity: 0.7;
            margin-top: 6px;
        }
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #71717a;
        }
        .empty-state p {
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Bot-to-Bot Conversation</h1>

        <div class="controls">
            <input type="text" id="topic" placeholder="Enter conversation topic..." value="artificial intelligence">
            <button class="btn-start" onclick="startDemo()">Start</button>
            <button class="btn-stop" onclick="stopDemo()">Stop</button>
            <button class="btn-clear" onclick="clearMessages()">Clear</button>
            <div class="audio-toggle">
                <input type="checkbox" id="audioEnabled" checked>
                <label for="audioEnabled">Audio</label>
            </div>
        </div>

        <div id="status" class="status disconnected">
            Disconnected
        </div>

        <div id="conversation" class="conversation">
            <div class="empty-state">
                <p>No conversation yet.</p>
                <p>Enter a topic and click Start to begin.</p>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        let audioContext = null;
        let isPlaying = false;
        const conversationDiv = document.getElementById('conversation');
        const statusDiv = document.getElementById('status');

        // Initialize audio context on user interaction
        function initAudio() {
            if (!audioContext) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)({
                    sampleRate: 24000
                });
            }
            if (audioContext.state === 'suspended') {
                audioContext.resume();
            }
        }

        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/demo/viewer/ws`);

            ws.onopen = () => {
                statusDiv.innerHTML = 'Connected - Watching conversation';
                statusDiv.className = 'status connected';
            };

            ws.onclose = () => {
                statusDiv.innerHTML = 'Disconnected - Reconnecting...';
                statusDiv.className = 'status disconnected';
                setTimeout(connect, 2000);
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);

                if (data.type === 'history') {
                    if (data.data && data.data.length > 0) {
                        conversationDiv.innerHTML = '';
                        data.data.forEach(msg => addMessage(msg));
                    }
                } else if (data.type === 'message') {
                    addMessage(data.data);
                } else if (data.type === 'audio') {
                    handleAudio(data.data);
                } else if (data.type === 'ping') {
                    // Keepalive
                }
            };
        }

        function addMessage(msg) {
            const emptyState = conversationDiv.querySelector('.empty-state');
            if (emptyState) {
                emptyState.remove();
            }

            const messageDiv = document.createElement('div');
            const speakerLower = msg.speaker.toLowerCase();
            const speakerClass = speakerLower === 'alice' ? 'alice' :
                                 speakerLower === 'bob' ? 'bob' : 'system';
            messageDiv.className = `message ${speakerClass}`;
            messageDiv.id = `msg-${Date.now()}`;

            const time = new Date(msg.timestamp).toLocaleTimeString();

            messageDiv.innerHTML = `
                <div class="speaker">${msg.speaker}</div>
                <div class="text">${msg.text}</div>
                <div class="time">${time}</div>
            `;

            conversationDiv.appendChild(messageDiv);
            conversationDiv.scrollTop = conversationDiv.scrollHeight;

            return messageDiv.id;
        }

        // Track scheduled audio end time for overlap calculations
        let scheduledEndTime = 0;
        let pendingAudioCount = 0;

        function handleAudio(audioData) {
            const audioEnabled = document.getElementById('audioEnabled').checked;
            if (!audioEnabled) return;

            try {
                initAudio();
                if (!audioContext) {
                    console.error('AudioContext not available');
                    return;
                }

                // Decode base64 audio
                const binaryString = atob(audioData.audio);
                if (binaryString.length === 0) {
                    console.warn('Empty audio data received');
                    return;
                }

                const bytes = new Uint8Array(binaryString.length);
                for (let i = 0; i < binaryString.length; i++) {
                    bytes[i] = binaryString.charCodeAt(i);
                }

                // Add variance to overlap for natural feel (+/- 100ms)
                const variance = (Math.random() - 0.5) * 200;
                const overlapMs = (audioData.overlap_ms || 0) + variance;

                console.log(`Received audio #${audioData.sequence}: ${audioData.speaker} (${audioData.energy}, overlap: ${Math.round(overlapMs)}ms)`);

                // Schedule directly - Web Audio API handles timing
                scheduleAudio({
                    data: bytes,
                    speaker: audioData.speaker,
                    sampleRate: audioData.sample_rate || 24000,
                    pace: audioData.pace || 0.5,
                    energy: audioData.energy || 'normal',
                    overlapMs: overlapMs
                });
            } catch (error) {
                console.error('Error handling audio:', error);
            }
        }

        function scheduleAudio(audio) {
            const speakerLower = audio.speaker.toLowerCase();
            pendingAudioCount++;

            try {
                // Ensure buffer is valid for Int16Array (must be even length)
                if (audio.data.buffer.byteLength % 2 !== 0) {
                    console.warn('Audio data not aligned for Int16Array, skipping');
                    pendingAudioCount--;
                    return;
                }

                // Convert PCM to AudioBuffer
                const pcmData = new Int16Array(audio.data.buffer);
                if (pcmData.length === 0) {
                    console.warn('Empty PCM data');
                    pendingAudioCount--;
                    return;
                }

                const floatData = new Float32Array(pcmData.length);
                for (let i = 0; i < pcmData.length; i++) {
                    floatData[i] = pcmData[i] / 32768.0;
                }

                const audioBuffer = audioContext.createBuffer(1, floatData.length, audio.sampleRate);
                audioBuffer.getChannelData(0).set(floatData);

                const duration = audioBuffer.duration;
                const now = audioContext.currentTime;

                // Calculate start time based on overlap
                // Negative overlapMs = start BEFORE previous ends (talking over)
                // Positive overlapMs = start AFTER previous ends (gap)
                let startTime;
                if (scheduledEndTime > now) {
                    // Previous audio still playing/scheduled
                    startTime = scheduledEndTime + (audio.overlapMs / 1000);
                    startTime = Math.max(now + 0.05, startTime);
                } else {
                    startTime = now + 0.05;
                }

                // Create gain node for crossfade
                const gainNode = audioContext.createGain();
                gainNode.connect(audioContext.destination);

                // If overlapping (negative overlapMs), start quieter and fade in
                if (audio.overlapMs < 0 && startTime < scheduledEndTime) {
                    gainNode.gain.setValueAtTime(0.6, startTime);
                    gainNode.gain.linearRampToValueAtTime(1.0, startTime + 0.2);
                } else {
                    gainNode.gain.setValueAtTime(1.0, startTime);
                }

                const source = audioContext.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(gainNode);

                // Update scheduled end time
                scheduledEndTime = startTime + duration;
                isPlaying = true;

                // Update UI
                updateSpeakingStatus(audio.speaker, audio.energy, audio.pace);
                const messages = document.querySelectorAll(`.message.${speakerLower}`);
                if (messages.length > 0) {
                    messages[messages.length - 1].classList.add('speaking');
                }

                source.onended = () => {
                    pendingAudioCount = Math.max(0, pendingAudioCount - 1);
                    document.querySelectorAll(`.message.${speakerLower}.speaking`).forEach(el => {
                        el.classList.remove('speaking');
                    });
                    if (audioContext.currentTime >= scheduledEndTime - 0.1) {
                        isPlaying = false;
                        updateStatusIdle();
                    }
                };

                source.start(startTime);
                console.log(`Scheduled ${audio.speaker}: start=${startTime.toFixed(2)}s, duration=${duration.toFixed(2)}s, overlap=${Math.round(audio.overlapMs)}ms, pending=${pendingAudioCount}`);

            } catch (error) {
                pendingAudioCount = Math.max(0, pendingAudioCount - 1);
                console.error('Error scheduling audio:', error);
            }
        }

        function updateSpeakingStatus(speaker, energy, pace) {
            const energyColors = {
                'calm': '#22c55e',
                'normal': '#3b82f6',
                'energetic': '#f59e0b',
                'heated': '#ef4444'
            };
            const color = energyColors[energy] || energyColors['normal'];
            statusDiv.innerHTML = `Connected - <span class="speaking-indicator" style="background: ${color}">${speaker} (${energy})</span>`;
        }

        function updateStatusIdle() {
            statusDiv.innerHTML = 'Connected - Watching conversation';
            statusDiv.className = 'status connected';
        }

        async function startDemo() {
            initAudio();
            // Reset audio state
            scheduledEndTime = 0;
            isPlaying = false;
            pendingAudioCount = 0;
            const topic = document.getElementById('topic').value || 'artificial intelligence';
            try {
                const response = await fetch(`/demo/start?topic=${encodeURIComponent(topic)}`, {
                    method: 'POST'
                });
                const data = await response.json();
                if (response.ok) {
                    statusDiv.innerHTML = `Started - Topic: ${topic}`;
                } else {
                    alert(data.detail || 'Failed to start demo');
                }
            } catch (error) {
                console.error('Error starting demo:', error);
                alert('Failed to start demo');
            }
        }

        async function stopDemo() {
            scheduledEndTime = 0;
            isPlaying = false;
            pendingAudioCount = 0;
            try {
                const response = await fetch('/demo/stop', { method: 'POST' });
                const data = await response.json();
                if (response.ok) {
                    statusDiv.innerHTML = 'Stopped';
                } else {
                    alert(data.detail || 'Failed to stop demo');
                }
            } catch (error) {
                console.error('Error stopping demo:', error);
                alert('Failed to stop demo');
            }
        }

        function clearMessages() {
            scheduledEndTime = 0;
            isPlaying = false;
            pendingAudioCount = 0;
            conversationDiv.innerHTML = `
                <div class="empty-state">
                    <p>No conversation yet.</p>
                    <p>Enter a topic and click Start to begin.</p>
                </div>
            `;
        }

        // Connect on page load
        connect();
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


# ========== PIPECAT BOT-TO-BOT DEMO ==========

# Global Pipecat demo service instance
pipecat_demo_service = PipecatDualBotService(max_turns=100, turn_delay_ms=300)  # High limit - conversation should end naturally


@app.get("/pipecat-demo/personas")
async def get_pipecat_personas():
    """Get list of available personas for Pipecat demo."""
    return pipecat_list_personas()


@app.post("/pipecat-demo/start")
async def start_pipecat_demo(
    topic: str = Query(default="", description="Conversation topic (optional)"),
    alice: str = Query(default=None, description="Alice persona filename"),
    bob: str = Query(default=None, description="Bob persona filename")
):
    """Start a Pipecat-based bot-to-bot conversation demo."""
    success = await pipecat_demo_service.start(
        topic=topic,
        alice_persona=alice,
        bob_persona=bob
    )
    if not success:
        raise HTTPException(400, "Pipecat demo already running or failed to start")

    state = pipecat_demo_service.get_state()
    return {
        "status": "started",
        "implementation": "pipecat",
        "alice_persona": state.get("alice_persona"),
        "bob_persona": state.get("bob_persona"),
        "topic": state.get("topic"),
        "message": "Pipecat bot conversation started"
    }


@app.post("/pipecat-demo/stop")
async def stop_pipecat_demo():
    """Stop the Pipecat bot-to-bot conversation demo."""
    success = await pipecat_demo_service.stop()
    if not success:
        raise HTTPException(400, "No Pipecat demo running")
    return {
        "status": "stopped",
        "message": "Pipecat bot conversation stopped"
    }


@app.get("/pipecat-demo/status")
async def get_pipecat_demo_status():
    """Get current Pipecat demo status."""
    state = pipecat_demo_service.get_state()
    state["implementation"] = "pipecat"
    return state


@app.websocket("/pipecat-demo/viewer/ws")
async def pipecat_demo_viewer_websocket(websocket: WebSocket):
    """WebSocket endpoint for viewing the Pipecat bot-to-bot conversation."""
    await websocket.accept()
    viewer_queue = pipecat_demo_service.register_viewer()

    try:
        # Send current conversation history on connect
        state = pipecat_demo_service.get_state()
        await websocket.send_json({
            "type": "history",
            "data": state.get("conversation_history", [])
        })

        # Stream new messages
        while True:
            try:
                message = await asyncio.wait_for(viewer_queue.get(), timeout=30.0)
                await websocket.send_json(message)
            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_json({"type": "ping"})
            except Exception as e:
                logger.error(f"Error sending to Pipecat viewer: {e}")
                break
    except Exception as e:
        logger.error(f"Pipecat viewer WebSocket error: {e}")
    finally:
        pipecat_demo_service.unregister_viewer(viewer_queue)
        try:
            await websocket.close()
        except:
            pass


@app.get("/pipecat-demo/viewer", response_class=HTMLResponse)
async def pipecat_demo_viewer_page():
    """Serve the Pipecat demo viewer HTML page."""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pipecat Bot-to-Bot Conversation</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            min-height: 100vh;
            color: #e4e4e7;
            padding: 20px;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            margin-bottom: 10px;
            color: #f472b6;
        }
        .badge {
            text-align: center;
            margin-bottom: 20px;
        }
        .badge span {
            background: linear-gradient(135deg, #ec4899 0%, #f472b6 100%);
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }
        .persona-selectors {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 16px;
        }
        .persona-group {
            background: #27272a;
            border-radius: 8px;
            padding: 12px;
        }
        .persona-group label {
            display: block;
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
            color: #a1a1aa;
        }
        .persona-group.agent label {
            color: #f472b6;
        }
        .persona-group.customer label {
            color: #22d3d1;
        }
        select {
            width: 100%;
            padding: 10px 12px;
            border: 2px solid #3f3f46;
            border-radius: 6px;
            background: #18181b;
            color: #e4e4e7;
            font-size: 14px;
            cursor: pointer;
        }
        select:focus {
            outline: none;
            border-color: #f472b6;
        }
        .controls {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
            justify-content: center;
            align-items: center;
        }
        input[type="text"] {
            padding: 12px 16px;
            border: 2px solid #3f3f46;
            border-radius: 8px;
            background: #27272a;
            color: #e4e4e7;
            font-size: 16px;
            flex: 1;
            min-width: 200px;
        }
        input[type="text"]:focus {
            outline: none;
            border-color: #f472b6;
        }
        button {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn-start {
            background: linear-gradient(135deg, #ec4899 0%, #f472b6 100%);
            color: white;
        }
        .btn-start:hover {
            opacity: 0.9;
        }
        .btn-stop {
            background: #ef4444;
            color: white;
        }
        .btn-stop:hover {
            background: #dc2626;
        }
        .btn-clear {
            background: #3f3f46;
            color: #e4e4e7;
        }
        .btn-clear:hover {
            background: #52525b;
        }
        .status {
            text-align: center;
            margin-bottom: 20px;
            padding: 10px;
            border-radius: 8px;
            background: #27272a;
        }
        .status.connected {
            border-left: 4px solid #22c55e;
        }
        .status.disconnected {
            border-left: 4px solid #ef4444;
        }
        .speaking-indicator {
            display: inline-block;
            margin-left: 10px;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            background: #ec4899;
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .conversation {
            background: #27272a;
            border-radius: 12px;
            padding: 20px;
            min-height: 400px;
            max-height: 600px;
            overflow-y: auto;
        }
        .message {
            margin-bottom: 16px;
            padding: 12px 16px;
            border-radius: 12px;
            max-width: 85%;
            animation: fadeIn 0.3s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .message.alice {
            background: linear-gradient(135deg, #ec4899 0%, #f472b6 100%);
            margin-right: auto;
        }
        .message.bob {
            background: linear-gradient(135deg, #0891b2 0%, #22d3d1 100%);
            margin-left: auto;
        }
        .message.system {
            background: #3f3f46;
            margin: 0 auto;
            text-align: center;
            font-style: italic;
        }
        .message .speaker {
            font-weight: 700;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 4px;
            opacity: 0.9;
        }
        .message .text {
            line-height: 1.5;
        }
        .message .meta {
            font-size: 11px;
            opacity: 0.7;
            margin-top: 6px;
            display: flex;
            justify-content: space-between;
        }
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #71717a;
        }
        .empty-state p {
            margin-bottom: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Pipecat Bot-to-Bot</h1>
        <div class="badge"><span>Pipecat Architecture</span></div>

        <div class="persona-selectors">
            <div class="persona-group agent">
                <label>Agent (Industry)</label>
                <select id="alicePersona">
                    <option value="">Loading...</option>
                </select>
            </div>
            <div class="persona-group customer">
                <label>Customer Type</label>
                <select id="bobPersona">
                    <option value="">Loading...</option>
                </select>
            </div>
        </div>

        <div class="controls">
            <input type="text" id="topic" placeholder="Enter conversation topic (optional)..." value="">
            <button class="btn-start" onclick="startDemo()">Start</button>
            <button class="btn-stop" onclick="stopDemo()">Stop</button>
            <button class="btn-clear" onclick="clearMessages()">Clear</button>
        </div>

        <div id="status" class="status disconnected">
            Disconnected
        </div>

        <div id="conversation" class="conversation">
            <div class="empty-state">
                <p>No conversation yet.</p>
                <p>Select personas and click Start to begin.</p>
            </div>
        </div>
    </div>

    <script>
        let ws = null;
        const conversationDiv = document.getElementById('conversation');
        const statusDiv = document.getElementById('status');

        // Format persona filename into readable label
        function formatPersonaLabel(filename) {
            // Remove .json extension and prefix (alice_ or bob_)
            let name = filename.replace('.json', '').replace(/^(alice_|bob_)/, '');
            // Convert underscores to spaces and capitalize words
            return name.split('_').map(word =>
                word.charAt(0).toUpperCase() + word.slice(1)
            ).join(' ');
        }

        // Load available personas on page load
        async function loadPersonas() {
            try {
                const response = await fetch('/pipecat-demo/personas');
                const personas = await response.json();

                const aliceSelect = document.getElementById('alicePersona');
                const bobSelect = document.getElementById('bobPersona');

                // Populate Alice (Agent) dropdown
                aliceSelect.innerHTML = '';
                personas.alice.forEach((filename, index) => {
                    const option = document.createElement('option');
                    option.value = filename + '.json';
                    option.textContent = formatPersonaLabel(filename);
                    if (index === 0) option.selected = true;
                    aliceSelect.appendChild(option);
                });

                // Populate Bob (Customer) dropdown
                bobSelect.innerHTML = '';
                personas.bob.forEach((filename, index) => {
                    const option = document.createElement('option');
                    option.value = filename + '.json';
                    option.textContent = formatPersonaLabel(filename);
                    if (index === 0) option.selected = true;
                    bobSelect.appendChild(option);
                });

            } catch (error) {
                console.error('Error loading personas:', error);
            }
        }

        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/pipecat-demo/viewer/ws`);

            ws.onopen = () => {
                statusDiv.innerHTML = 'Connected - Watching Pipecat conversation';
                statusDiv.className = 'status connected';
            };

            ws.onclose = () => {
                statusDiv.innerHTML = 'Disconnected - Reconnecting...';
                statusDiv.className = 'status disconnected';
                setTimeout(connect, 2000);
            };

            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };

            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);

                if (data.type === 'history') {
                    if (data.data && data.data.length > 0) {
                        conversationDiv.innerHTML = '';
                        data.data.forEach(msg => addMessage(msg));
                    }
                } else if (data.type === 'message') {
                    addMessage(data.data);
                } else if (data.type === 'ping') {
                    // Keepalive
                }
            };
        }

        function addMessage(msg) {
            const emptyState = conversationDiv.querySelector('.empty-state');
            if (emptyState) {
                emptyState.remove();
            }

            const messageDiv = document.createElement('div');
            const speakerLower = msg.speaker.toLowerCase();
            const speakerClass = speakerLower === 'alice' ? 'alice' :
                                 speakerLower === 'bob' ? 'bob' : 'system';
            messageDiv.className = `message ${speakerClass}`;

            const time = new Date(msg.timestamp).toLocaleTimeString();
            const energy = msg.energy || 'normal';

            messageDiv.innerHTML = `
                <div class="speaker">${msg.speaker}</div>
                <div class="text">${msg.text}</div>
                <div class="meta">
                    <span>${time}</span>
                    <span>${energy}</span>
                </div>
            `;

            conversationDiv.appendChild(messageDiv);
            conversationDiv.scrollTop = conversationDiv.scrollHeight;
        }

        async function startDemo() {
            const topic = document.getElementById('topic').value;
            const alice = document.getElementById('alicePersona').value;
            const bob = document.getElementById('bobPersona').value;

            // Build query string
            const params = new URLSearchParams();
            if (topic) params.append('topic', topic);
            if (alice) params.append('alice', alice);
            if (bob) params.append('bob', bob);

            try {
                const response = await fetch(`/pipecat-demo/start?${params.toString()}`, {
                    method: 'POST'
                });
                const data = await response.json();
                if (response.ok) {
                    const aliceLabel = formatPersonaLabel(alice);
                    const bobLabel = formatPersonaLabel(bob);
                    statusDiv.innerHTML = `Started - ${aliceLabel} vs ${bobLabel}`;
                } else {
                    alert(data.detail || 'Failed to start Pipecat demo');
                }
            } catch (error) {
                console.error('Error starting Pipecat demo:', error);
                alert('Failed to start Pipecat demo');
            }
        }

        async function stopDemo() {
            try {
                const response = await fetch('/pipecat-demo/stop', { method: 'POST' });
                const data = await response.json();
                if (response.ok) {
                    statusDiv.innerHTML = 'Stopped';
                } else {
                    alert(data.detail || 'Failed to stop Pipecat demo');
                }
            } catch (error) {
                console.error('Error stopping Pipecat demo:', error);
                alert('Failed to stop Pipecat demo');
            }
        }

        function clearMessages() {
            conversationDiv.innerHTML = `
                <div class="empty-state">
                    <p>No conversation yet.</p>
                    <p>Select personas and click Start to begin.</p>
                </div>
            `;
        }

        // Initialize on page load
        loadPersonas();
        connect();
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)