from fastapi import FastAPI, Depends, WebSocket, HTTPException, Form, File, UploadFile, Request, Query, BackgroundTasks
from fastapi.responses import Response, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
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
from voice_agent import run_voice_bot, run_human_voice_demo
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
from datetime import datetime
import json

# Bot demo imports
from bot_demo import DualBotService, list_personas
from bot_demo.persona_loader import load_persona as load_bot_persona

# Pipecat bot demo imports
from bot_demo_pipecat import PipecatDualBotService, list_personas as pipecat_list_personas

# Daily bot demo imports (requires Linux/WSL - daily-python not available on Windows)
from bot_demo_daily import DailyBotService, list_personas as daily_list_personas

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

# Mount static files for the voice client
VOICE_CLIENT_DIST = Path(__file__).parent / "client" / "outrival-voice-client" / "dist" / "assets"
if VOICE_CLIENT_DIST.exists():
    app.mount("/human-demo/assets", StaticFiles(directory=VOICE_CLIENT_DIST), name="human-demo-assets")

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


@app.get("/demo/viewer")
async def demo_viewer_page():
    """Serve the demo viewer HTML page with no-cache headers."""
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
    return Response(
        content=html_content,
        media_type="text/html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )


# =============================================================================
# PIPECAT BOT-TO-BOT DEMO API
# =============================================================================
#
# This section provides REST API endpoints and WebSocket connections for the
# Pipecat-based bot-to-bot conversation demo.
#
# ENDPOINTS OVERVIEW:
# -------------------
# GET  /pipecat-demo/personas     - List available persona files
# POST /pipecat-demo/start        - Start a new conversation
# POST /pipecat-demo/stop         - Stop the current conversation
# GET  /pipecat-demo/status       - Get conversation state
# WS   /pipecat-demo/viewer/ws    - WebSocket for real-time updates
# GET  /pipecat-demo/viewer       - HTML viewer page
#
# TYPICAL FLOW:
# -------------
# 1. Browser opens /pipecat-demo/viewer (HTML page)
# 2. Page fetches /pipecat-demo/personas to populate dropdowns
# 3. Page connects to /pipecat-demo/viewer/ws WebSocket
# 4. User clicks Start -> POST /pipecat-demo/start
# 5. WebSocket receives messages and audio in real-time
# 6. User clicks Stop -> POST /pipecat-demo/stop (or conversation ends naturally)
#
# IMPLEMENTATION:
# ---------------
# All business logic is in PipecatDualBotService (dual_bot_service.py).
# These endpoints are thin wrappers that delegate to the service.
#
# =============================================================================

# Global service instance - handles one conversation at a time
# max_turns=100 is a safety limit; conversations should end naturally via farewell detection
pipecat_demo_service = PipecatDualBotService(max_turns=100, turn_delay_ms=300)


# -----------------------------------------------------------------------------
# GET /pipecat-demo/personas - List available persona files
# -----------------------------------------------------------------------------
@app.get("/pipecat-demo/personas")
async def get_pipecat_personas():
    """
    Get list of available personas for the Pipecat demo.

    Returns JSON with alice and bob persona lists:
    {
        "alice": ["alice_bank_teller", "alice_insurance_agent", ...],
        "bob": ["bob_bank_upset_customer", "bob_insurance_frustrated_claimant", ...]
    }

    Used by the viewer page to populate the persona dropdown selectors.
    """
    return pipecat_list_personas()


# -----------------------------------------------------------------------------
# POST /pipecat-demo/start - Start a new conversation
# -----------------------------------------------------------------------------
@app.post("/pipecat-demo/start")
async def start_pipecat_demo(
    topic: str = Query(default="", description="Conversation topic (optional)"),
    alice: str = Query(default=None, description="Alice persona filename (e.g., 'alice_bank_teller.json')"),
    bob: str = Query(default=None, description="Bob persona filename (e.g., 'bob_bank_upset_customer.json')"),
    enable_audio: bool = Query(default=False, description="Enable ElevenLabs TTS audio generation")
):
    """
    Start a Pipecat-based bot-to-bot conversation.

    Query Parameters:
        topic: Optional topic to seed the conversation
        alice: Alice persona filename (defaults to alice_insurance_agent.json)
        bob: Bob persona filename (defaults to bob_insurance_frustrated_claimant.json)
        enable_audio: If true, generates TTS audio via ElevenLabs (requires API key)

    Returns:
        JSON with status, selected personas, and audio_enabled flag

    Errors:
        400: If a conversation is already running or failed to start

    Example:
        POST /pipecat-demo/start?alice=alice_bank_teller.json&bob=bob_bank_upset_customer.json&enable_audio=true
    """
    success = await pipecat_demo_service.start(
        topic=topic,
        alice_persona=alice,
        bob_persona=bob,
        enable_audio=enable_audio
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
        "audio_enabled": state.get("audio_enabled"),
        "message": "Pipecat bot conversation started"
    }


# -----------------------------------------------------------------------------
# POST /pipecat-demo/stop - Stop the current conversation
# -----------------------------------------------------------------------------
@app.post("/pipecat-demo/stop")
async def stop_pipecat_demo():
    """
    Stop the current Pipecat bot-to-bot conversation.

    This triggers a graceful shutdown:
    1. Stops the conversation orchestrator
    2. Waits for TTS workers to finish processing queued audio
    3. Cleans up resources

    Returns:
        JSON with status: "stopped"

    Errors:
        400: If no conversation is currently running
    """
    success = await pipecat_demo_service.stop()
    if not success:
        raise HTTPException(400, "No Pipecat demo running")
    return {
        "status": "stopped",
        "message": "Pipecat bot conversation stopped"
    }


# -----------------------------------------------------------------------------
# GET /pipecat-demo/status - Get current conversation state
# -----------------------------------------------------------------------------
@app.get("/pipecat-demo/status")
async def get_pipecat_demo_status():
    """
    Get the current state of the Pipecat demo.

    Returns JSON with:
        is_running: bool - Whether a conversation is active
        turn_count: int - Number of turns completed
        conversation_history: list - All messages in the conversation
        audio_enabled: bool - Whether TTS is active
        tts_queue_size: int - Number of pending TTS jobs
        alice_persona: str - Current Alice persona filename
        bob_persona: str - Current Bob persona filename
        topic: str - Conversation topic (if set)

    Useful for polling status or debugging.
    """
    state = pipecat_demo_service.get_state()
    state["implementation"] = "pipecat"
    return state


# -----------------------------------------------------------------------------
# WebSocket /pipecat-demo/viewer/ws - Real-time conversation updates
# -----------------------------------------------------------------------------
@app.websocket("/pipecat-demo/viewer/ws")
async def pipecat_demo_viewer_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for viewing the Pipecat bot-to-bot conversation in real-time.

    CONNECTION FLOW:
    ----------------
    1. Client connects, server accepts
    2. Server immediately sends conversation history (type: "history")
    3. Server streams messages as they occur (type: "message", "audio", "ping")
    4. On disconnect, client is unregistered

    MESSAGE TYPES SENT TO CLIENT:
    -----------------------------
    - {"type": "history", "data": [...]} - Full conversation history on connect
    - {"type": "message", "data": {...}} - New text message from Alice or Bob
    - {"type": "audio", "data": {...}}   - Base64-encoded audio data for playback
    - {"type": "ping"}                   - Keepalive (every 30 seconds of inactivity)

    AUDIO DATA FORMAT:
    ------------------
    {
        "type": "audio",
        "data": {
            "speaker": "Alice" | "Bob",
            "audio": "<base64-encoded PCM>",
            "format": "pcm",
            "sample_rate": 24000,
            "sequence": 1,           // For ordering playback
            "pace": 0.5,             // Speech pace indicator
            "energy": "normal",      // Energy level
            "overlap_ms": 200        // Suggested overlap with previous
        }
    }
    """
    await websocket.accept()

    # Register this viewer to receive broadcasts
    viewer_queue = pipecat_demo_service.register_viewer()

    try:
        # Send current conversation history immediately on connect
        # This allows late-joining viewers to see what's already happened
        state = pipecat_demo_service.get_state()
        await websocket.send_json({
            "type": "history",
            "data": state.get("conversation_history", [])
        })

        # Main loop: stream new messages to this viewer
        while True:
            try:
                # Wait for next message from the service (with timeout for keepalive)
                message = await asyncio.wait_for(viewer_queue.get(), timeout=30.0)
                await websocket.send_json(message)
            except asyncio.TimeoutError:
                # No messages for 30 seconds - send keepalive to prevent disconnect
                await websocket.send_json({"type": "ping"})
            except Exception as e:
                logger.error(f"Error sending to Pipecat viewer: {e}")
                break
    except Exception as e:
        logger.error(f"Pipecat viewer WebSocket error: {e}")
    finally:
        # Clean up: unregister viewer and close connection
        pipecat_demo_service.unregister_viewer(viewer_queue)
        try:
            await websocket.close()
        except:
            pass


# -----------------------------------------------------------------------------
# GET /pipecat-demo/viewer - HTML Viewer Page
# -----------------------------------------------------------------------------
# The viewer page is embedded below as an HTML string. It provides:
# - Persona selection dropdowns (Alice and Bob)
# - Start/Stop/Clear buttons
# - Real-time conversation display
# - Audio playback using Web Audio API
# - Visual indicators for who is speaking
#
# For details on the viewer implementation, see API_REFERENCE.md
# -----------------------------------------------------------------------------
@app.get("/pipecat-demo/viewer")
async def pipecat_demo_viewer_page():
    """Serve the Pipecat demo viewer HTML page with no-cache headers."""
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
            <div class="audio-toggle">
                <input type="checkbox" id="audioEnabled">
                <label for="audioEnabled">Audio</label>
            </div>
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
        let audioContext = null;
        let scheduledEndTime = 0;
        let pendingAudioCount = 0;
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
                } else if (data.type === 'audio') {
                    handleAudio(data.data);
                } else if (data.type === 'ping') {
                    // Keepalive
                }
            };
        }

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
            pendingAudioCount++;

            try {
                // Ensure buffer is valid for Int16Array
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

                const audioBuffer = audioContext.createBuffer(1, pcmData.length, audio.sampleRate);
                const channelData = audioBuffer.getChannelData(0);

                // Convert Int16 to Float32
                for (let i = 0; i < pcmData.length; i++) {
                    channelData[i] = pcmData[i] / 32768.0;
                }

                // Create buffer source
                const source = audioContext.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(audioContext.destination);

                // Calculate when to start (with overlap)
                const now = audioContext.currentTime;
                const audioDuration = audioBuffer.duration;

                let startTime;
                if (scheduledEndTime > now) {
                    // Previous audio still scheduled - apply overlap
                    startTime = Math.max(now, scheduledEndTime + (audio.overlapMs / 1000));
                } else {
                    // No pending audio, start immediately
                    startTime = now;
                }

                source.start(startTime);
                scheduledEndTime = startTime + audioDuration;

                console.log(`Scheduled audio for ${audio.speaker}: start=${startTime.toFixed(2)}s, duration=${audioDuration.toFixed(2)}s`);

                source.onended = () => {
                    pendingAudioCount--;
                };

            } catch (error) {
                console.error('Error scheduling audio:', error);
                pendingAudioCount--;
            }
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
            const enableAudio = document.getElementById('audioEnabled').checked;

            // Initialize audio context if audio is enabled (requires user interaction)
            if (enableAudio) {
                initAudio();
            }

            // Build query string
            const params = new URLSearchParams();
            if (topic) params.append('topic', topic);
            if (alice) params.append('alice', alice);
            if (bob) params.append('bob', bob);
            params.append('enable_audio', enableAudio);

            try {
                const response = await fetch(`/pipecat-demo/start?${params.toString()}`, {
                    method: 'POST'
                });
                const data = await response.json();
                if (response.ok) {
                    const aliceLabel = formatPersonaLabel(alice);
                    const bobLabel = formatPersonaLabel(bob);
                    const audioStatus = enableAudio ? ' (with audio)' : '';
                    statusDiv.innerHTML = `Started - ${aliceLabel} vs ${bobLabel}${audioStatus}`;
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
    return Response(
        content=html_content,
        media_type="text/html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )


# ========== HUMAN-TO-AGENT VOICE DEMO ==========

@app.get("/human-demo/personas")
async def get_human_demo_personas():
    """Get list of available Alice personas for human demo."""
    personas = list_personas()  # Returns {"alice": [...], "bob": [...]}

    result = []
    for filename in personas.get("alice", []):
        try:
            persona = load_bot_persona(filename)
            result.append({
                "filename": filename,
                "name": persona.name,
                "role": persona.role,
                "company": persona.company_name,
            })
        except Exception as e:
            logger.warning(f"Failed to load persona {filename}: {e}")

    return {"personas": result}


@app.websocket("/human-demo/ws/{persona_name}")
async def human_demo_websocket(websocket: WebSocket, persona_name: str):
    """WebSocket endpoint for human-to-agent voice conversation."""
    await websocket.accept()

    try:
        # Load the persona
        persona = load_bot_persona(persona_name)
        if not persona:
            logger.error(f"Persona not found: {persona_name}")
            await websocket.close(code=1008, reason="Persona not found")
            return

        # Generate ephemeral session ID
        session_id = str(uuid4())

        logger.info(f"Human demo started: persona={persona.name}, session={session_id}")

        await run_human_voice_demo(
            websocket_client=websocket,
            persona=persona,
            session_id=session_id,
        )

    except FileNotFoundError as e:
        logger.error(f"Persona file not found: {persona_name}")
        try:
            await websocket.close(code=1008, reason="Persona not found")
        except:
            pass
    except Exception as e:
        logger.exception(f"Human demo WebSocket error: {e}")
    finally:
        try:
            if websocket.client_state.name != "DISCONNECTED":
                await websocket.close()
        except:
            pass
        logger.info(f"Human demo WebSocket closed for persona={persona_name}")


@app.get("/human-demo/viewer")
async def human_demo_viewer_page():
    """Serve the human-to-agent voice demo viewer HTML page."""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Talk to Alice - Voice Demo</title>
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
            max-width: 600px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            margin-bottom: 10px;
            color: #34d399;
        }
        .subtitle {
            text-align: center;
            margin-bottom: 30px;
            color: #a1a1aa;
            font-size: 14px;
        }
        .persona-selector {
            background: #27272a;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
        }
        .persona-selector label {
            display: block;
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 10px;
            color: #a1a1aa;
        }
        select {
            width: 100%;
            padding: 14px 16px;
            border: 2px solid #3f3f46;
            border-radius: 8px;
            background: #18181b;
            color: #e4e4e7;
            font-size: 16px;
            cursor: pointer;
        }
        select:focus {
            outline: none;
            border-color: #34d399;
        }
        .persona-info {
            margin-top: 12px;
            padding: 12px;
            background: #18181b;
            border-radius: 8px;
            font-size: 13px;
            color: #a1a1aa;
        }
        .persona-info .name {
            color: #34d399;
            font-weight: 600;
            font-size: 15px;
        }
        .controls {
            display: flex;
            gap: 12px;
            margin-bottom: 20px;
        }
        button {
            flex: 1;
            padding: 16px 24px;
            border: none;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn-connect {
            background: linear-gradient(135deg, #059669 0%, #34d399 100%);
            color: white;
        }
        .btn-connect:hover:not(:disabled) {
            opacity: 0.9;
            transform: translateY(-1px);
        }
        .btn-connect:disabled {
            background: #3f3f46;
            cursor: not-allowed;
        }
        .btn-disconnect {
            background: #ef4444;
            color: white;
        }
        .btn-disconnect:hover:not(:disabled) {
            background: #dc2626;
        }
        .btn-disconnect:disabled {
            background: #3f3f46;
            cursor: not-allowed;
        }
        .status-panel {
            background: #27272a;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
        }
        .status-indicator {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            padding: 12px 20px;
            border-radius: 24px;
            font-weight: 600;
            margin-bottom: 16px;
        }
        .status-indicator.disconnected {
            background: #3f3f46;
            color: #a1a1aa;
        }
        .status-indicator.connecting {
            background: #fbbf24;
            color: #1a1a2e;
        }
        .status-indicator.connected {
            background: #34d399;
            color: #1a1a2e;
        }
        .status-indicator.error {
            background: #ef4444;
            color: white;
        }
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: currentColor;
        }
        .status-indicator.connected .status-dot {
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.6; transform: scale(1.2); }
        }
        .mic-status {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            padding: 10px;
            border-radius: 8px;
            background: #18181b;
            font-size: 14px;
        }
        .mic-icon {
            font-size: 20px;
        }
        .mic-status.active {
            background: linear-gradient(135deg, #059669 0%, #34d399 100%);
            color: white;
        }
        .instructions {
            margin-top: 20px;
            padding: 16px;
            background: #18181b;
            border-radius: 8px;
            font-size: 13px;
            color: #71717a;
            line-height: 1.6;
        }
        .instructions h3 {
            color: #a1a1aa;
            margin-bottom: 8px;
            font-size: 14px;
        }
        .instructions ul {
            margin-left: 20px;
        }
        .instructions li {
            margin-bottom: 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Talk to Alice</h1>
        <p class="subtitle">Have a voice conversation with an AI customer service agent</p>

        <div class="persona-selector">
            <label>Select a persona:</label>
            <select id="personaSelect" onchange="updatePersonaInfo()">
                <option value="">Loading personas...</option>
            </select>
            <div id="personaInfo" class="persona-info" style="display: none;"></div>
        </div>

        <div class="controls">
            <button id="connectBtn" class="btn-connect" onclick="connect()">Connect</button>
            <button id="disconnectBtn" class="btn-disconnect" onclick="disconnect()" disabled>Disconnect</button>
        </div>

        <div class="status-panel">
            <div id="statusIndicator" class="status-indicator disconnected">
                <span class="status-dot"></span>
                <span id="statusText">Disconnected</span>
            </div>
            <div id="micStatus" class="mic-status">
                <span class="mic-icon">🎤</span>
                <span>Microphone inactive</span>
            </div>
        </div>

        <div class="instructions">
            <h3>How to use:</h3>
            <ul>
                <li>Select a persona from the dropdown</li>
                <li>Click "Connect" to start the conversation</li>
                <li>Allow microphone access when prompted</li>
                <li>Speak naturally - Alice will respond</li>
                <li>Say "goodbye" to end the conversation</li>
            </ul>
        </div>
    </div>

    <script type="module" src="/human-demo/assets/index-DOzFJ6E1.js"></script>
    <script type="module">
        // Wait for the bundle to load, then use its exports
        // The bundle exposes PipecatClient, WebSocketTransport, ProtobufFrameSerializer globally

        let client = null;
        let personas = [];
        let PipecatClient, WebSocketTransport, ProtobufFrameSerializer;

        // Dynamic import from the bundle - we need to wait for it
        async function initLibraries() {
            // Import from the bundled module
            const clientModule = await import('/human-demo/assets/index-DOzFJ6E1.js');
            // The bundle exports everything we need
            return clientModule;
        }

        // Load personas on page load
        async function loadPersonas() {
            try {
                const response = await fetch('/human-demo/personas');
                const data = await response.json();
                personas = data.personas;

                const select = document.getElementById('personaSelect');
                select.innerHTML = '';

                personas.forEach((persona, index) => {
                    const option = document.createElement('option');
                    option.value = persona.filename;
                    option.textContent = persona.name + ' - ' + persona.role;
                    if (index === 0) option.selected = true;
                    select.appendChild(option);
                });

                updatePersonaInfo();
            } catch (error) {
                console.error('Error loading personas:', error);
                document.getElementById('personaSelect').innerHTML = '<option value="">Error loading personas</option>';
            }
        }

        window.updatePersonaInfo = function() {
            const select = document.getElementById('personaSelect');
            const infoDiv = document.getElementById('personaInfo');
            const filename = select.value;

            const persona = personas.find(p => p.filename === filename);
            if (persona) {
                infoDiv.innerHTML = '<div class="name">' + persona.name + '</div>' +
                    '<div>' + persona.role + '</div>' +
                    '<div>' + persona.company + '</div>';
                infoDiv.style.display = 'block';
            } else {
                infoDiv.style.display = 'none';
            }
        }

        function setStatus(status, text) {
            const indicator = document.getElementById('statusIndicator');
            const statusText = document.getElementById('statusText');
            indicator.className = 'status-indicator ' + status;
            statusText.textContent = text;
        }

        function setMicStatus(active) {
            const micStatus = document.getElementById('micStatus');
            if (active) {
                micStatus.className = 'mic-status active';
                micStatus.innerHTML = '<span class="mic-icon">🎤</span><span>Microphone active - speak now</span>';
            } else {
                micStatus.className = 'mic-status';
                micStatus.innerHTML = '<span class="mic-icon">🎤</span><span>Microphone inactive</span>';
            }
        }

        function setButtons(connected) {
            document.getElementById('connectBtn').disabled = connected;
            document.getElementById('disconnectBtn').disabled = !connected;
            document.getElementById('personaSelect').disabled = connected;
        }

        function log(msg) {
            console.log(msg);
            // Also show in a log panel if it exists
            const logEl = document.getElementById('logPanel');
            if (logEl) {
                logEl.textContent += msg + '\\n';
                logEl.scrollTop = logEl.scrollHeight;
            }
        }

        window.connect = async function() {
            const personaFilename = document.getElementById('personaSelect').value;
            if (!personaFilename) {
                alert('Please select a persona');
                return;
            }

            setStatus('connecting', 'Connecting...');
            setButtons(true);

            try {
                // Dynamically import from the bundle
                const { PipecatClient, WebSocketTransport, ProtobufFrameSerializer } = await import('https://esm.run/@pipecat-ai/client-js@1.5.0');
                const wsTransport = await import('https://esm.run/@pipecat-ai/websocket-transport@1.5.0');

                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = protocol + '//' + window.location.host + '/human-demo/ws/' + personaFilename;

                log('Connecting to: ' + wsUrl);

                client = new PipecatClient({
                    transport: new wsTransport.WebSocketTransport({
                        serializer: new wsTransport.ProtobufFrameSerializer(),
                        recorderSampleRate: 16000,
                        playerSampleRate: 24000,  // Gemini Live native audio outputs at 24kHz
                    }),
                    enableCam: false,
                    enableMic: true,
                    callbacks: {
                        onBotConnected: () => {
                            setStatus('connected', 'Connected');
                            log('[bot] connected');
                        },
                        onBotReady: () => {
                            setStatus('connected', 'Connected - Alice is ready');
                            setMicStatus(true);
                            log('[bot] ready (start talking)');
                        },
                        onBotDisconnected: () => {
                            setStatus('disconnected', 'Disconnected');
                            setMicStatus(false);
                            setButtons(false);
                            client = null;
                            log('[bot] disconnected');
                        },
                        onDisconnected: () => {
                            setStatus('disconnected', 'Disconnected');
                            setMicStatus(false);
                            setButtons(false);
                            client = null;
                            log('[disconnected]');
                        },
                        onTransportStateChanged: (state) => {
                            log('[transport state] ' + state);
                            if (state === 'disconnected' || state === 'error') {
                                setStatus('disconnected', 'Disconnected');
                                setMicStatus(false);
                                setButtons(false);
                                client = null;
                            }
                        },
                        onUserTranscript: (t) => {
                            const text = t?.text ?? t?.transcript ?? '';
                            if (text) log('[you] ' + text);
                        },
                        onBotTranscript: (t) => {
                            const text = t?.text ?? t?.transcript ?? '';
                            if (text) log('[alice] ' + text);
                        },
                        onError: (e) => {
                            log('[error] ' + (e?.message ?? String(e)));
                            setStatus('error', 'Error: ' + (e?.message ?? 'Unknown error'));
                            setMicStatus(false);
                            setButtons(false);
                        },
                    },
                });

                await client.connect({ wsUrl });

            } catch (error) {
                console.error('Connection error:', error);
                log('[error] ' + error.message);
                setStatus('error', 'Failed to connect: ' + error.message);
                setMicStatus(false);
                setButtons(false);
                client = null;
            }
        }

        window.disconnect = async function() {
            if (client) {
                try {
                    await client.disconnect();
                } catch (error) {
                    console.error('Disconnect error:', error);
                }
                client = null;
            }
            setStatus('disconnected', 'Disconnected');
            setMicStatus(false);
            setButtons(false);
        }

        // Initialize on page load
        loadPersonas();
    </script>
</body>
</html>
    """
    return Response(
        content=html_content,
        media_type="text/html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )


# ========== EXPERIMENTAL: PERSONA TESTING ==========
# This section can be safely removed without affecting main functionality

from bot_demo_pipecat.persona_tester import (
    run_persona_test,
    list_test_results,
    load_test_result,
    find_customer_personas_for_agent,
)

@app.post("/persona-test/run/{agent_persona}")
async def run_persona_test_endpoint(agent_persona: str, background_tasks: BackgroundTasks):
    """
    Run automated tests for an agent against all matching customer personas.

    This runs in the background and saves results to JSON files.
    """
    # Validate agent exists
    try:
        from bot_demo_pipecat.persona_loader import load_persona
        load_persona(agent_persona)
    except FileNotFoundError:
        raise HTTPException(404, f"Agent persona not found: {agent_persona}")

    # Check for matching customers
    customers = find_customer_personas_for_agent(agent_persona)
    if not customers:
        raise HTTPException(400, f"No customer personas found for {agent_persona}")

    # Run test in background
    background_tasks.add_task(run_persona_test, agent_persona)

    return {
        "status": "started",
        "agent_persona": agent_persona,
        "customer_personas": customers,
        "message": f"Testing against {len(customers)} customer personas. Check /persona-test/results for output."
    }


@app.get("/persona-test/results")
async def list_persona_test_results():
    """List all available test result files."""
    return {"results": list_test_results()}


@app.get("/persona-test/results/{filename}")
async def get_persona_test_result(filename: str):
    """Get a specific test result."""
    result = load_test_result(filename)
    if not result:
        raise HTTPException(404, f"Result not found: {filename}")
    return result


@app.get("/persona-test/agents")
async def list_testable_agents():
    """List all agent personas that can be tested."""
    from bot_demo_pipecat.persona_loader import PERSONAS_DIR

    agents = []
    for file in PERSONAS_DIR.glob("alice_*.json"):
        customers = find_customer_personas_for_agent(file.name)
        agents.append({
            "filename": file.name,
            "customer_count": len(customers),
            "customers": customers
        })

    return {"agents": agents}


# =============================================================================
# DAILY BOT-TO-BOT DEMO API
# =============================================================================
#
# This section provides REST API endpoints and WebSocket connections for the
# Daily.co WebRTC-based bot-to-bot conversation demo.
#
# ARCHITECTURE:
# -------------
# Both Alice and Bob join a shared Daily room as participants. Each bot runs
# a full Pipecat pipeline with GeminiLiveLLMService. Conversation flows
# naturally through Daily's audio mixing - no explicit message passing needed.
#
# ENDPOINTS:
# ----------
# GET  /daily-demo/personas     - List available persona files
# POST /daily-demo/start        - Start a new conversation
# POST /daily-demo/stop         - Stop the current conversation
# GET  /daily-demo/status       - Get conversation state
# WS   /daily-demo/viewer/ws    - WebSocket for real-time transcript updates
# GET  /daily-demo/viewer       - HTML viewer page
#
# REQUIRED ENVIRONMENT VARIABLES:
# -------------------------------
# - DAILY_API_KEY: Daily.co API key for room management
# - GEMINI_API_KEY: For GeminiLiveLLMService
#
# =============================================================================

# Global service instance - handles one conversation at a time
daily_demo_service = DailyBotService()


# -----------------------------------------------------------------------------
# GET /daily-demo/personas - List available persona files
# -----------------------------------------------------------------------------
@app.get("/daily-demo/personas")
async def get_daily_personas():
    """
    Get list of available personas for the Daily demo.

    Returns JSON with alice and bob persona lists:
    {
        "alice": ["alice_bank_teller", "alice_insurance_agent", ...],
        "bob": ["bob_bank_upset_customer", "bob_insurance_frustrated_claimant", ...]
    }
    """
    return daily_list_personas()


# -----------------------------------------------------------------------------
# POST /daily-demo/start - Start a new conversation
# -----------------------------------------------------------------------------
@app.post("/daily-demo/start")
async def start_daily_demo(
    topic: str = Query(default="", description="Conversation topic (optional)"),
    alice: str = Query(default=None, description="Alice persona filename (e.g., 'alice_bank_teller.json')"),
    bob: str = Query(default=None, description="Bob persona filename (e.g., 'bob_bank_upset_customer.json')"),
):
    """
    Start a Daily WebRTC bot-to-bot conversation.

    This creates a Daily room, has both bots join via WebRTC, and starts the conversation.
    Alice speaks first, then natural turn-taking occurs through Daily's audio mixing.

    Query Parameters:
        topic: Optional topic to seed the conversation
        alice: Alice persona filename (defaults to alice_insurance_agent.json)
        bob: Bob persona filename (defaults to bob_insurance_frustrated_claimant.json)

    Returns:
        JSON with status, selected personas, and room URL

    Errors:
        400: If a conversation is already running or failed to start

    Example:
        POST /daily-demo/start?alice=alice_bank_teller.json&bob=bob_bank_upset_customer.json
    """
    success = await daily_demo_service.start(
        topic=topic,
        alice_persona=alice,
        bob_persona=bob,
    )
    if not success:
        raise HTTPException(400, "Daily demo already running or failed to start")

    state = daily_demo_service.get_state()
    return {
        "status": "started",
        "implementation": "daily",
        "alice_persona": state.get("alice_persona"),
        "bob_persona": state.get("bob_persona"),
        "topic": state.get("topic"),
        "room_url": state.get("room_url"),
        "message": "Daily WebRTC bot conversation started"
    }


# -----------------------------------------------------------------------------
# POST /daily-demo/stop - Stop the current conversation
# -----------------------------------------------------------------------------
@app.post("/daily-demo/stop")
async def stop_daily_demo():
    """
    Stop the current Daily bot-to-bot conversation.

    This shuts down both bot pipelines and deletes the Daily room.

    Returns:
        JSON with status: "stopped"

    Errors:
        400: If no conversation is currently running
    """
    success = await daily_demo_service.stop()
    if not success:
        raise HTTPException(400, "No Daily demo running")
    return {
        "status": "stopped",
        "message": "Daily bot conversation stopped"
    }


# -----------------------------------------------------------------------------
# GET /daily-demo/status - Get current conversation state
# -----------------------------------------------------------------------------
@app.get("/daily-demo/status")
async def get_daily_demo_status():
    """
    Get the current state of the Daily demo.

    Returns JSON with:
        is_running: bool - Whether a conversation is active
        conversation_history: list - All messages in the conversation
        alice_persona: str - Current Alice persona filename
        bob_persona: str - Current Bob persona filename
        topic: str - Conversation topic (if set)
        room_url: str - Daily.co room URL (if running)
    """
    state = daily_demo_service.get_state()
    state["implementation"] = "daily"
    return state


# -----------------------------------------------------------------------------
# WebSocket /daily-demo/viewer/ws - Real-time transcript updates
# -----------------------------------------------------------------------------
@app.websocket("/daily-demo/viewer/ws")
async def daily_demo_viewer_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for viewing the Daily bot-to-bot conversation in real-time.

    MESSAGE TYPES SENT TO CLIENT:
    -----------------------------
    - {"type": "history", "data": [...]} - Full conversation history on connect
    - {"type": "message", "data": {...}} - New text message from Alice or Bob
    - {"type": "ping"}                   - Keepalive (every 30 seconds of inactivity)
    """
    await websocket.accept()
    viewer_queue = daily_demo_service.register_viewer()

    try:
        # Send current conversation history on connect
        state = daily_demo_service.get_state()
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
                logger.error(f"Error sending to Daily viewer: {e}")
                break
    except Exception as e:
        logger.error(f"Daily viewer WebSocket error: {e}")
    finally:
        daily_demo_service.unregister_viewer(viewer_queue)
        try:
            await websocket.close()
        except:
            pass


# -----------------------------------------------------------------------------
# GET /daily-demo/viewer - HTML Viewer Page
# -----------------------------------------------------------------------------
@app.get("/daily-demo/viewer")
async def daily_demo_viewer_page():
    """Serve the Daily demo viewer HTML page."""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Daily Bot-to-Bot Conversation</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            min-height: 100vh;
            color: #e2e8f0;
            padding: 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            margin-bottom: 8px;
            color: #38bdf8;
        }
        .subtitle {
            text-align: center;
            color: #94a3b8;
            margin-bottom: 24px;
            font-size: 14px;
        }
        .persona-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 20px;
        }
        .persona-box {
            background: #1e293b;
            border-radius: 12px;
            padding: 16px;
            border: 1px solid #334155;
        }
        .persona-box label {
            display: block;
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 8px;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .persona-box.alice label {
            color: #a78bfa;
        }
        .persona-box.bob label {
            color: #2dd4bf;
        }
        select {
            width: 100%;
            padding: 12px;
            border: 2px solid #334155;
            border-radius: 8px;
            background: #0f172a;
            color: #e2e8f0;
            font-size: 14px;
            cursor: pointer;
        }
        select:focus {
            outline: none;
            border-color: #38bdf8;
        }
        .controls {
            display: flex;
            gap: 12px;
            margin-bottom: 20px;
            flex-wrap: wrap;
            align-items: center;
        }
        input[type="text"] {
            flex: 1;
            min-width: 200px;
            padding: 12px 16px;
            border: 2px solid #334155;
            border-radius: 8px;
            background: #0f172a;
            color: #e2e8f0;
            font-size: 14px;
        }
        input[type="text"]:focus {
            outline: none;
            border-color: #38bdf8;
        }
        button {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn-start {
            background: linear-gradient(135deg, #059669 0%, #10b981 100%);
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
            background: #334155;
            color: #e2e8f0;
        }
        .btn-clear:hover {
            background: #475569;
        }
        .status {
            text-align: center;
            margin-bottom: 20px;
            padding: 12px;
            border-radius: 8px;
            background: #1e293b;
            border: 1px solid #334155;
        }
        .status.connected {
            border-color: #10b981;
            color: #10b981;
        }
        .status.disconnected {
            border-color: #ef4444;
            color: #ef4444;
        }
        .room-info {
            text-align: center;
            margin-bottom: 20px;
            padding: 12px;
            border-radius: 8px;
            background: #1e293b;
            border: 1px solid #38bdf8;
            display: none;
        }
        .room-info.visible {
            display: block;
        }
        .room-info a {
            color: #38bdf8;
            text-decoration: none;
        }
        .room-info a:hover {
            text-decoration: underline;
        }
        .conversation {
            background: #1e293b;
            border-radius: 12px;
            padding: 20px;
            min-height: 400px;
            max-height: 600px;
            overflow-y: auto;
            border: 1px solid #334155;
        }
        .message {
            margin-bottom: 16px;
            padding: 14px 18px;
            border-radius: 12px;
            max-width: 80%;
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
            background: linear-gradient(135deg, #0d9488 0%, #2dd4bf 100%);
            margin-left: auto;
        }
        .message .speaker {
            font-weight: 700;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 4px;
            opacity: 0.9;
        }
        .message .text {
            line-height: 1.5;
            font-size: 15px;
        }
        .message .time {
            font-size: 11px;
            opacity: 0.7;
            margin-top: 6px;
        }
        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: #64748b;
        }
        .empty-state p {
            margin-bottom: 10px;
        }
        .badge {
            display: inline-block;
            padding: 4px 8px;
            background: #38bdf8;
            color: #0f172a;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            margin-left: 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Daily Bot-to-Bot<span class="badge">Daily</span></h1>
        <p class="subtitle">Real-time voice conversation through Daily.co WebRTC</p>

        <div class="persona-row">
            <div class="persona-box alice">
                <label>Agent (Alice)</label>
                <select id="alicePersona">
                    <option>Loading personas...</option>
                </select>
            </div>
            <div class="persona-box bob">
                <label>Customer (Bob)</label>
                <select id="bobPersona">
                    <option>Loading personas...</option>
                </select>
            </div>
        </div>

        <div class="controls">
            <input type="text" id="topic" placeholder="Conversation topic (optional)...">
            <button class="btn-start" onclick="startDemo()">Start</button>
            <button class="btn-stop" onclick="stopDemo()">Stop</button>
            <button class="btn-clear" onclick="clearMessages()">Clear</button>
        </div>

        <div id="status" class="status disconnected">
            Disconnected
        </div>

        <div id="roomInfo" class="room-info">
            Room: <a id="roomUrl" href="#" target="_blank"></a>
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
        const roomInfoDiv = document.getElementById('roomInfo');
        const roomUrlLink = document.getElementById('roomUrl');

        function formatPersonaLabel(filename) {
            let name = filename.replace('.json', '').replace(/^(alice_|bob_)/, '');
            return name.split('_').map(word =>
                word.charAt(0).toUpperCase() + word.slice(1)
            ).join(' ');
        }

        async function loadPersonas() {
            try {
                const response = await fetch('/daily-demo/personas');
                const personas = await response.json();

                const aliceSelect = document.getElementById('alicePersona');
                const bobSelect = document.getElementById('bobPersona');

                aliceSelect.innerHTML = '';
                personas.alice.forEach((filename, index) => {
                    const option = document.createElement('option');
                    option.value = filename + '.json';
                    option.textContent = formatPersonaLabel(filename);
                    if (index === 0) option.selected = true;
                    aliceSelect.appendChild(option);
                });

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
            ws = new WebSocket(`${protocol}//${window.location.host}/daily-demo/viewer/ws`);

            ws.onopen = () => {
                statusDiv.innerHTML = 'Connected - Watching Daily conversation';
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
            const speakerClass = speakerLower === 'alice' ? 'alice' : 'bob';
            messageDiv.className = `message ${speakerClass}`;

            const time = new Date(msg.timestamp).toLocaleTimeString();

            messageDiv.innerHTML = `
                <div class="speaker">${msg.speaker}</div>
                <div class="text">${msg.text}</div>
                <div class="time">${time}</div>
            `;

            conversationDiv.appendChild(messageDiv);
            conversationDiv.scrollTop = conversationDiv.scrollHeight;
        }

        async function startDemo() {
            const topic = document.getElementById('topic').value;
            const alice = document.getElementById('alicePersona').value;
            const bob = document.getElementById('bobPersona').value;

            const params = new URLSearchParams();
            if (topic) params.append('topic', topic);
            if (alice) params.append('alice', alice);
            if (bob) params.append('bob', bob);

            try {
                const response = await fetch(`/daily-demo/start?${params.toString()}`, {
                    method: 'POST'
                });
                const data = await response.json();
                if (response.ok) {
                    const aliceLabel = formatPersonaLabel(alice);
                    const bobLabel = formatPersonaLabel(bob);
                    statusDiv.innerHTML = `Started - ${aliceLabel} vs ${bobLabel}`;

                    if (data.room_url) {
                        roomUrlLink.href = data.room_url;
                        roomUrlLink.textContent = data.room_url;
                        roomInfoDiv.classList.add('visible');
                    }
                } else {
                    alert(data.detail || 'Failed to start demo');
                }
            } catch (error) {
                console.error('Error starting demo:', error);
                alert('Failed to start demo');
            }
        }

        async function stopDemo() {
            try {
                const response = await fetch('/daily-demo/stop', { method: 'POST' });
                const data = await response.json();
                if (response.ok) {
                    statusDiv.innerHTML = 'Stopped';
                    roomInfoDiv.classList.remove('visible');
                } else {
                    alert(data.detail || 'Failed to stop demo');
                }
            } catch (error) {
                console.error('Error stopping demo:', error);
                alert('Failed to stop demo');
            }
        }

        function clearMessages() {
            conversationDiv.innerHTML = `
                <div class="empty-state">
                    <p>No conversation yet.</p>
                    <p>Select personas and click Start to begin.</p>
                </div>
            `;
            roomInfoDiv.classList.remove('visible');
        }

        // Initialize on page load
        loadPersonas();
        connect();
    </script>
</body>
</html>
    """
    return Response(
        content=html_content,
        media_type="text/html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)