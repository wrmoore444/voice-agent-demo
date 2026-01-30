"""
=============================================================================
MODELS.PY - SQLAlchemy Database Models
=============================================================================

This module defines the database schema for the Voice Agent Demo application.
Uses SQLAlchemy ORM with async support (via aiosqlite for SQLite, asyncpg for
PostgreSQL, or aiomysql for MySQL).

TABLES:
-------
- users: User accounts that own agents and conversations
- agents: AI agent configurations with custom prompts
- conversations: Voice conversation sessions with audio and transcripts
- transcriptions: Individual transcript entries for conversations
- menu_items: Demo data for restaurant menu (example use case)
- event_bookings: Demo data for event bookings (example use case)

RELATIONSHIPS:
--------------
    User ──┬── Agent ──── Conversation ──── Transcription
           └── Conversation

DATABASE CONFIGURATION:
-----------------------
Set DATABASE_URL in .env file. Examples:
- SQLite: sqlite+aiosqlite:///./interview.db (default, works out of box)
- PostgreSQL: postgresql+asyncpg://user:pass@localhost/dbname
- MySQL: mysql+aiomysql://user:pass@localhost/dbname

Tables are auto-created on application startup (see main.py lifespan).
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, ForeignKey, LargeBinary
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class User(Base):
    """
    User account that can own agents and conversations.

    Attributes:
        id: Primary key
        username: Unique username for login
        password: Password (stored as-is - production should hash this!)
        created_at: Account creation timestamp
        agents: List of Agent objects owned by this user
        conversations: List of Conversation objects owned by this user
    """
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(128), unique=True, nullable=False)
    password = Column(String(256), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    agents = relationship("Agent", back_populates="owner", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="owner", cascade="all, delete-orphan")


class Agent(Base):
    """
    AI agent configuration with custom system prompt.

    An Agent defines the personality and behavior of an AI voice assistant.
    Each agent has a custom prompt that instructs the LLM how to behave.

    Attributes:
        id: Primary key
        name: Display name for the agent (e.g., "Customer Service Bot")
        prompt: Custom system prompt that defines agent behavior
        user_id: Foreign key to the owning User
        created_at: Agent creation timestamp
        owner: User object that owns this agent
        conversations: List of Conversation objects using this agent
    """
    __tablename__ = "agents"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(256), nullable=False)
    prompt = Column(Text, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    owner = relationship("User", back_populates="agents")
    conversations = relationship("Conversation", back_populates="agent", cascade="all, delete-orphan")


class Conversation(Base):
    """
    A voice conversation session between a user and an agent.

    Stores the conversation metadata, optional audio recording, extracted
    datapoints, and links to individual transcription entries.

    Attributes:
        id: Primary key
        agent_id: Foreign key to the Agent used for this conversation
        user_id: Foreign key to the User who initiated the conversation
        uuid: Unique identifier for API access (e.g., "conv_abc123")
        audio_file: Raw audio bytes of the conversation (optional)
        audio_filename: Original filename if audio was uploaded
        audio_mime: MIME type of audio (e.g., "audio/wav")
        datapoints: JSON string of extracted data (see datapoint_extractor.py)
        created_at: Conversation start timestamp
        updated_at: Last modification timestamp
        agent: Agent object used for this conversation
        owner: User object who owns this conversation
        transcriptions: List of Transcription entries (speaker + text)
    """
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    uuid = Column(String(64), unique=True, nullable=False)

    audio_file = Column(LargeBinary, nullable=True)
    audio_filename = Column(String(256), nullable=True)
    audio_mime = Column(String(128), nullable=True)
    datapoints = Column(Text, nullable=True)  # JSON extracted by datapoint_extractor.py

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    agent = relationship("Agent", back_populates="conversations")
    owner = relationship("User", back_populates="conversations")
    transcriptions = relationship("Transcription", back_populates="conversation", cascade="all, delete-orphan")


class Transcription(Base):
    """
    A single transcription entry from a conversation.

    Each entry represents one utterance from either the user or the assistant.
    These are created in real-time during voice conversations.

    Attributes:
        id: Primary key
        conversation_id: Foreign key to parent Conversation
        text: The transcribed text content
        speaker: Who spoke - "user" or "assistant"
        timestamp: When this utterance occurred
        conversation: Parent Conversation object
    """
    __tablename__ = "transcriptions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    text = Column(Text, nullable=False)
    speaker = Column(String(50), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationship
    conversation = relationship("Conversation", back_populates="transcriptions")


class MenuItem(Base):
    """
    Menu item for restaurant demo use case.

    Example data model showing how an AI agent could access
    menu information during a voice conversation.

    Attributes:
        id: Primary key
        category: Menu category (e.g., "Appetizers", "Main Course")
        title: Item name (e.g., "Caesar Salad")
        description: Item description
        image: Optional image bytes
        image_filename: Original image filename
        image_mime: Image MIME type
        price: Price as string (e.g., "$12.99")
        created_at: Creation timestamp
        updated_at: Last modification timestamp
    """
    __tablename__ = "menu_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    category = Column(String(128), nullable=False, index=True)
    title = Column(String(256), nullable=False)
    description = Column(Text, nullable=True)
    image = Column(LargeBinary, nullable=True)
    image_filename = Column(String(256), nullable=True)
    image_mime = Column(String(128), nullable=True)
    price = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class EventBooking(Base):
    """
    Event booking for venue demo use case.

    Example data model showing how an AI agent could collect
    event booking information during a voice conversation.

    Attributes:
        id: Primary key
        full_name: Customer's full name
        email: Customer's email address
        phone: Customer's phone number
        contact_method: Preferred contact method (phone/email/text)
        contact_consent: Whether customer consented to contact
        company: Company name (optional)
        event_type: Type of event (birthday, corporate, etc.)
        event_type_other: Custom event type if "other" selected
        event_date: Date of the event
        date_flexible: Whether date is flexible
        start_time: Event start time
        end_time: Event end time (optional)
        guest_count: Number of guests range
        preferred_space: Venue space preference
        event_vibe: Description of desired event atmosphere
        food_style: Food service style preferences
        status: Booking status (pending/confirmed/cancelled)
        created_at: Creation timestamp
        updated_at: Last modification timestamp
    """
    __tablename__ = "event_bookings"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Contact Information
    full_name = Column(String(256), nullable=False)
    email = Column(String(256), nullable=False)
    phone = Column(String(50), nullable=False)
    contact_method = Column(String(50), nullable=False)  # phone, email, text
    contact_consent = Column(String(10), nullable=False, default="false")  # true, false
    company = Column(String(256), nullable=True)
    
    # Event Details
    event_type = Column(String(128), nullable=False)  # birthday, corporate, watch_party, bachelor_bachelorette, holiday, other
    event_type_other = Column(String(256), nullable=True)
    event_date = Column(DateTime, nullable=False)
    date_flexible = Column(String(10), nullable=False)  # yes, no
    start_time = Column(String(20), nullable=False)
    end_time = Column(String(20), nullable=True)
    guest_count = Column(String(50), nullable=False)  # 1-10, 11-25, 26-50, 51-75, 76-plus
    
    # Venue Preference
    preferred_space = Column(String(50), nullable=False)  # sand_box, sand_bar, not_sure
    event_vibe = Column(Text, nullable=True)
    
    # Food & Beverage (stored as comma-separated values)
    food_style = Column(Text, nullable=True)  # passed_appetizers, buffet, plated, etc.
    
    # Metadata
    status = Column(String(50), default="pending")  # pending, confirmed, cancelled
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)