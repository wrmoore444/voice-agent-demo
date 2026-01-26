from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, ForeignKey, LargeBinary
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(128), unique=True, nullable=False)
    password = Column(String(256), nullable=False)  
    created_at = Column(DateTime, default=datetime.utcnow)

    agents = relationship("Agent", back_populates="owner", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="owner", cascade="all, delete-orphan")


class Agent(Base):
    __tablename__ = "agents"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(256), nullable=False)
    prompt = Column(Text, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    owner = relationship("User", back_populates="agents")
    conversations = relationship("Conversation", back_populates="agent", cascade="all, delete-orphan")


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    uuid = Column(String(64), unique=True, nullable=False)
    
    audio_file = Column(LargeBinary, nullable=True)
    audio_filename = Column(String(256), nullable=True)
    audio_mime = Column(String(128), nullable=True)
    datapoints = Column(Text, nullable=True)  # JSON or structured data extracted from conversation
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    agent = relationship("Agent", back_populates="conversations")
    owner = relationship("User", back_populates="conversations")
    transcriptions = relationship("Transcription", back_populates="conversation", cascade="all, delete-orphan")


class Transcription(Base):
    __tablename__ = "transcriptions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    text = Column(Text, nullable=False)
    speaker = Column(String(50), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationship
    conversation = relationship("Conversation", back_populates="transcriptions")


class MenuItem(Base):
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