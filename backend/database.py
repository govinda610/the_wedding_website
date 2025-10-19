from sqlalchemy import create_engine, Column, Integer, String, Boolean, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func
import os

# Production: Use absolute path in home directory
# Local dev: Use current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, "wedding_database.db")

# Database setup
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

print(f"BASE_DIR = {BASE_DIR}")
print(f"DATABASE_PATH = {DATABASE_PATH}")
print(f"DATABASE_URL = {DATABASE_URL}")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class RSVPModel(Base):
    __tablename__ = "rsvp_responses"
    
    id = Column(Integer, primary_key=True, index=True)
    attending = Column(String, nullable=False)  # 'yes', 'no', or 'objection'
    guest_count = Column(Integer, nullable=False)
    name = Column(String, nullable=False)
    phone = Column(String)
    email = Column(String)
    dietary_requirements = Column(String)
    special_requests = Column(Text)
    side = Column(String, default="Not Specified")
    created_at = Column(DateTime, default=func.now())

class GrapevineModel(Base):
    __tablename__ = "grapevine_messages"
    
    id = Column(Integer, primary_key=True, index=True)
    message = Column(Text, nullable=False)
    author_name = Column(String, default="Anonymous")
    created_at = Column(DateTime, default=func.now())

class GalleryModel(Base):
    __tablename__ = "gallery_items"
    
    id = Column(Integer, primary_key=True, index=True)
    type = Column(String, nullable=False)  # 'photo' or 'testimonial'
    content = Column(Text, nullable=False)  # file path for photos, text for testimonials
    author_name = Column(String, default="Anonymous")
    created_at = Column(DateTime, default=func.now())

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize database
def init_db():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)