from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

# RSVP Models
class RSVPRequest(BaseModel):
    attending: str  # 'yes', 'no', or 'objection'
    guest_count: int = Field(ge=1, le=10)
    name: str
    phone: Optional[str] = None
    email: Optional[str] = None
    dietary_requirements: Optional[str] = None
    special_requests: Optional[str] = None

class RSVPResponse(BaseModel):
    id: int
    attending: str
    guest_count: int
    name: str
    phone: Optional[str]
    email: Optional[str]
    dietary_requirements: Optional[str]
    special_requests: Optional[str]
    created_at: datetime

# Grapevine Models
class GrapevineRequest(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    author_name: Optional[str] = "Anonymous"

class GrapevineResponse(BaseModel):
    id: int
    message: str
    author_name: str
    created_at: datetime

# Gallery Models
class TestimonialRequest(BaseModel):
    content: str = Field(min_length=1, max_length=1000)
    author_name: Optional[str] = "Anonymous"

class CombinedMemoryRequest(BaseModel):
    message: Optional[str] = None
    author_name: Optional[str] = "Anonymous"

class GalleryResponse(BaseModel):
    id: int
    type: str  # 'photo', 'testimonial', or 'combined'
    content: str  # file path for photos, text content for testimonials, JSON for combined
    author_name: str
    created_at: datetime