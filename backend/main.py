import sys
from pathlib import Path

# Add parent directory to path for face_recognition module
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import shutil
import os
import uuid
import json
import zipfile
import io
from datetime import datetime

# Import our modules
from database import get_db, init_db, RSVPModel, GrapevineModel, GalleryModel
from models import (
    RSVPRequest, RSVPResponse, 
    GrapevineRequest, GrapevineResponse,
    TestimonialRequest, CombinedMemoryRequest, GalleryResponse
)

# Import face recognition router
from face_recognition.api import router as face_router

# Initialize FastAPI app
app = FastAPI(title="Wedding Website API", version="1.0.0")

# Configure CORS - allows both DuckDNS domain and local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://govindkiginni.duckdns.org",  # Production DuckDNS domain (HTTPS)
        "http://govindkiginni.duckdns.org", # Production DuckDNS domain (HTTP)
        "http://43.205.76.228", # AWS LIGHTSAIL STATIC IP
        "https://43.205.76.228",  
        "http://localhost:8001",               # Local testing
        "http://127.0.0.1:8001",               # Local testing (IP)
        "http://localhost:5500",               # Live Server (VS Code)
        "http://127.0.0.1:5500",               # Live Server (VS Code)
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Create uploads directory if it doesn't exist
# Use absolute path for production deployment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount static files for the frontend - serve from parent directory
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..")), name="static")

# Mount thumbnails directory for face search
_thumbnails_dir = os.path.join(os.path.dirname(__file__), "..", "data", "thumbnails")
if os.path.exists(_thumbnails_dir):
    app.mount("/thumbnails", StaticFiles(directory=_thumbnails_dir), name="thumbnails")

# Mount full-size wedding photos directory (replaces Google Drive dependency)
_photos_dir = os.path.join(os.path.dirname(__file__), "..", "data", "images")
if os.path.exists(_photos_dir):
    app.mount("/photos", StaticFiles(directory=_photos_dir), name="photos")
    print(f"✓ Full photos directory mounted: {_photos_dir}")

# Include face recognition router
app.include_router(face_router)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_db()
    # Preload face recognition models in background
    try:
        from face_recognition import get_processor
        get_processor()  # This loads all models into memory
        print("✓ Face recognition models loaded")
    except Exception as e:
        print(f"⚠ Face recognition models not loaded: {e}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

# Frontend page routes
_frontend_dir = os.path.join(os.path.dirname(__file__), "..")

@app.get("/find-my-photos")
async def serve_find_my_photos():
    """Serve the face search page"""
    return FileResponse(os.path.join(_frontend_dir, "find-my-photos.html"))

@app.get("/find-my-photos.html")
async def serve_find_my_photos_html():
    """Serve the face search page (with .html extension)"""
    return FileResponse(os.path.join(_frontend_dir, "find-my-photos.html"))

@app.get("/grapevine")
async def serve_grapevine():
    """Serve the grapevine page"""
    return FileResponse(os.path.join(_frontend_dir, "grapevine.html"))

@app.get("/grapevine.html")
async def serve_grapevine_html():
    """Serve the grapevine page (with .html extension)"""
    return FileResponse(os.path.join(_frontend_dir, "grapevine.html"))

@app.get("/face-search.js")
async def serve_face_search_js():
    """Serve the face search JavaScript file"""
    return FileResponse(os.path.join(_frontend_dir, "static", "js", "face-search.js"), media_type="application/javascript")

# RSVP Endpoints
@app.post("/api/rsvp", response_model=RSVPResponse)
async def submit_rsvp(rsvp: RSVPRequest, db: Session = Depends(get_db)):
    """Submit RSVP response"""
    try:
        db_rsvp = RSVPModel(
            attending=rsvp.attending,
            guest_count=rsvp.guest_count,
            name=rsvp.name,
            phone=rsvp.phone,
            email=rsvp.email,
            dietary_requirements=rsvp.dietary_requirements,
            special_requests=rsvp.special_requests,
            side=rsvp.side
        )
        db.add(db_rsvp)
        db.commit()
        db.refresh(db_rsvp)
        
        return RSVPResponse(
            id=db_rsvp.id,
            attending=db_rsvp.attending,
            guest_count=db_rsvp.guest_count,
            name=db_rsvp.name,
            phone=db_rsvp.phone,
            email=db_rsvp.email,
            dietary_requirements=db_rsvp.dietary_requirements,
            special_requests=db_rsvp.special_requests,
            side=db_rsvp.side,
            created_at=db_rsvp.created_at
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting RSVP: {str(e)}")

@app.get("/api/rsvp", response_model=List[RSVPResponse])
async def get_rsvps(db: Session = Depends(get_db)):
    """Get all RSVP responses (for admin view)"""
    rsvps = db.query(RSVPModel).all()
    return [RSVPResponse(
        id=rsvp.id,
        attending=rsvp.attending,
        guest_count=rsvp.guest_count,
        contact_info=rsvp.contact_info,
        dietary_requirements=rsvp.dietary_requirements,
        special_requests=rsvp.special_requests,
        created_at=rsvp.created_at
    ) for rsvp in rsvps]

# Grapevine Endpoints
@app.post("/api/grapevine", response_model=GrapevineResponse)
async def submit_message(message: GrapevineRequest, db: Session = Depends(get_db)):
    """Submit new grapevine message"""
    try:
        db_message = GrapevineModel(
            message=message.message,
            author_name=message.author_name or "Anonymous"
        )
        db.add(db_message)
        db.commit()
        db.refresh(db_message)
        
        return GrapevineResponse(
            id=db_message.id,
            message=db_message.message,
            author_name=db_message.author_name,
            created_at=db_message.created_at
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting message: {str(e)}")

@app.get("/api/grapevine", response_model=List[GrapevineResponse])
async def get_messages(db: Session = Depends(get_db)):
    """Get all grapevine messages"""
    messages = db.query(GrapevineModel).order_by(GrapevineModel.created_at.asc()).all()
    return [GrapevineResponse(
        id=msg.id,
        message=msg.message,
        author_name=msg.author_name,
        created_at=msg.created_at
    ) for msg in messages]

# Gallery Endpoints
@app.post("/api/gallery/photo", response_model=GalleryResponse)
async def upload_photo(
    photo: UploadFile = File(...),
    author_name: Optional[str] = Form("Anonymous"),
    db: Session = Depends(get_db)
):
    """Upload photo to gallery"""
    try:
        # Validate file type
        if not photo.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Validate file size (5MB max)
        if photo.size > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File size must be less than 5MB")
        
        # Generate unique filename
        file_extension = photo.filename.split(".")[-1] if "." in photo.filename else "jpg"
        unique_filename = f"{uuid.uuid4()}.{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(photo.file, buffer)
        
        # Save to database
        db_gallery = GalleryModel(
            type="photo",
            content=unique_filename,  # Store just the filename
            author_name=author_name or "Anonymous"
        )
        db.add(db_gallery)
        db.commit()
        db.refresh(db_gallery)
        
        return GalleryResponse(
            id=db_gallery.id,
            type=db_gallery.type,
            content=db_gallery.content,
            author_name=db_gallery.author_name,
            created_at=db_gallery.created_at
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading photo: {str(e)}")

@app.post("/api/gallery/testimonial", response_model=GalleryResponse)
async def submit_testimonial(testimonial: TestimonialRequest, db: Session = Depends(get_db)):
    """Submit testimonial to gallery"""
    try:
        db_gallery = GalleryModel(
            type="testimonial",
            content=testimonial.content,
            author_name=testimonial.author_name or "Anonymous"
        )
        db.add(db_gallery)
        db.commit()
        db.refresh(db_gallery)
        
        return GalleryResponse(
            id=db_gallery.id,
            type=db_gallery.type,
            content=db_gallery.content,
            author_name=db_gallery.author_name,
            created_at=db_gallery.created_at
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting testimonial: {str(e)}")

@app.post("/api/gallery/combined", response_model=GalleryResponse)
async def submit_combined_memory(
    photo: Optional[UploadFile] = File(None),
    message: Optional[str] = Form(None),
    author_name: Optional[str] = Form("Anonymous"),
    db: Session = Depends(get_db)
):
    """Submit combined photo and text memory"""
    try:
        import json
        
        # Validate that at least one content type is provided
        if not photo and not message:
            raise HTTPException(status_code=400, detail="Please provide either a photo, message, or both")
        
        content_data = {}
        
        # Handle photo if provided
        if photo:
            # Validate file type
            if not photo.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="File must be an image")
            
            # Validate file size (5MB max)
            if photo.size > 5 * 1024 * 1024:
                raise HTTPException(status_code=400, detail="File size must be less than 5MB")
            
            # Generate unique filename
            file_extension = photo.filename.split(".")[-1] if "." in photo.filename else "jpg"
            unique_filename = f"{uuid.uuid4()}.{file_extension}"
            file_path = os.path.join(UPLOAD_DIR, unique_filename)
            
            # Save file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(photo.file, buffer)
            
            content_data["photo"] = unique_filename
        
        # Handle message if provided
        if message and message.strip():
            content_data["message"] = message.strip()
        
        # Save to database
        db_gallery = GalleryModel(
            type="combined",
            content=json.dumps(content_data),  # Store as JSON
            author_name=author_name or "Anonymous"
        )
        db.add(db_gallery)
        db.commit()
        db.refresh(db_gallery)
        
        return GalleryResponse(
            id=db_gallery.id,
            type=db_gallery.type,
            content=db_gallery.content,
            author_name=db_gallery.author_name,
            created_at=db_gallery.created_at
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error submitting combined memory: {str(e)}")

@app.get("/api/gallery", response_model=List[GalleryResponse])
async def get_gallery_items(db: Session = Depends(get_db)):
    """Get all gallery items"""
    items = db.query(GalleryModel).order_by(GalleryModel.created_at.desc()).all()
    return [GalleryResponse(
        id=item.id,
        type=item.type,
        content=item.content,
        author_name=item.author_name,
        created_at=item.created_at
    ) for item in items]

@app.post("/api/gallery/share-wedding-photos")
async def share_wedding_photos(
    image_ids: List[str] = Form(...),
    message: Optional[str] = Form(None),
    author_name: Optional[str] = Form("Anonymous"),
    db: Session = Depends(get_db)
):
    """
    Share wedding photos from Find My Photos to Memories page.
    
    Unlike /api/gallery/combined which handles file uploads,
    this endpoint accepts image IDs of existing wedding photos.
    """
    import json
    
    if not image_ids:
        raise HTTPException(status_code=400, detail="Please select at least one photo to share")
    
    # Verify at least one image exists
    photos_dir = os.path.join(os.path.dirname(__file__), "..", "data", "images")
    valid_ids = []
    for img_id in image_ids:
        img_path = os.path.join(photos_dir, f"{img_id}.jpg")
        if os.path.exists(img_path):
            valid_ids.append(img_id)
    
    if not valid_ids:
        raise HTTPException(status_code=404, detail="No valid photos found")
    
    # Create gallery entries for each photo
    created_items = []
    for img_id in valid_ids:
        content_data = {
            "wedding_photo_id": img_id,  # Reference to existing photo
            "thumbnail": f"/thumbnails/{img_id}.jpg",
            "full_image": f"/photos/{img_id}.jpg"
        }
        
        # Add message only to first photo
        if message and img_id == valid_ids[0]:
            content_data["message"] = message.strip()
        
        db_gallery = GalleryModel(
            type="wedding_photo",  # New type for shared wedding photos
            content=json.dumps(content_data),
            author_name=author_name or "Anonymous"
        )
        db.add(db_gallery)
        created_items.append(db_gallery)
    
    db.commit()
    
    return {
        "status": "success",
        "message": f"Shared {len(valid_ids)} photo(s) to Memories!",
        "shared_count": len(valid_ids)
    }

@app.post("/api/download-zip")
async def download_photos_zip(image_ids: List[str] = Form(...)):
    """
    Create and download a ZIP file containing selected photos.
    Used when user selects >10 photos for download.
    """
    if not image_ids:
        raise HTTPException(status_code=400, detail="No image IDs provided")
    
    # Limit to prevent abuse
    if len(image_ids) > 500:
        raise HTTPException(status_code=400, detail="Maximum 500 photos per download")
    
    # Create in-memory ZIP file
    zip_buffer = io.BytesIO()
    photos_dir = os.path.join(os.path.dirname(__file__), "..", "data", "images")
    
    found_count = 0
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for image_id in image_ids:
            # Try common image extensions
            for ext in ['jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG']:
                file_path = os.path.join(photos_dir, f"{image_id}.{ext}")
                if os.path.exists(file_path):
                    # Add to ZIP with a clean filename
                    arcname = f"wedding_photos/{image_id}.{ext.lower()}"
                    zip_file.write(file_path, arcname)
                    found_count += 1
                    break
    
    if found_count == 0:
        raise HTTPException(status_code=404, detail="No photos found")
    
    # Seek to beginning of buffer
    zip_buffer.seek(0)
    
    # Generate filename with count
    filename = f"wedding_photos_{found_count}_images.zip"
    
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )

@app.get("/api/uploads/{filename}")
async def get_uploaded_file(filename: str):
    """Serve uploaded files"""
    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="File not found")

# Serve the frontend at root
@app.get("/")
async def serve_frontend():
    """Serve the main index page"""
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "index.html"))

# Serve index.html explicitly (for back button navigation)
@app.get("/index.html")
async def serve_index():
    """Serve the main index page via index.html route"""
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "index.html"))

# Serve individual HTML pages
@app.get("/our-story.html")
async def serve_our_story():
    """Serve the our story page"""
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "our-story.html"))

@app.get("/memories.html")  
async def serve_memories():
    """Serve the memories page"""
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "memories.html"))

@app.get("/events.html")
async def serve_events():
    """Serve the events page"""
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "events.html"))

@app.get("/venue.html")
async def serve_venue():
    """Serve the venue page"""
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "venue.html"))

@app.get("/rsvp.html")
async def serve_rsvp():
    """Serve the RSVP page"""
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "rsvp.html"))

# Serve PWA files
@app.get("/sw.js")
async def serve_service_worker():
    """Serve the service worker file"""
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "sw.js"), media_type="application/javascript")

@app.get("/manifest.json")
async def serve_manifest():
    """Serve the PWA manifest file"""
    return FileResponse(os.path.join(os.path.dirname(__file__), "..", "manifest.json"), media_type="application/json")

# Serve static images from the images directory
@app.get("/images/{path:path}")
async def serve_images(path: str):
    """Serve static images from the images directory"""
    file_path = os.path.join(os.path.dirname(__file__), "..", "images", path)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Image not found")

# Serve favicon and apple touch icons
@app.get("/favicon.ico")
async def serve_favicon():
    """Serve favicon"""
    # Try SVG first
    svg_path = os.path.join(os.path.dirname(__file__), "..", "images", "favicon.svg")
    if os.path.exists(svg_path):
        return FileResponse(svg_path, media_type="image/svg+xml")
    # Fallback to ICO
    ico_path = os.path.join(os.path.dirname(__file__), "..", "images", "favicon.ico")
    if os.path.exists(ico_path):
        return FileResponse(ico_path)
    # Return 204 No Content if favicon doesn't exist
    from fastapi.responses import Response
    return Response(status_code=204)

@app.get("/apple-touch-icon.png")
async def serve_apple_icon():
    """Serve apple touch icon"""
    # Try SVG first (iOS supports SVG icons)
    svg_path = os.path.join(os.path.dirname(__file__), "..", "images", "icon-192.svg")
    if os.path.exists(svg_path):
        return FileResponse(svg_path, media_type="image/svg+xml")
    # Fallback to PNG
    png_path = os.path.join(os.path.dirname(__file__), "..", "images", "icon-192.png")
    if os.path.exists(png_path):
        return FileResponse(png_path)
    from fastapi.responses import Response
    return Response(status_code=204)

@app.get("/apple-touch-icon-precomposed.png")
async def serve_apple_icon_precomposed():
    """Serve apple touch icon precomposed"""
    # Try SVG first
    svg_path = os.path.join(os.path.dirname(__file__), "..", "images", "icon-192.svg")
    if os.path.exists(svg_path):
        return FileResponse(svg_path, media_type="image/svg+xml")
    # Fallback to PNG
    png_path = os.path.join(os.path.dirname(__file__), "..", "images", "icon-192.png")
    if os.path.exists(png_path):
        return FileResponse(png_path)
    from fastapi.responses import Response
    return Response(status_code=204)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)