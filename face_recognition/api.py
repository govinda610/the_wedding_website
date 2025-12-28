"""
Face Recognition API Routes
============================

FastAPI routes for face recognition functionality.
Add these routes to the existing wedding website backend.

ENDPOINTS:
----------
POST /api/face/search      - Upload image, find matching photos
GET  /api/face/status/{id} - Check status of search task
GET  /api/face/stats       - Get index statistics

MEMORY OPTIMIZATIONS (for 2GB RAM):
-----------------------------------
1. Uses ThreadPoolExecutor (not ProcessPool) - shares models in memory
2. LRU cache for task results - prevents memory leak
3. ONNX Runtime releases GIL - enables true parallel inference
"""

import asyncio
import uuid
import threading
from collections import OrderedDict
from typing import List, Optional, Dict, Any
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel


# =============================================================================
# LRU CACHE FOR TASK RESULTS (Fix #2: Prevent memory leak)
# =============================================================================

class LRUCache(OrderedDict):
    """
    Least Recently Used cache with a maximum size.
    Thread-safe implementation for concurrent access.
    
    When cache exceeds max_size, oldest (least recently accessed) entries
    are automatically removed. This prevents unbounded memory growth.
    
    Uses RLock (reentrant lock) to allow nested method calls.
    """
    
    def __init__(self, max_size: int = 100):
        super().__init__()
        self.max_size = max_size
        # Use RLock (reentrant) to allow nested lock acquisition
        self.lock = threading.RLock()
    
    def __setitem__(self, key, value):
        with self.lock:
            # Move to end if exists (mark as recently used)
            try:
                if super().__contains__(key):
                    self.move_to_end(key)
            except KeyError:
                pass  # Key was evicted by another thread, continue with set
            super().__setitem__(key, value)
            # Remove oldest if over capacity
            while len(self) > self.max_size:
                try:
                    self.popitem(last=False)
                except KeyError:
                    break  # Cache is empty or race condition, stop evicting
    
    def __getitem__(self, key):
        with self.lock:
            try:
                value = super().__getitem__(key)
                # Only move_to_end if we successfully got the value
                try:
                    self.move_to_end(key)
                except KeyError:
                    pass  # Key was evicted between get and move, that's OK
                return value
            except KeyError:
                raise  # Re-raise KeyError for callers to handle
    
    def __contains__(self, key):
        with self.lock:
            return super().__contains__(key)
    
    def get(self, key, default=None):
        with self.lock:
            try:
                value = super().__getitem__(key)
                try:
                    self.move_to_end(key)
                except KeyError:
                    pass  # Key was evicted, that's OK
                return value
            except KeyError:
                return default


# Task results storage with LRU eviction (max 100 concurrent tasks)
task_results: LRUCache = LRUCache(max_size=100)


# =============================================================================
# THREAD POOL EXECUTOR (Fix #1: Share models in memory)
# =============================================================================

# Thread pool for inference (ONNX Runtime releases GIL, so threads work well)
_executor: Optional[ThreadPoolExecutor] = None

# Shared processor instance (thread-safe due to GIL release in ONNX)
_processor = None
_processor_lock = threading.Lock()


def get_executor() -> ThreadPoolExecutor:
    """Get or create the thread pool executor."""
    global _executor
    if _executor is None:
        from face_recognition.config import Config
        # Use threads, not processes - models stay loaded once
        _executor = ThreadPoolExecutor(max_workers=Config.MAX_WORKERS)
    return _executor


def get_shared_processor():
    """
    Get the shared FaceProcessor instance.
    
    Thread-safe: uses lock for initialization only.
    Once initialized, concurrent access is safe because:
    - ONNX Runtime releases GIL during inference
    - FAISS search is also thread-safe
    """
    global _processor
    if _processor is None:
        with _processor_lock:
            # Double-check locking pattern
            if _processor is None:
                from face_recognition import FaceProcessor
                _processor = FaceProcessor()
    return _processor


# =============================================================================
# MODELS
# =============================================================================

class SearchResult(BaseModel):
    """Result of a face search."""
    image_id: str
    event: str
    date: str
    similarity: float
    emotions: Dict[str, Any]  # Can contain floats and string 'dominant'
    drive_thumbnail: str
    drive_view: str = ""  # Full-size view URL
    drive_download: str
    # Bride/groom presence flags for filtering
    has_bride: bool = False
    has_groom: bool = False


class DebugInfo(BaseModel):
    """Debug info for evaluation framework."""
    detected_faces: List[Dict[str, Any]] = []
    processing_time_ms: int = 0
    index_total_faces: int = 0


class SearchResponse(BaseModel):
    """Response from face search endpoint."""
    task_id: str
    status: str
    message: str
    face_count: Optional[int] = None
    total_matches: Optional[int] = None
    matches: Optional[List[SearchResult]] = None
    debug: Optional[DebugInfo] = None


class StatsResponse(BaseModel):
    """Response from stats endpoint."""
    total_faces: int
    embedder_model: str
    similarity_threshold: float
    emotion_analysis_enabled: bool


# =============================================================================
# WORKER FUNCTION (runs in thread, shares models)
# =============================================================================

def process_face_search(image_bytes: bytes, threshold: float) -> Dict:
    """
    Process face search in a thread.
    
    This function runs in ThreadPoolExecutor:
    - Models are loaded ONCE and shared across threads
    - ONNX Runtime releases GIL during inference
    - Multiple threads can run inference in parallel
    
    Args:
        image_bytes: Raw image bytes from upload
        threshold: Similarity threshold for matching
    
    Returns:
        Dict with search results including debug info for evaluation
    """
    import time
    start_time = time.time()
    
    try:
        # Get shared processor (models loaded once)
        processor = get_shared_processor()
        
        # Search for matching faces
        results = processor.search_faces(image_bytes, threshold=threshold)
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Build debug info for evaluation framework
        detected_faces = []
        for i, face in enumerate(results.get("faces", [])):
            bbox = face.get("bbox", [0, 0, 0, 0])
            detected_faces.append({
                "face_index": i,
                "bbox": bbox,
                "confidence": round(face.get("confidence", 0), 3),
                "size": int(bbox[2] * bbox[3]) if len(bbox) >= 4 else 0,
                "width": int(bbox[2]) if len(bbox) >= 3 else 0,
                "height": int(bbox[3]) if len(bbox) >= 4 else 0
            })
        
        # Add URLs to matches - ALL LOCAL (no Google Drive dependency)
        for match in results.get("matches", []):
            file_id = match['image_id']
            # Local thumbnail for fast grid loading
            match["drive_thumbnail"] = f"/thumbnails/{file_id}.jpg"
            # Local full-size image for lightbox view
            match["drive_view"] = f"/photos/{file_id}.jpg"
            # Local full-size image for download
            match["drive_download"] = f"/photos/{file_id}.jpg"
        
        # Annotate with bride/groom presence
        try:
            from .bride_groom import get_bride_groom_matcher
            matcher = get_bride_groom_matcher()
            matcher.annotate_matches(results.get("matches", []))
        except Exception as e:
            print(f"Warning: Bride/groom annotation failed: {e}")
        
        return {
            "status": "completed",
            "face_count": results.get("face_count", 0),
            "total_matches": results.get("total_count", 0),
            "matches": results.get("matches", []),
            "threshold": threshold,
            # Debug info for evaluation framework
            "debug": {
                "detected_faces": detected_faces,
                "processing_time_ms": processing_time_ms,
                "index_total_faces": processor.get_stats().get("total_faces", 0)
            }
        }
        
    except Exception as e:
        import traceback
        return {
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# =============================================================================
# API ROUTES
# =============================================================================

router = APIRouter(prefix="/api/face", tags=["face-recognition"])


@router.post("/search", response_model=SearchResponse)
async def search_faces(
    file: UploadFile = File(...),
    threshold: float = Form(0.5)
):
    """
    Search for matching photos by uploading a face image.
    
    This endpoint:
    1. Receives the uploaded image
    2. Submits it to the thread pool for async processing
    3. Returns a task_id immediately
    4. Client polls /status/{task_id} for results
    
    Args:
        file: Uploaded image file (JPEG, PNG, etc.)
        threshold: Similarity threshold (0.3-0.7, default 0.5)
    
    Returns:
        Task ID for polling status
    """
    # Validate threshold
    if not 0.3 <= threshold <= 0.9:
        raise HTTPException(status_code=400, detail="Threshold must be between 0.3 and 0.9")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read image bytes
    image_bytes = await file.read()
    
    # Validate file size (max 10MB)
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 10MB)")
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    # Submit to thread pool
    loop = asyncio.get_event_loop()
    executor = get_executor()
    future = loop.run_in_executor(executor, process_face_search, image_bytes, threshold)
    
    # Store future for status checking (LRU cache auto-evicts old entries)
    task_results[task_id] = {
        "status": "processing",
        "future": future
    }
    
    return SearchResponse(
        task_id=task_id,
        status="processing",
        message="Search started. Poll /api/face/status/{task_id} for results."
    )


def process_multi_image_search(image_bytes_list: List[bytes], threshold: float) -> Dict:
    """Process multi-image search in thread pool using hybrid strategy."""
    import time
    start = time.time()
    
    try:
        processor = get_shared_processor()
        result = processor.search_faces_multi_image(image_bytes_list, threshold)
        
        processing_time = time.time() - start
        
        # Add URLs to matches - ALL LOCAL (no Google Drive dependency)
        matches = result.get("matches", [])
        for match in matches:
            file_id = match.get('image_id', '')
            match["drive_thumbnail"] = f"/thumbnails/{file_id}.jpg"
            match["drive_view"] = f"/photos/{file_id}.jpg"
            match["drive_download"] = f"/photos/{file_id}.jpg"
        
        # Annotate with bride/groom presence
        try:
            from .bride_groom import get_bride_groom_matcher
            matcher = get_bride_groom_matcher()
            matcher.annotate_matches(matches)
        except Exception as e:
            print(f"Warning: Bride/groom annotation failed: {e}")
        
        return {
            "status": "completed",
            "results": matches,
            "total_count": result.get("total_count", 0),
            "faces_detected": result.get("face_count", 0),
            "query_images_used": result.get("query_images_used", 0),
            "strategy": result.get("strategy", "hybrid"),
            "debug": {
                "detected_faces": result.get("faces", []),
                "processing_time_seconds": processing_time,
                "index_total_faces": processor.get_stats()["total_faces"],
                "mean_matches_count": result.get("mean_matches_count", 0),
                **result.get("debug", {})
            }
        }
    except Exception as e:
        import traceback
        return {
            "status": "failed",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@router.post("/search-multi", response_model=SearchResponse)
async def search_faces_multi(
    files: List[UploadFile] = File(...),
    threshold: float = Form(0.55)
):
    """
    Search using MULTIPLE images for better accuracy.
    
    Uses hybrid strategy:
    1. Mean embedding (reduces false negatives from variance)
    2. Union search (catches matches any image finds)
    3. Voting (reduces false positives via consensus)
    
    Args:
        files: 1-5 uploaded image files
        threshold: Similarity threshold (default 0.55)
    
    Returns:
        Task ID for polling status
    """
    # Validate threshold
    if not 0.3 <= threshold <= 0.9:
        raise HTTPException(status_code=400, detail="Threshold must be between 0.3 and 0.9")
    
    # Validate file count
    if len(files) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 images allowed")
    
    if len(files) < 1:
        raise HTTPException(status_code=400, detail="At least 1 image required")
    
    # Read all image bytes
    image_bytes_list = []
    for file in files:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"File {file.filename} must be an image")
        
        image_bytes = await file.read()
        
        if len(image_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail=f"File {file.filename} too large (max 10MB)")
        
        image_bytes_list.append(image_bytes)
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    # Submit to thread pool
    loop = asyncio.get_event_loop()
    executor = get_executor()
    future = loop.run_in_executor(executor, process_multi_image_search, image_bytes_list, threshold)
    
    # Store future
    task_results[task_id] = {
        "status": "processing",
        "future": future
    }
    
    return SearchResponse(
        task_id=task_id,
        status="processing",
        message=f"Multi-image search started with {len(files)} images. Poll /api/face/status/{task_id} for results."
    )


@router.get("/status/{task_id}", response_model=SearchResponse)
async def get_search_status(task_id: str):
    """
    Get the status of a face search task.
    
    Client should poll this endpoint every 1-2 seconds until
    status is "completed" or "failed".
    
    Args:
        task_id: Task ID from /search endpoint
    
    Returns:
        Current status and results if completed
    """
    if task_id not in task_results:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = task_results.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Check if still processing
    if task.get("status") == "processing":
        future = task.get("future")
        
        if future and future.done():
            # Get result from future
            result = future.result()
            
            # Update stored result
            task_results[task_id] = result
            
            if result["status"] == "completed":
                # Handle both single-image (total_matches) and multi-image (total_count) responses
                total = result.get("total_matches") or result.get("total_count", 0)
                face_count = result.get("face_count") or result.get("faces_detected", 1)
                
                return SearchResponse(
                    task_id=task_id,
                    status="completed",
                    message=f"Found {total} matching photos",
                    face_count=face_count,
                    total_matches=total,
                    matches=[SearchResult(**m) for m in result.get("matches", result.get("results", []))],
                    debug=DebugInfo(**result.get("debug", {})) if result.get("debug") else None
                )
            else:
                return SearchResponse(
                    task_id=task_id,
                    status="failed",
                    message=result.get("error", "Unknown error")
                )
        else:
            return SearchResponse(
                task_id=task_id,
                status="processing",
                message="Still processing..."
            )
    
    # Already completed or failed
    if task.get("status") == "completed":
        return SearchResponse(
            task_id=task_id,
            status="completed",
            message=f"Found {task['total_matches']} matching photos",
            face_count=task.get("face_count"),
            total_matches=task.get("total_matches"),
            matches=[SearchResult(**m) for m in task.get("matches", [])],
            debug=DebugInfo(**task.get("debug", {})) if task.get("debug") else None
        )
    else:
        return SearchResponse(
            task_id=task_id,
            status="failed",
            message=task.get("error", "Unknown error")
        )


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get statistics about the face recognition index.
    
    Returns:
        Number of faces indexed, model info, etc.
    """
    try:
        processor = get_shared_processor()
        stats = processor.get_stats()
        return StatsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# INTEGRATION HELPER
# =============================================================================

def add_face_routes(app):
    """
    Add face recognition routes to an existing FastAPI app.
    
    Usage:
        from face_api import add_face_routes
        add_face_routes(app)
    """
    app.include_router(router)
