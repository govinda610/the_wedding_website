"""
Face Recognition System Configuration
======================================

This module contains all configurable settings for the face recognition system.
The system is designed to easily switch between different models (ArcFace, MobileFaceNet)
by changing a single configuration variable.

USAGE:
------
    from face_recognition.config import Config
    
    # Get current model settings
    model_path = Config.EMBEDDER_MODEL_PATH
    threshold = Config.SIMILARITY_THRESHOLD

SWITCHING MODELS:
-----------------
    To switch from ArcFace to MobileFaceNet:
    1. Download MobileFaceNet ONNX model
    2. Change EMBEDDER_MODEL in this file from "arcface" to "mobilefacenet"
    3. Restart the application
"""

import os
from pathlib import Path
from typing import Literal

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Base directory for the face recognition system
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "face_recognition" / "models"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# MODEL CONFIGURATION (EASILY SWITCHABLE)
# =============================================================================

class Config:
    """
    Central configuration for the face recognition system.
    
    To switch models, change EMBEDDER_MODEL to one of:
    - "arcface": Higher accuracy, slower (~150-300ms), larger model (261MB)
    - "mobilefacenet": Lower accuracy, faster (~20-50ms), tiny model (4MB)
    """
    
    # =========================================================================
    # MODEL SELECTION
    # =========================================================================
    # FaceNet512 from deepface-onnx (validated, accurate)
    EMBEDDER_MODEL: str = "facenet512"
    
    # =========================================================================
    # MODEL PATHS
    # =========================================================================
    # Face detection model (YuNet - fast and accurate)
    YUNET_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
    
    # =========================================================================
    # EMBEDDER SETTINGS (DERIVED FROM MODEL SELECTION)
    # =========================================================================
    @classmethod
    def get_embedder_model_path(cls) -> Path:
        """Get the path to the currently selected embedder model."""
        if cls.EMBEDDER_MODEL == "arcface":
            return cls.ARCFACE_MODEL_PATH
        elif cls.EMBEDDER_MODEL == "mobilefacenet":
            return cls.MOBILEFACENET_MODEL_PATH
        else:
            raise ValueError(f"Unknown embedder model: {cls.EMBEDDER_MODEL}")
    
    @classmethod
    def get_input_size(cls) -> tuple:
        """Get the input size for the currently selected embedder model."""
        if cls.EMBEDDER_MODEL == "arcface":
            return (112, 112)  # ArcFace uses 112x112
        elif cls.EMBEDDER_MODEL == "mobilefacenet":
            return (112, 112)  # MobileFaceNet also uses 112x112
        else:
            raise ValueError(f"Unknown embedder model: {cls.EMBEDDER_MODEL}")
    
    # =========================================================================
    # FACE DETECTION SETTINGS
    # =========================================================================
    # Minimum confidence for face detection (0-1)
    # Lowered from 0.75 to catch more faces in varied lighting
    DETECTION_CONFIDENCE_THRESHOLD = 0.68
    
    # Minimum face size in pixels (ignore tiny faces)
    # Reduced from 50 to 40 to catch more faces
    MIN_FACE_SIZE = 40
    
    # =========================================================================
    # SIMILARITY SEARCH SETTINGS
    # =========================================================================
    # Cosine similarity thresholds for FaceNet512
    # Same-face scores: ~0.75-0.85 for same person
    # Different-face scores: ~0.10-0.30
    
    # Standard threshold for face matching
    SIMILARITY_THRESHOLD = float(os.getenv("FACE_THRESHOLD", "0.50"))
    
    # Stricter threshold for crowd scenes (many faces)
    MULTI_FACE_THRESHOLD = float(os.getenv("FACE_MULTI_THRESHOLD", "0.60"))
    
    # For backward compatibility with classmethods
    @classmethod
    def get_similarity_threshold(cls) -> float:
        return cls.SIMILARITY_THRESHOLD
    
    @classmethod
    def get_multi_face_threshold(cls) -> float:
        return cls.MULTI_FACE_THRESHOLD
    
    # Number of faces in an image above which we apply stricter threshold
    # Increased from 3 to 5 since wedding photos often have 5+ people
    MULTI_FACE_COUNT = 5

    
    # Embedding dimension (both ArcFace and MobileFaceNet use 512)
    EMBEDDING_DIM = 512
    
    # Minimum face quality score (blur/brightness check)
    MIN_FACE_QUALITY = 0.3
    
    # =========================================================================
    # DATA PATHS
    # =========================================================================
    # Face embeddings storage
    EMBEDDINGS_FILE = DATA_DIR / "face_embeddings.npz"
    
    # Face metadata (bounding boxes, emotions, image IDs)
    # Use classmethods below for model-specific paths
    METADATA_FILE = DATA_DIR / "face_metadata.json"
    
    # FAISS index for similarity search
    # Use classmethods below for model-specific paths
    FAISS_INDEX_FILE = DATA_DIR / "face_index.faiss"
    
    # Full images directory (for direct VPS serving)
    FULL_IMAGES_DIR = DATA_DIR / "full_images"
    
    # Ingestion progress tracking (for resume capability)
    PROGRESS_FILE = DATA_DIR / "ingestion_progress.json"
    
    # Wedding photos metadata CSV
    PHOTOS_METADATA_CSV = BASE_DIR / "wedding_photos_metadata.csv"
    
    # =========================================================================
    # DATA PATH METHODS
    # =========================================================================
    @classmethod
    def get_faiss_index_path(cls) -> Path:
        """Get FAISS index path."""
        return DATA_DIR / "face_index.faiss"
    
    @classmethod
    def get_metadata_path(cls) -> Path:
        """Get metadata JSON path."""
        return DATA_DIR / "face_metadata.json"
    
    # =========================================================================
    # CONCURRENCY SETTINGS
    # =========================================================================
    # Number of worker processes for face processing
    # Set to number of CPU cores (2 for Lightsail)
    MAX_WORKERS = int(os.getenv("FACE_WORKERS", "2"))
    
    # =========================================================================
    # GOOGLE DRIVE SETTINGS
    # =========================================================================
    # Delay between Drive API calls (rate limiting)
    DRIVE_API_DELAY_SECONDS = 0.5
    
    # Batch size for saving progress during ingestion
    INGESTION_BATCH_SIZE = 10
    
    # =========================================================================
    # EMOTION ANALYSIS SETTINGS
    # =========================================================================
    # Whether to analyze emotions during ingestion
    ENABLE_EMOTION_ANALYSIS = True
    
    # Emotion categories
    EMOTION_CATEGORIES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


# =============================================================================
# HELPER FUNCTION TO PRINT CURRENT CONFIG
# =============================================================================

def print_config():
    """Print the current configuration for debugging."""
    print("=" * 60)
    print("FACE RECOGNITION SYSTEM CONFIGURATION")
    print("=" * 60)
    print(f"Embedder Model: FaceNet512 (99.65% LFW accuracy)")
    print(f"FAISS Index: {Config.get_faiss_index_path()}")
    print(f"Metadata: {Config.get_metadata_path()}")
    print(f"Similarity Threshold: {Config.SIMILARITY_THRESHOLD}")
    print(f"Max Workers: {Config.MAX_WORKERS}")
    print(f"Emotion Analysis: {Config.ENABLE_EMOTION_ANALYSIS}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
