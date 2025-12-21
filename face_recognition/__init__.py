"""
Face Recognition Module
=======================

This package provides face detection, embedding generation, emotion analysis,
and similarity search functionality for the wedding photo recognition system.

COMPONENTS:
-----------
- config: Configuration settings (switchable models)
- detector: Face detection using YuNet
- embedder: Face embedding generation using ArcFace/MobileFaceNet
- emotion: Emotion analysis using DeepFace
- search: FAISS-based similarity search
- processor: Main class that ties everything together

USAGE:
------
    from face_recognition import FaceProcessor, Config
    
    processor = FaceProcessor()
    results = processor.search_faces(image_bytes, threshold=0.5)
"""

from .config import Config, print_config
from .processor import FaceProcessor, get_processor

__version__ = "1.0.0"
__all__ = ["Config", "FaceProcessor", "get_processor", "print_config"]
