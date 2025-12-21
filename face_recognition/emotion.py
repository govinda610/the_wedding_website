"""
Emotion Analyzer Module (ONNX Version)
======================================

This module analyzes emotions in face images using ONNX-based model.
Much lighter than TensorFlow-based DeepFace (~6MB vs ~300MB RAM).

EDUCATIONAL NOTES:
------------------
Emotion detection uses a CNN trained on facial expressions.
It classifies each face into one of 7 categories:
- angry, disgust, fear, happy, sad, surprise, neutral

The output is a probability distribution over these categories.
For example: {"happy": 0.7, "neutral": 0.2, "sad": 0.1, ...}

MEMORY OPTIMIZATION:
--------------------
- Uses ONNX Runtime instead of TensorFlow
- Model size: 6MB (vs ~100MB for TensorFlow version)
- RAM usage: ~20MB (vs ~300MB for TensorFlow version)

USAGE:
------
    from face_recognition.emotion import EmotionAnalyzer
    
    analyzer = EmotionAnalyzer()
    emotions = analyzer.analyze(face_image)
    # emotions = {"happy": 0.7, "neutral": 0.2, ..., "dominant": "happy"}
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, Optional
from .config import Config

# Add deepface-onnx to path
_lib_path = Path(__file__).parent.parent / "lib" / "deepface_onnx"
if str(_lib_path) not in sys.path:
    sys.path.insert(0, str(_lib_path))


class EmotionAnalyzer:
    """
    Emotion analyzer using ONNX-based model.
    
    Much lighter than TensorFlow-based DeepFace:
    - Model size: 6MB
    - RAM usage: ~20MB
    - Same accuracy for emotion classification
    
    Note: Downloads model on first use (~6MB from Google Drive).
    """
    
    # Emotion labels in model output order
    EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
    INPUT_SIZE = (48, 48)  # Emotion model input size
    
    def __init__(self):
        """Initialize the emotion analyzer."""
        self.enabled = Config.ENABLE_EMOTION_ANALYSIS
        self.session = None
        self.input_name = None
        self.output_name = None
        
        if self.enabled:
            self._load_model()
    
    def _load_model(self):
        """Load the ONNX emotion model."""
        try:
            # Change to lib directory for proper model download path
            original_cwd = os.getcwd()
            os.chdir(str(_lib_path))
            
            try:
                from utils import FaceAnalysis
                
                # Load emotion model (downloads if not present)
                fa = FaceAnalysis(model_name="emotion")
                
                # Store session for direct use
                self.session = fa.model
                self.input_name = fa.input_name
                self.output_name = fa.output_name
                
                print("✓ Emotion analyzer loaded: ONNX (6MB, lightweight)")
            finally:
                os.chdir(original_cwd)
                
        except Exception as e:
            print(f"⚠ Failed to load ONNX emotion model: {e}")
            print("  Emotion analysis will be disabled.")
            self.enabled = False
    
    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for emotion model.
        
        Steps:
        1. Convert to grayscale
        2. Resize to 48x48
        3. Normalize to [0, 1]
        4. Add batch and channel dimensions
        
        Args:
            face_image: Face image (BGR format)
        
        Returns:
            Preprocessed image ready for inference
        """
        # Convert to grayscale
        if len(face_image.shape) == 3:
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_image
        
        # Resize to expected input size
        resized = cv2.resize(gray, self.INPUT_SIZE)
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch and channel dimensions: (48, 48) -> (1, 48, 48, 1)
        batched = np.expand_dims(normalized, axis=0)  # (1, 48, 48)
        batched = np.expand_dims(batched, axis=3)      # (1, 48, 48, 1)
        
        return batched
    
    def analyze(self, face_image: np.ndarray) -> Dict[str, float]:
        """
        Analyze emotions in a face image.
        
        Args:
            face_image: Cropped face image (BGR format)
        
        Returns:
            Dictionary with emotion scores and dominant emotion:
            {
                "angry": 0.01,
                "disgust": 0.01,
                "fear": 0.01,
                "happy": 0.85,
                "sad": 0.02,
                "surprise": 0.05,
                "neutral": 0.05,
                "dominant": "happy"
            }
        """
        if not self.enabled or self.session is None:
            return self._empty_result()
        
        try:
            # Preprocess
            input_data = self.preprocess(face_image)
            
            # Run inference
            outputs = self.session.run(
                [self.output_name],
                {self.input_name: input_data}
            )
            
            # Get probabilities (apply softmax if needed)
            probs = outputs[0][0]
            
            # Some models output logits, apply softmax
            if probs.min() < 0 or probs.max() > 1:
                # Softmax
                exp_probs = np.exp(probs - np.max(probs))
                probs = exp_probs / exp_probs.sum()
            
            # Create emotion dict
            emotions = {
                label: float(prob)
                for label, prob in zip(self.EMOTION_LABELS, probs)
            }
            
            # Add dominant emotion
            emotions['dominant'] = max(emotions, key=lambda k: emotions[k] if k != 'dominant' else 0)
            
            return emotions
            
        except Exception as e:
            # Return empty result on any error
            # Emotion analysis is non-critical
            print(f"⚠ Emotion analysis failed: {e}")
            return self._empty_result()
    
    def _empty_result(self) -> Dict[str, float]:
        """Return empty emotion result when analysis is not available."""
        return {
            "angry": 0.0,
            "disgust": 0.0,
            "fear": 0.0,
            "happy": 0.0,
            "sad": 0.0,
            "surprise": 0.0,
            "neutral": 1.0,  # Default to neutral
            "dominant": "neutral"
        }


# Singleton instance for reuse
_analyzer_instance = None

def get_emotion_analyzer() -> EmotionAnalyzer:
    """Get or create the singleton emotion analyzer instance."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = EmotionAnalyzer()
    return _analyzer_instance
