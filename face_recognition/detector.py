"""
Face Detector Module
====================

This module provides face detection using YuNet (OpenCV's fast face detector).
YuNet is much faster than RetinaFace on CPU (~50ms vs 250ms per image).

EDUCATIONAL NOTES:
------------------
Face detection finds bounding boxes around faces in an image. It does NOT
identify who the face belongs to - that's the job of the embedder.

YuNet outputs:
- Bounding box: [x, y, width, height]
- Confidence score: 0-1
- 5 Facial landmarks: right eye, left eye, nose tip, right mouth corner, left mouth corner

The landmarks are used for face alignment before embedding generation.

USAGE:
------
    from face_recognition.detector import FaceDetector
    
    detector = FaceDetector()
    faces = detector.detect(image)
    # faces = [{"bbox": [x,y,w,h], "confidence": 0.99, "landmarks": [...]}]
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from .config import Config


class FaceDetector:
    """
    Face detector using YuNet (OpenCV DNN).
    
    YuNet is a lightweight face detection model designed for real-time
    applications. It's part of OpenCV's model zoo.
    
    Performance on CPU:
    - ~50ms per image at 640x480
    - Accurate enough for wedding photos
    - Detects multiple faces per image
    """
    
    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialize the face detector.
        
        Args:
            model_path: Path to YuNet ONNX model. Uses default if not specified.
        """
        self.model_path = model_path or Config.YUNET_MODEL_PATH
        self.detector = None
        self.input_size = (640, 480)  # Default input size
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the YuNet face detection model."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"YuNet model not found at {self.model_path}. "
                f"Download it from: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet"
            )
        
        # Create YuNet detector
        # OpenCV's FaceDetectorYN handles the model loading
        self.detector = cv2.FaceDetectorYN.create(
            str(self.model_path),
            "",  # No config file needed for ONNX
            self.input_size,
            Config.DETECTION_CONFIDENCE_THRESHOLD,
            0.3,  # NMS threshold (non-max suppression)
            5000  # Top K before NMS
        )
        
        print(f"✓ Face detector loaded: YuNet from {self.model_path.name}")
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in an image.
        
        Args:
            image: OpenCV image (BGR format, numpy array)
        
        Returns:
            List of detected faces, each with:
            - bbox: [x, y, width, height]
            - confidence: detection confidence (0-1)
            - landmarks: 5 facial landmarks for alignment
        """
        if image is None or image.size == 0:
            return []
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Update detector input size to match image
        self.detector.setInputSize((width, height))
        
        # Detect faces
        # Returns: N x 15 array
        # Columns: x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score
        # Where: re=right eye, le=left eye, nt=nose tip, rcm=right corner mouth, lcm=left corner mouth
        _, faces = self.detector.detect(image)
        
        if faces is None:
            return []
        
        results = []
        for face in faces:
            # Extract bounding box
            x, y, w, h = face[:4].astype(int)
            
            # Filter out tiny faces
            if w < Config.MIN_FACE_SIZE or h < Config.MIN_FACE_SIZE:
                continue
            
            # Extract landmarks (5 points, 2D)
            landmarks = face[4:14].reshape(5, 2)
            
            # Confidence score
            confidence = float(face[14])
            
            # Filter by confidence
            if confidence < Config.DETECTION_CONFIDENCE_THRESHOLD:
                continue
            
            results.append({
                "bbox": [int(x), int(y), int(w), int(h)],
                "confidence": confidence,
                "landmarks": landmarks.tolist(),
                # Landmark names for reference:
                # 0: right eye, 1: left eye, 2: nose tip, 3: right mouth, 4: left mouth
            })
        
        return results
    
    def detect_from_bytes(self, image_bytes: bytes) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect faces from image bytes.
        
        Handles:
        - Image decoding
        - EXIF orientation correction
        - Auto-scaling for large images (YuNet works best under 2MP)
        
        Args:
            image_bytes: Raw image bytes (JPEG, PNG, etc.)
        
        Returns:
            Tuple of (image as numpy array, list of detected faces)
        """
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        # Handle EXIF orientation for JPEG images
        # WhatsApp images often have orientation metadata
        try:
            image = self._fix_exif_orientation(image, image_bytes)
        except Exception:
            pass  # If EXIF handling fails, proceed with original
        
        # Scale down very large images for better detection
        # YuNet works best on images under ~1920px on longest side
        MAX_DIM = 1920
        h, w = image.shape[:2]
        scale = 1.0
        
        if max(h, w) > MAX_DIM:
            scale = MAX_DIM / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            faces = self.detect(scaled_image)
            
            # Scale bounding boxes back to original size
            for face in faces:
                face["bbox"] = [
                    int(face["bbox"][0] / scale),
                    int(face["bbox"][1] / scale),
                    int(face["bbox"][2] / scale),
                    int(face["bbox"][3] / scale)
                ]
                # Scale landmarks too
                face["landmarks"] = [[int(x / scale), int(y / scale)] for x, y in face["landmarks"]]
        else:
            faces = self.detect(image)
        
        return image, faces
    
    def _fix_exif_orientation(self, image: np.ndarray, image_bytes: bytes) -> np.ndarray:
        """
        Fix image orientation based on EXIF data.
        
        Many phone cameras store images in a rotated orientation with EXIF
        metadata indicating how to display them. OpenCV ignores this,
        so we need to handle it manually.
        """
        # Quick check for EXIF marker in JPEG
        if len(image_bytes) < 12 or image_bytes[0:2] != b'\xff\xd8':
            return image  # Not JPEG
        
        # Look for orientation in EXIF
        # EXIF orientation values:
        # 1 = normal, 3 = rotate 180, 6 = rotate 90 CW, 8 = rotate 90 CCW
        try:
            # Simple EXIF parsing for orientation
            # Full EXIF parsing would require a library like PIL
            import struct
            
            # Find APP1 marker (EXIF)
            idx = 2
            while idx < min(len(image_bytes), 10000):
                if image_bytes[idx] != 0xFF:
                    break
                marker = image_bytes[idx + 1]
                if marker == 0xE1:  # APP1 (EXIF)
                    # Parse EXIF for orientation
                    exif_data = image_bytes[idx + 4:idx + 4 + struct.unpack('>H', image_bytes[idx + 2:idx + 4])[0]]
                    if exif_data[:4] == b'Exif':
                        # Find orientation tag (0x0112)
                        exif_str = exif_data.hex()
                        if '0112' in exif_str:
                            # Found orientation tag, get value
                            pos = exif_str.find('0112')
                            orientation = int(exif_str[pos + 16:pos + 18], 16)
                            if orientation == 3:
                                return cv2.rotate(image, cv2.ROTATE_180)
                            elif orientation == 6:
                                return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                            elif orientation == 8:
                                return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    break
                else:
                    # Skip to next marker
                    if idx + 3 < len(image_bytes):
                        length = struct.unpack('>H', image_bytes[idx + 2:idx + 4])[0]
                        idx += 2 + length
                    else:
                        break
        except Exception:
            pass
        
        return image
    
    def crop_face(
        self, 
        image: np.ndarray, 
        face: Dict, 
        padding: float = 0.2
    ) -> np.ndarray:
        """
        Crop a face from the image with optional padding.
        
        Args:
            image: Full image
            face: Face dict with bbox
            padding: Padding ratio around the face (0.2 = 20%)
        
        Returns:
            Cropped face image
        """
        x, y, w, h = face["bbox"]
        
        # Add padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        # Calculate padded coordinates
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(image.shape[1], x + w + pad_w)
        y2 = min(image.shape[0], y + h + pad_h)
        
        return image[y1:y2, x1:x2]
    
    def align_face(
        self, 
        image: np.ndarray, 
        face: Dict, 
        output_size: Tuple[int, int] = (160, 160)
    ) -> np.ndarray:
        """
        Align and crop face using landmarks for consistent embedding.
        
        Face alignment ensures the face is upright and centered,
        which significantly improves embedding quality.
        
        Args:
            image: Full image
            face: Face dict with landmarks
            output_size: Size of output aligned face (default 160x160 for FaceNet512)
        
        Returns:
            Aligned and cropped face image
        """
        landmarks = np.array(face["landmarks"])
        
        # Source points (detected landmarks)
        src_pts = landmarks.astype(np.float32)
        
        # Destination points (ideal positions for 160x160)
        # Scaled from 112x112 ArcFace positions by factor of 160/112 = 1.4286
        scale = output_size[0] / 112.0
        dst_pts = np.array([
            [38.2946 * scale, 51.6963 * scale],  # Right eye
            [73.5318 * scale, 51.5014 * scale],  # Left eye
            [56.0252 * scale, 71.7366 * scale],  # Nose tip
            [41.5493 * scale, 92.3655 * scale],  # Right mouth corner
            [70.7299 * scale, 92.2041 * scale]   # Left mouth corner
        ], dtype=np.float32)
        
        # Calculate similarity transform
        tform = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]
        
        if tform is None:
            # Fallback to simple crop if alignment fails
            return self.crop_face(image, face)
        
        # Apply transformation
        aligned = cv2.warpAffine(
            image, 
            tform, 
            output_size,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return aligned


# Singleton instance for reuse
_detector_instance = None

def get_detector() -> FaceDetector:
    """Get or create the singleton face detector instance."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = FaceDetector()
    return _detector_instance
