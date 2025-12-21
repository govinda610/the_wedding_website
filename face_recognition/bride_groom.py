"""
Bride/Groom Detection Module
============================

Provides functionality to identify photos containing the bride or groom
by matching face embeddings against reference images.

USAGE:
------
    from face_recognition.bride_groom import get_bride_groom_matcher
    
    matcher = get_bride_groom_matcher()
    
    # Check if an embedding matches bride/groom
    is_bride = matcher.is_bride(embedding, threshold=0.5)
    is_groom = matcher.is_groom(embedding, threshold=0.5)
    
    # Get all indexed image_ids containing bride/groom
    bride_images = matcher.get_bride_image_ids()
    groom_images = matcher.get_groom_image_ids()
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from .config import Config, DATA_DIR
import logging

logger = logging.getLogger(__name__)

# Singleton instance
_matcher_instance = None


def get_bride_groom_matcher():
    """Get singleton instance of BrideGroomMatcher."""
    global _matcher_instance
    if _matcher_instance is None:
        _matcher_instance = BrideGroomMatcher()
    return _matcher_instance


class BrideGroomMatcher:
    """
    Matches faces against bride/groom reference embeddings.
    
    Uses pre-computed average embeddings from reference images.
    Caches results for efficiency.
    """
    
    # Reference images directory
    REFERENCE_DIR = DATA_DIR / "groom_bride_images"
    
    # Matching threshold (slightly lower than search to catch variations)
    MATCH_THRESHOLD = 0.45
    
    @classmethod
    def get_cache_file(cls) -> Path:
        """Get embeddings cache file path."""
        return DATA_DIR / "reference" / "bride_groom_embeddings.npz"
    
    @classmethod
    def get_presence_cache_file(cls) -> Path:
        """Get presence cache file path."""
        return DATA_DIR / "reference" / "bride_groom_presence.json"
    
    def __init__(self):
        """Initialize matcher - will lazily load embeddings on first use."""
        self.bride_embedding: Optional[np.ndarray] = None
        self.groom_embedding: Optional[np.ndarray] = None
        self._loaded = False
        
        # Pre-computed sets of image_ids containing bride/groom
        self._bride_image_ids: Optional[Set[str]] = None
        self._groom_image_ids: Optional[Set[str]] = None
        
        # Set model-specific cache paths
        self.cache_file = self.get_cache_file()
        self.presence_cache_file = self.get_presence_cache_file()
        
        model_name = "FaceNet512"
        logger.debug(f"BrideGroomMatcher initialized for {model_name}")
    
    def _ensure_loaded(self):
        """Load embeddings if not already loaded."""
        if self._loaded:
            return
        
        # Try to load from cache first
        if self.cache_file.exists():
            self._load_from_cache()
        else:
            # Compute from reference images
            self._compute_embeddings()
        
        self._loaded = True
    
    def _load_from_cache(self):
        """Load pre-computed embeddings from cache file."""
        try:
            data = np.load(str(self.cache_file))
            self.bride_embedding = data['bride']
            self.groom_embedding = data['groom']
            model_name = "FaceNet512"
            print(f"✓ Loaded bride/groom embeddings from cache ({model_name})")
        except Exception as e:
            print(f"Warning: Failed to load cache: {e}")
            self._compute_embeddings()
    
    def _compute_embeddings(self):
        """Compute average embeddings from reference images."""
        from .detector import get_detector
        from .embedder import get_embedder
        
        detector = get_detector()
        embedder = get_embedder()
        
        def get_average_embedding(image_pattern: str) -> Optional[np.ndarray]:
            """Get average embedding from matching images."""
            embeddings = []
            
            for img_path in self.REFERENCE_DIR.glob(image_pattern):
                try:
                    # Read image
                    with open(img_path, 'rb') as f:
                        image_bytes = f.read()
                    
                    # Detect faces
                    image, faces = detector.detect_from_bytes(image_bytes)
                    
                    if not faces:
                        print(f"  Warning: No face in {img_path.name}")
                        continue
                    
                    # Use largest face
                    best_face = max(faces, key=lambda f: f["bbox"][2] * f["bbox"][3])
                    
                    # Align face to 160x160 for FaceNet512
                    aligned_face = detector.align_face(image, best_face, output_size=(160, 160))
                    
                    # Get embedding from aligned face
                    embedding = embedder.get_embedding(aligned_face)
                    embeddings.append(embedding)
                    print(f"  ✓ Processed {img_path.name}")

                    
                except Exception as e:
                    print(f"  Warning: Failed to process {img_path.name}: {e}")
            
            if not embeddings:
                return None
            
            # Average and normalize
            avg = np.mean(embeddings, axis=0)
            avg = avg / np.linalg.norm(avg)
            return avg
        
        print("Computing bride/groom reference embeddings...")
        
        # Compute bride embedding (bride1.jpeg, bride2.jpeg, etc.)
        print("Processing bride images:")
        self.bride_embedding = get_average_embedding("bride*.jpeg")
        
        # Compute groom embedding
        print("Processing groom images:")
        self.groom_embedding = get_average_embedding("groom*.jpeg")
        
        # Save to cache
        self._save_to_cache()
    
    def _save_to_cache(self):
        """Save computed embeddings to cache file."""
        try:
            # Create directory if needed
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            np.savez(
                str(self.cache_file),
                bride=self.bride_embedding if self.bride_embedding is not None else np.zeros(512),
                groom=self.groom_embedding if self.groom_embedding is not None else np.zeros(512)
            )
            model_name = "FaceNet512"
            print(f"✓ Saved embeddings to {self.cache_file} ({model_name})")
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")
    
    def is_bride(self, embedding: np.ndarray, threshold: float = None) -> bool:
        """Check if an embedding matches the bride."""
        self._ensure_loaded()
        
        if self.bride_embedding is None:
            return False
        
        threshold = threshold or self.MATCH_THRESHOLD
        similarity = float(np.dot(embedding, self.bride_embedding))
        return similarity >= threshold
    
    def is_groom(self, embedding: np.ndarray, threshold: float = None) -> bool:
        """Check if an embedding matches the groom."""
        self._ensure_loaded()
        
        if self.groom_embedding is None:
            return False
        
        threshold = threshold or self.MATCH_THRESHOLD
        similarity = float(np.dot(embedding, self.groom_embedding))
        return similarity >= threshold
    
    def check_embedding(self, embedding: np.ndarray, threshold: float = None) -> Dict[str, bool]:
        """Check if embedding matches bride, groom, or both."""
        return {
            "is_bride": self.is_bride(embedding, threshold),
            "is_groom": self.is_groom(embedding, threshold)
        }
    
    def compute_image_presence(self):
        """
        Pre-compute which indexed images contain bride/groom.
        
        This scans ALL indexed face embeddings and caches which image_ids
        contain bride/groom faces. Heavy operation - run once.
        """
        from .search import get_index
        import faiss
        
        self._ensure_loaded()
        
        index = get_index()
        
        self._bride_image_ids = set()
        self._groom_image_ids = set()
        
        print("Pre-computing bride/groom presence in all indexed images...")
        
        # Check if index exists and has faces
        if index.index is None or index.index.ntotal == 0:
            print("  Error: Index is empty. Bride/groom filter won't work.")
            return {"bride_images": 0, "groom_images": 0}
        
        total_faces = index.index.ntotal
        print(f"  Scanning {total_faces} face embeddings...")
        
        # Get all face metadata
        for i, meta in enumerate(index.face_metadata):
            if i >= total_faces:
                break
            
            # Use FAISS reconstruct to get the embedding back from index
            try:
                embedding = index.index.reconstruct(i)
            except Exception as e:
                print(f"  Warning: Failed to reconstruct embedding {i}: {e}")
                continue
            
            image_id = meta.get("image_id", "")
            
            # Check against bride/groom
            if self.is_bride(embedding):
                self._bride_image_ids.add(image_id)
            
            if self.is_groom(embedding):
                self._groom_image_ids.add(image_id)
            
            if (i + 1) % 5000 == 0:
                print(f"  Processed {i + 1}/{total_faces} faces...")
        
        print(f"✓ Found {len(self._bride_image_ids)} images with bride")
        print(f"✓ Found {len(self._groom_image_ids)} images with groom")
        
        # Save to disk for persistence across restarts
        self._save_presence_cache()
        
        return {
            "bride_images": len(self._bride_image_ids),
            "groom_images": len(self._groom_image_ids)
        }
    
    def _save_presence_cache(self):
        """Save computed image ID sets to disk for persistence."""
        import json
        try:
            self.presence_cache_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "bride_image_ids": list(self._bride_image_ids or []),
                "groom_image_ids": list(self._groom_image_ids or [])
            }
            with open(self.presence_cache_file, 'w') as f:
                json.dump(data, f)
            model_name = "FaceNet512"
            print(f"✓ Saved presence cache to {self.presence_cache_file} ({model_name})")
        except Exception as e:
            print(f"Warning: Failed to save presence cache: {e}")
    
    def _load_presence_cache(self) -> bool:
        """Try to load image ID sets from disk cache. Returns True if successful."""
        import json
        if not self.presence_cache_file.exists():
            return False
        try:
            with open(self.presence_cache_file, 'r') as f:
                data = json.load(f)
            self._bride_image_ids = set(data.get("bride_image_ids", []))
            self._groom_image_ids = set(data.get("groom_image_ids", []))
            model_name = "FaceNet512"
            print(f"✓ Loaded presence cache ({model_name}): {len(self._bride_image_ids)} bride, {len(self._groom_image_ids)} groom images")
            return True
        except Exception as e:
            print(f"Warning: Failed to load presence cache: {e}")
            return False
    
    def get_bride_image_ids(self) -> Set[str]:
        """Get set of image_ids containing the bride."""
        if self._bride_image_ids is None:
            # Try loading from cache first
            if not self._load_presence_cache():
                self.compute_image_presence()
        return self._bride_image_ids
    
    def get_groom_image_ids(self) -> Set[str]:
        """Get set of image_ids containing the groom."""
        if self._groom_image_ids is None:
            # Try loading from cache first (will also load bride if not already)
            if not self._load_presence_cache():
                self.compute_image_presence()
        return self._groom_image_ids
    
    def annotate_matches(self, matches: List[Dict]) -> List[Dict]:
        """
        Add has_bride/has_groom flags to search result matches.
        
        Args:
            matches: List of match dicts from search
        
        Returns:
            Same matches with has_bride/has_groom added
        """
        bride_ids = self.get_bride_image_ids()
        groom_ids = self.get_groom_image_ids()
        
        for match in matches:
            image_id = match.get("image_id", "")
            match["has_bride"] = image_id in bride_ids
            match["has_groom"] = image_id in groom_ids
        
        return matches
