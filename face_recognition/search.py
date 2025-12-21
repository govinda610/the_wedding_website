"""
FAISS Search Module
===================

This module handles similarity search using FAISS (Facebook AI Similarity Search).
It stores face embeddings and enables fast retrieval of similar faces.

EDUCATIONAL NOTES:
------------------
FAISS is a library for efficient similarity search and clustering of dense vectors.
It's used by Facebook/Meta for searching billions of images.

For our wedding photos (~4,000 faces), we use IndexFlatIP:
- "Flat" = stores all vectors in a flat array (exact search)
- "IP" = Inner Product (for cosine similarity on normalized vectors)

This gives 100% accuracy and is fast enough for 4,000 vectors (~5ms search).

THRESHOLD-BASED SEARCH:
-----------------------
Instead of returning top-K results, we return ALL faces above a similarity threshold.
This is important because one person can appear in 200+ wedding photos!

USAGE:
------
    from face_recognition.search import FaceIndex
    
    index = FaceIndex()
    index.add_embeddings(embeddings, face_ids)
    
    # Search returns ALL matches above threshold
    matches = index.search(query_embedding, threshold=0.5)
"""

import json
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from .config import Config


class FaceIndex:
    """
    FAISS-based face index for similarity search.
    
    Uses IndexFlatIP (exact search with inner product) which gives
    100% accuracy and is fast enough for ~10,000 vectors.
    
    The index stores embeddings and maps them to face metadata.
    """
    
    def __init__(self):
        """Initialize the face index."""
        self.index = None
        self.face_metadata = []
        self.embeddings = None
        self._is_loaded = False
        
        # Try to load existing index
        self._try_load()
    
    def _try_load(self):
        """Try to load existing index and metadata from disk."""
        index_path = Config.get_faiss_index_path()
        metadata_path = Config.get_metadata_path()
        
        if index_path.exists() and metadata_path.exists():
            try:
                self.load()
                model_name = "FaceNet512"
                print(f"✓ Face index loaded ({model_name}): {len(self.face_metadata)} faces")
            except Exception as e:
                print(f"⚠ Failed to load existing index: {e}")
                print("  Will create new index when embeddings are added.")
    
    def create_index(self, dimension: int = Config.EMBEDDING_DIM):
        """
        Create a new empty FAISS index.
        
        Args:
            dimension: Embedding dimension (512 for ArcFace/MobileFaceNet)
        """
        # IndexFlatIP = Flat index with Inner Product
        # When vectors are L2-normalized, IP = cosine similarity
        self.index = faiss.IndexFlatIP(dimension)
        self.face_metadata = []
        self.embeddings = None
        self._is_loaded = False
        
        print(f"✓ Created new FAISS index (dim={dimension})")
    
    def add_face(
        self, 
        embedding: np.ndarray, 
        image_id: str,
        bbox: List[int],
        emotions: Dict[str, float],
        event: str = "",
        date: str = ""
    ) -> int:
        """
        Add a single face to the index.
        
        Args:
            embedding: 512-dim face embedding (normalized)
            image_id: Google Drive file ID of the source image
            bbox: Bounding box [x, y, w, h]
            emotions: Emotion scores from analyzer
            event: Event name (e.g., "sangeet", "wedding")
            date: Date string
        
        Returns:
            Face ID (index in the metadata list)
        """
        if self.index is None:
            self.create_index()
        
        # Ensure embedding is 2D for FAISS
        embedding = embedding.reshape(1, -1).astype(np.float32)
        
        # Normalize (should already be normalized, but ensure it)
        faiss.normalize_L2(embedding)
        
        # Add to FAISS index
        self.index.add(embedding)
        
        # Store metadata
        face_id = len(self.face_metadata)
        self.face_metadata.append({
            "face_id": face_id,
            "image_id": image_id,
            "bbox": bbox,
            "emotions": emotions,
            "event": event,
            "date": date
        })
        
        return face_id
    
    def add_embeddings_batch(
        self,
        embeddings: np.ndarray,
        metadata_list: List[Dict]
    ):
        """
        Add multiple embeddings at once (more efficient).
        
        Args:
            embeddings: Array of embeddings, shape (N, 512)
            metadata_list: List of metadata dicts for each face
        """
        if len(embeddings) != len(metadata_list):
            raise ValueError("Embeddings and metadata must have same length")
        
        if self.index is None:
            self.create_index()
        
        # Ensure embeddings are float32 and normalized
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)
        
        # Add to FAISS
        self.index.add(embeddings)
        
        # Add face IDs to metadata
        start_id = len(self.face_metadata)
        for i, meta in enumerate(metadata_list):
            meta["face_id"] = start_id + i
            self.face_metadata.append(meta)
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        threshold: float = Config.get_similarity_threshold()
    ) -> List[Dict]:
        """
        Search for all faces similar to query, above threshold.
        
        This returns ALL matches above threshold, not top-K.
        A person could appear in 200+ photos - we return ALL of them.
        
        Applies multi-face filtering: stricter threshold for images with many faces
        to reduce false positives from crowd scenes.
        
        Args:
            query_embedding: 512-dim query embedding
            threshold: Minimum cosine similarity (default 0.55)
        
        Returns:
            List of matching faces with similarity scores, sorted by similarity
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # Ensure query is 2D and normalized
        query = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)
        
        # Search ALL vectors (k = total count)
        # This is fast for 4,000 vectors (~5ms)
        k = self.index.ntotal
        distances, indices = self.index.search(query, k)
        
        # Count faces per image for multi-face filtering
        face_count_per_image = {}
        for meta in self.face_metadata:
            img_id = meta.get("image_id", "")
            face_count_per_image[img_id] = face_count_per_image.get(img_id, 0) + 1
        
        # Filter by threshold and collect matches
        matches = []
        for dist, idx in zip(distances[0], indices[0]):
            # Skip invalid indices (FAISS returns -1 for empty slots)
            if idx < 0 or idx >= len(self.face_metadata):
                continue
            
            # Get metadata
            meta = self.face_metadata[idx].copy()
            img_id = meta.get("image_id", "")
            
            # Apply stricter threshold for images with many faces
            # This reduces false positives from crowded scenes
            num_faces_in_image = face_count_per_image.get(img_id, 1)
            effective_threshold = threshold
            if num_faces_in_image > Config.MULTI_FACE_COUNT:
                effective_threshold = max(threshold, Config.get_multi_face_threshold())
            
            if dist >= effective_threshold:
                meta["similarity"] = float(dist)
                meta["face_count_in_image"] = num_faces_in_image
                matches.append(meta)
        
        # Sort by similarity (highest first)
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        
        return matches
    
    def search_multiple(
        self,
        query_embeddings: np.ndarray,
        threshold: float = Config.get_similarity_threshold()
    ) -> List[Dict]:
        """
        Search for faces matching ANY of multiple query embeddings.
        
        Used when user uploads a photo with multiple faces.
        Returns union of all matches, deduplicated by image_id.
        
        Args:
            query_embeddings: Array of query embeddings, shape (N, 512)
            threshold: Minimum cosine similarity
        
        Returns:
            Deduplicated list of matching images
        """
        all_matches = []
        for embedding in query_embeddings:
            matches = self.search(embedding, threshold)
            all_matches.extend(matches)
        
        # Deduplicate by image_id, keeping highest similarity
        seen = {}
        for match in all_matches:
            img_id = match["image_id"]
            if img_id not in seen or match["similarity"] > seen[img_id]["similarity"]:
                seen[img_id] = match
        
        # Sort by similarity
        results = list(seen.values())
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        return results
    
    def save(self):
        """Save index and metadata to disk."""
        if self.index is None:
            print("⚠ No index to save")
            return
        
        index_path = Config.get_faiss_index_path()
        metadata_path = Config.get_metadata_path()
        
        # Create data directory if needed
        index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(index_path))
        
        # Save metadata as JSON
        with open(metadata_path, 'w') as f:
            json.dump({
                "faces": self.face_metadata,
                "total_count": len(self.face_metadata)
            }, f, indent=2)
        
        model_name = "FaceNet512"
        print(f"✓ Saved index ({model_name}) with {len(self.face_metadata)} faces")
    
    def load(self):
        """Load index and metadata from disk."""
        index_path = Config.get_faiss_index_path()
        metadata_path = Config.get_metadata_path()
        
        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_path))
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            data = json.load(f)
            self.face_metadata = data["faces"]
        
        self._is_loaded = True
    
    def rebuild(self, embeddings: np.ndarray, metadata_list: List[Dict]):
        """
        Rebuild the entire index from scratch.
        
        Useful when you want to completely replace the index.
        
        Args:
            embeddings: All embeddings
            metadata_list: All metadata
        """
        self.create_index()
        self.add_embeddings_batch(embeddings, metadata_list)
        self.save()
    
    @property
    def total_faces(self) -> int:
        """Get total number of faces in the index."""
        if self.index is None:
            return 0
        return self.index.ntotal


# Singleton instance for reuse
_index_instance = None

def get_index() -> FaceIndex:
    """Get or create the singleton face index instance."""
    global _index_instance
    if _index_instance is None:
        _index_instance = FaceIndex()
    return _index_instance
