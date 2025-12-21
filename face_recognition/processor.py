"""
Face Processor Module
=====================

This is the main entry point for face recognition operations.
It combines detection, embedding, emotion analysis, and search.

USAGE:
------
    from face_recognition.processor import FaceProcessor
    
    # Create processor
    processor = FaceProcessor()
    
    # Process a query image (returns matching photos)
    results = processor.search_faces(image_bytes, threshold=0.5)
    
    # Process a wedding photo for indexing
    faces = processor.process_image(image_bytes, image_id="drive_file_id")
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from .config import Config
from .detector import FaceDetector, get_detector
from .embedder import FaceEmbedder, get_embedder
from .emotion import EmotionAnalyzer, get_emotion_analyzer
from .search import FaceIndex, get_index


class FaceProcessor:
    """
    Main face processing class.
    
    Combines all face recognition components:
    - Detection (YuNet)
    - Embedding (ArcFace/MobileFaceNet)
    - Emotion analysis (DeepFace)
    - Similarity search (FAISS)
    
    Designed for both:
    - Ingestion: Processing wedding photos to build the index
    - Query: Finding photos matching a user's face
    """
    
    def __init__(self):
        """Initialize the face processor with all components."""
        self.detector = get_detector()
        self.embedder = get_embedder()
        self.emotion_analyzer = get_emotion_analyzer()
        self.index = get_index()
    
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in an image.
        
        Args:
            image: OpenCV image (BGR format)
        
        Returns:
            List of detected faces with bounding boxes and landmarks
        """
        return self.detector.detect(image)
    
    def get_face_embedding(
        self, 
        image: np.ndarray, 
        face: Dict
    ) -> np.ndarray:
        """
        Get embedding for a detected face.
        
        Args:
            image: Full image
            face: Face dict with landmarks
        
        Returns:
            512-dimensional embedding vector
        """
        # Align face to 160x160 for FaceNet512
        aligned = self.detector.align_face(image, face, output_size=(160, 160))
        
        # Get embedding
        return self.embedder.get_embedding(aligned)
    
    def analyze_emotion(
        self, 
        image: np.ndarray, 
        face: Dict
    ) -> Dict[str, float]:
        """
        Analyze emotions for a detected face.
        
        Args:
            image: Full image
            face: Face dict with bbox
        
        Returns:
            Emotion scores
        """
        # Crop face for emotion analysis
        cropped = self.detector.crop_face(image, face, padding=0.1)
        
        # Analyze emotions
        return self.emotion_analyzer.analyze(cropped)
    
    def process_image(
        self,
        image: np.ndarray,
        image_id: str,
        event: str = "",
        date: str = "",
        add_to_index: bool = True
    ) -> List[Dict]:
        """
        Process an image for indexing: detect, embed, analyze, and store.
        
        Used during ingestion to process wedding photos.
        
        Args:
            image: OpenCV image (BGR format)
            image_id: Google Drive file ID
            event: Event name
            date: Date string
            add_to_index: Whether to add to FAISS index
        
        Returns:
            List of processed faces with embeddings and emotions
        """
        # Detect faces
        faces = self.detect_faces(image)
        
        processed_faces = []
        for face in faces:
            # Get embedding
            embedding = self.get_face_embedding(image, face)
            
            # Analyze emotion
            emotions = self.analyze_emotion(image, face)
            
            face_data = {
                "bbox": face["bbox"],
                "confidence": face["confidence"],
                "embedding": embedding,
                "emotions": emotions,
                "image_id": image_id,
                "event": event,
                "date": date
            }
            
            # Add to index if requested
            if add_to_index:
                face_id = self.index.add_face(
                    embedding=embedding,
                    image_id=image_id,
                    bbox=face["bbox"],
                    emotions=emotions,
                    event=event,
                    date=date
                )
                face_data["face_id"] = face_id
            
            processed_faces.append(face_data)
        
        return processed_faces
    
    def process_image_bytes(
        self,
        image_bytes: bytes,
        image_id: str,
        event: str = "",
        date: str = "",
        add_to_index: bool = True
    ) -> List[Dict]:
        """
        Process image from bytes (convenience wrapper).
        
        Args:
            image_bytes: Raw image bytes
            image_id: Google Drive file ID
            event: Event name
            date: Date string
            add_to_index: Whether to add to FAISS index
        
        Returns:
            List of processed faces
        """
        image, _ = self.detector.detect_from_bytes(image_bytes)
        return self.process_image(image, image_id, event, date, add_to_index)
    
    def search_faces(
        self,
        image_bytes: bytes,
        threshold: float = Config.get_similarity_threshold()
    ) -> Dict:
        """
        Search for matching photos given a query image.
        
        This is the main search function called when a user uploads a photo.
        Returns ALL matching photos above the threshold.
        
        Args:
            image_bytes: Query image bytes
            threshold: Similarity threshold
        
        Returns:
            Dict with:
            - faces: List of detected faces in query
            - matches: List of matching photos (deduplicated)
            - total_count: Number of matching photos
        """
        # Decode image and detect faces
        image, faces = self.detector.detect_from_bytes(image_bytes)
        
        if not faces:
            return {
                "faces": [],
                "matches": [],
                "total_count": 0,
                "message": "No faces detected in the query image"
            }
        
        # Get embeddings for all detected faces
        embeddings = []
        face_info = []
        for face in faces:
            embedding = self.get_face_embedding(image, face)
            embeddings.append(embedding)
            
            # Get face thumbnail as base64 (for UI if multiple faces)
            cropped = self.detector.crop_face(image, face)
            face_info.append({
                "bbox": face["bbox"],
                "confidence": face["confidence"]
            })
        
        # Search for matches
        if len(embeddings) == 1:
            matches = self.index.search(embeddings[0], threshold)
        else:
            # Multiple faces: search for all, return union
            embeddings_array = np.array(embeddings)
            matches = self.index.search_multiple(embeddings_array, threshold)
        
        return {
            "faces": face_info,
            "face_count": len(faces),
            "matches": matches,
            "total_count": len(matches),
            "threshold": threshold
        }
    
    def search_faces_multi_image(
        self,
        image_bytes_list: List[bytes],
        threshold: float = Config.get_similarity_threshold()
    ) -> Dict:
        """
        HYBRID multi-image search for better accuracy.
        
        Combines THREE strategies:
        1. Mean embedding: Averages all query embeddings to smooth variance (reduces FN)
        2. Union search: Searches with each individual embedding (reduces FN)
        3. Voting: Counts how many queries match each result (reduces FP)
        
        Final filter: Keep results that either:
        - Matched the mean embedding, OR
        - Got votes from ≥2 individual queries, OR
        - Have very high similarity (>0.65)
        
        Args:
            image_bytes_list: List of query image bytes (1-5 images)
            threshold: Minimum similarity threshold
        
        Returns:
            Dict with combined results, voting info, and debug data
        """
        import faiss
        
        all_embeddings = []
        all_face_info = []
        
        # Step 1: Extract best face embedding from each image
        for i, image_bytes in enumerate(image_bytes_list):
            try:
                image, faces = self.detector.detect_from_bytes(image_bytes)
                
                if not faces:
                    continue
                
                # Use the largest face (most likely the subject)
                best_face = max(faces, key=lambda f: f["bbox"][2] * f["bbox"][3])
                embedding = self.get_face_embedding(image, best_face)
                all_embeddings.append(embedding)
                all_face_info.append({
                    "image_index": i,
                    "bbox": best_face["bbox"],
                    "confidence": best_face["confidence"],
                    "size": best_face["bbox"][2] * best_face["bbox"][3]
                })
            except Exception as e:
                print(f"Warning: Failed to process query image {i}: {e}")
                continue
        
        if not all_embeddings:
            return {
                "faces": [],
                "face_count": 0,
                "matches": [],
                "total_count": 0,
                "message": "No faces detected in any query images",
                "threshold": threshold
            }
        
        num_queries = len(all_embeddings)
        
        # Step 2: Create MEAN embedding (variance smoothing)
        embeddings_array = np.array(all_embeddings)
        mean_embedding = np.mean(embeddings_array, axis=0)
        # Re-normalize the mean embedding (important for cosine similarity!)
        mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
        
        # Step 3: Search with MEAN embedding
        mean_matches = self.index.search(mean_embedding, threshold)
        mean_matched_ids = set(m["image_id"] for m in mean_matches)
        
        # Step 4: Search with EACH individual embedding and count votes
        all_matches = {}  # image_id -> {votes, max_similarity, meta}
        
        for emb in all_embeddings:
            matches = self.index.search(emb, threshold)
            for match in matches:
                img_id = match["image_id"]
                if img_id not in all_matches:
                    all_matches[img_id] = {
                        "votes": 0,
                        "max_similarity": 0,
                        "meta": match
                    }
                all_matches[img_id]["votes"] += 1
                if match["similarity"] > all_matches[img_id]["max_similarity"]:
                    all_matches[img_id]["max_similarity"] = match["similarity"]
                    all_matches[img_id]["meta"] = match
        
        # Step 5: Apply hybrid filter
        # Keep if: matched_mean OR votes>=2 OR very high similarity
        final_matches = []
        for img_id, data in all_matches.items():
            votes = data["votes"]
            similarity = data["max_similarity"]
            matched_mean = img_id in mean_matched_ids
            
            # Hybrid filter logic
            keep = False
            reason = ""
            
            if matched_mean:
                keep = True
                reason = "mean_match"
            elif votes >= 2:
                keep = True
                reason = f"votes_{votes}"
            elif similarity >= 0.65:
                keep = True
                reason = "high_similarity"
            elif num_queries == 1:
                # Single image: keep all matches (fallback)
                keep = True
                reason = "single_query"
            
            if keep:
                meta = data["meta"].copy()
                meta["vote_count"] = votes
                meta["vote_fraction"] = votes / num_queries
                meta["matched_mean"] = matched_mean
                meta["match_reason"] = reason
                
                # Combined score: similarity + consensus bonus
                consensus_bonus = 0.1 * (votes / num_queries)
                mean_bonus = 0.05 if matched_mean else 0
                meta["combined_score"] = similarity + consensus_bonus + mean_bonus
                
                final_matches.append(meta)
        
        # Sort by combined score (best matches first)
        final_matches.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return {
            "faces": all_face_info,
            "face_count": num_queries,
            "query_images_used": num_queries,
            "matches": final_matches,
            "total_count": len(final_matches),
            "threshold": threshold,
            "strategy": "hybrid_mean_union_voting",
            "mean_matches_count": len(mean_matched_ids),
            "debug": {
                "total_union_matches": len(all_matches),
                "mean_embedding_matches": len(mean_matched_ids),
                "filtered_matches": len(final_matches)
            }
        }
    
    def save_index(self):
        """Save the current index to disk."""
        self.index.save()
    
    def get_stats(self) -> Dict:
        """Get statistics about the current index."""
        return {
            "total_faces": self.index.total_faces,
            "embedder_model": Config.EMBEDDER_MODEL,
            "similarity_threshold": Config.get_similarity_threshold(),
            "emotion_analysis_enabled": Config.ENABLE_EMOTION_ANALYSIS
        }


# Singleton instance
_processor_instance = None

def get_processor() -> FaceProcessor:
    """Get or create the singleton face processor instance."""
    global _processor_instance
    if _processor_instance is None:
        _processor_instance = FaceProcessor()
    return _processor_instance
