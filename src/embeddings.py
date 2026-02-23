"""
Image embedding utilities using CLIP model.
Converts images into high-dimensional vector representations for similarity search.
"""

from sentence_transformers import SentenceTransformer
from typing import List
from PIL import Image


class ImageEmbedder:
    """Handles image embedding generation using CLIP model."""

    def __init__(self, model_name: str, dimension: int):
        """
        Initialize the Sentence Transformers model for image embeddings.

        Args:
            model_name: Name of the sentence-transformers model
            dimension: Dimension of the embedding
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = dimension
    
    def get_image_embedding(self, image_path: str) -> List[float]:
        """
        Generate embedding vector for an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of float values representing the image embedding
        """
        # TODO
        # Load image
        image = Image.open(image_path)
        
        # Generate embedding (returns numpy array)
        embedding = self.model.encode(image, convert_to_numpy=True)
        
        # Convert to list of floats
        return embedding.tolist()

if __name__ == "__main__":
    # Initialize embedder
    embedder = ImageEmbedder(model_name='clip-ViT-B-32', dimension=512)
    
    image_path = "static/clownfish.jpeg"

    embedding = embedder.get_image_embedding(image_path)
    print(f"Embedding vector (dimension: {len(embedding)}):")
    print(embedding)