"""
CLIP Classifier for Object Recognition

Uses OpenAI's CLIP model for zero-shot image classification.
Classifies cropped regions from SAM segmentation masks.
"""

import torch
import clip
from PIL import Image
import numpy as np
import cv2


class CLIPClassifier:
    """
    CLIP-based object classifier for SAM mask regions
    Uses zero-shot classification with predefined categories
    """

    def __init__(self, categories=None):
        """
        Initialize CLIP model

        Args:
            categories: List of object categories to classify
                       If None, uses default common objects
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load CLIP model (ViT-B/32 - balance of speed and accuracy)
        print("Loading CLIP model...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        print(f"CLIP loaded on {self.device}")

        # Default categories for common objects
        if categories is None:
            self.categories = [
                # Vehicles
                "car", "truck", "bus", "motorcycle", "bicycle",
                "van", "vehicle", "taxi", "ambulance",

                # People
                "person", "people", "child", "adult", "worker",
                "pedestrian", "crowd",

                # Traffic/Road
                "traffic light", "stop sign", "road sign", "traffic sign",
                "building", "tree", "sky", "road", "sidewalk",

                # Safety equipment
                "helmet", "safety helmet", "hard hat",
                "safety vest", "high visibility vest",
                "tool", "equipment",

                # Medical
                "medicine", "pill", "tablet", "capsule",
                "medication", "drug", "pharmaceutical",

                # Materials
                "metal sheet", "steel plate", "aluminum",
                "metal", "steel", "iron plate",

                # Packaging
                "package", "box", "container", "crate",
                "carton", "parcel"
            ]
        else:
            self.categories = categories

        print(f"CLIP classifier initialized with {len(self.categories)} categories")

        # Precompute text features for categories
        print("Encoding text prompts...")
        text_inputs = torch.cat([
            clip.tokenize(f"a photo of a {c}") for c in self.categories
        ]).to(self.device)

        with torch.no_grad():
            self.text_features = self.model.encode_text(text_inputs)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

        print("✓ CLIP ready for classification")

    def classify_region(self, image, mask):
        """
        Classify object in masked region

        Args:
            image: Original image (numpy BGR array from cv2)
            mask: Binary mask (numpy array, True where object is)

        Returns:
            {
                'label': str,  # Best matching category
                'confidence': float,  # 0.0-1.0
                'top_3': list  # Top 3 predictions with scores
            }
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Crop to bounding box of mask
        y_indices, x_indices = np.where(mask)
        if len(y_indices) == 0 or len(x_indices) == 0:
            return {
                'label': 'unknown',
                'confidence': 0.0,
                'top_3': []
            }

        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()

        # Add padding (10% of bbox size)
        pad_x = int((x_max - x_min) * 0.1)
        pad_y = int((y_max - y_min) * 0.1)

        x_min = max(0, x_min - pad_x)
        x_max = min(image.shape[1], x_max + pad_x)
        y_min = max(0, y_min - pad_y)
        y_max = min(image.shape[0], y_max + pad_y)

        # Crop region
        cropped = image_rgb[y_min:y_max, x_min:x_max]

        # Ensure cropped region is not empty
        if cropped.size == 0:
            return {
                'label': 'unknown',
                'confidence': 0.0,
                'top_3': []
            }

        # Convert to PIL Image
        pil_image = Image.fromarray(cropped)

        # Preprocess for CLIP
        image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)

        # Encode image
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            # Calculate similarity with categories
            similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
            values, indices = similarity[0].topk(3)

        # Get top 3 predictions
        top_3 = [
            {
                'label': self.categories[idx],
                'confidence': val.item()
            }
            for val, idx in zip(values, indices)
        ]

        return {
            'label': top_3[0]['label'],
            'confidence': top_3[0]['confidence'],
            'top_3': top_3
        }

    def add_categories(self, new_categories):
        """
        Add new categories to the classifier

        Args:
            new_categories: List of new category names
        """
        # Add to existing categories
        self.categories.extend(new_categories)

        # Re-encode all text features
        text_inputs = torch.cat([
            clip.tokenize(f"a photo of a {c}") for c in self.categories
        ]).to(self.device)

        with torch.no_grad():
            self.text_features = self.model.encode_text(text_inputs)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

        print(f"Categories updated: {len(self.categories)} total")
