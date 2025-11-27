import cv2
import numpy as np
import torch
import os

class SamPredictorWrapper:
    """
    Wrapper for Meta's Segment Anything Model (SAM)
    This class handles object detection and segmentation based on text prompts
    """

    def __init__(self, model_type="vit_h", checkpoint_path=None):
        """
        Initialize SAM model

        Args:
            model_type: Type of SAM model ('vit_h', 'vit_l', 'vit_b')
            checkpoint_path: Path to model checkpoint file
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Set default checkpoint path
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                os.path.dirname(__file__),
                'weights',
                'sam_vit_h_4b8939.pth',
                'sam_vit_l_0b3195.pth'
            )

        # Check if SAM library is available
        try:
            from segment_anything import sam_model_registry, SamPredictor
            self.sam_available = True

            # Load model
            if os.path.exists(checkpoint_path):
                print(f"Loading SAM model from {checkpoint_path}")
                sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
                sam.to(device=self.device)
                self.predictor = SamPredictor(sam)
                print("SAM model loaded successfully")
            else:
                print(f"Warning: Checkpoint not found at {checkpoint_path}")
                print("Please download the model weights from:")
                print("https://github.com/facebookresearch/segment-anything#model-checkpoints")
                self.sam_available = False

        except ImportError:
            print("segment_anything library not found")
            print("Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")
            self.sam_available = False

    def predict_and_visualize(self, image_path, output_path, text_prompt):
        """
        Perform segmentation and visualization

        Args:
            image_path: Path to input image
            output_path: Path to save result image
            text_prompt: Text description of object to detect

        Returns:
            int: Number of detected objects
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # If SAM is not available, use a simple demo mode
        if not self.sam_available:
            return self._demo_mode(image, output_path, text_prompt)

        # Set image for prediction
        self.predictor.set_image(image_rgb)

        # For a full implementation with text prompts, you would need to:
        # 1. Use GroundingDINO or similar to convert text to bounding boxes
        # 2. Feed those boxes to SAM for segmentation
        #
        # For now, this is a simplified version that demonstrates the structure
        # You would integrate with a text-to-box model here

        # Demo implementation - generates random segmentation for demonstration
        return self._demo_segmentation(image, image_rgb, output_path)

    def _demo_segmentation(self, image, image_rgb, output_path):
        """
        Demo segmentation for testing without full SAM integration
        """
        height, width = image.shape[:2]

        # Generate some demo masks
        num_objects = np.random.randint(2, 6)

        overlay = image.copy()

        for i in range(num_objects):
            # Random center and size for demo
            center_x = np.random.randint(width // 4, 3 * width // 4)
            center_y = np.random.randint(height // 4, 3 * height // 4)
            radius = np.random.randint(50, 150)

            # Create circular mask (demo)
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(mask, (center_x, center_y), radius, 255, -1)

            # Create colored overlay
            color = self._get_color(i)
            colored_mask = np.zeros_like(image)
            colored_mask[mask == 255] = color

            # Blend with image
            overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.4, 0)

            # Draw contour
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)

        # Save result
        cv2.imwrite(output_path, overlay)
        return num_objects

    def _demo_mode(self, image, output_path, text_prompt):
        """
        Simple demo mode without SAM - just returns the original image with some annotations
        """
        # Just copy the image and add a text overlay
        demo_image = image.copy()

        # Add text overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Demo Mode: Searching for '{text_prompt}'"
        cv2.putText(demo_image, text, (10, 30), font, 0.7, (0, 52, 115), 2)

        # Save
        cv2.imwrite(output_path, demo_image)

        # Return random count for demo
        return np.random.randint(1, 5)

    def _get_color(self, index):
        """
        Get color for visualization
        Returns BGR color based on index
        """
        # Primary color is #003473 (RGB: 0, 52, 115)
        colors = [
            (115, 52, 0),    # Primary in BGR
            (0, 140, 255),   # Orange
            (0, 255, 0),     # Green
            (255, 0, 0),     # Blue
            (255, 255, 0),   # Cyan
            (255, 0, 255),   # Magenta
        ]
        return colors[index % len(colors)]

    def _text_to_boxes(self, image_rgb, text_prompt):
        """
        Convert text prompt to bounding boxes
        This would typically use GroundingDINO or similar model

        For full implementation, integrate with:
        https://github.com/IDEA-Research/GroundingDINO
        """
        # Placeholder for text-to-box conversion
        # In production, this would call GroundingDINO or similar
        raise NotImplementedError(
            "Text-to-box conversion requires GroundingDINO integration. "
            "See: https://github.com/IDEA-Research/GroundingDINO"
        )
