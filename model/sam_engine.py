import cv2
import numpy as np
import torch
import os
from PIL import Image
import sys

class SamPredictorWrapper:
    """
    Wrapper for Meta's Segment Anything Model 2 (SAM 2) with CLIP classification
    Handles instant segmentation and object identification
    """

    def __init__(self, model_type="hiera_base_plus", checkpoint_path=None):
        """
        Initialize SAM 2 model and CLIP classifier

        Args:
            model_type: Type of SAM 2 model ('hiera_tiny', 'hiera_small', 'hiera_base_plus', 'hiera_large')
            checkpoint_path: Path to model checkpoint file
        """
        self.model_type = model_type  # Store model type for display
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Set default checkpoint path for SAM 2
        if checkpoint_path is None:
            # Map model types to their SAM 2 checkpoint filenames
            model_checkpoints = {
                'hiera_tiny': 'sam2_hiera_tiny.pt',
                'hiera_small': 'sam2_hiera_small.pt',
                'hiera_base_plus': 'sam2_hiera_base_plus.pt',
                'hiera_large': 'sam2_hiera_large.pt'
            }

            checkpoint_filename = model_checkpoints.get(model_type, 'sam2_hiera_base_plus.pt')
            checkpoint_path = os.path.join(
                os.path.dirname(__file__),
                'weights',
                checkpoint_filename
            )

        # Check if SAM 2 library is available
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            # Load SAM 2 model
            if os.path.exists(checkpoint_path):
                print(f"Loading SAM 2 model from {checkpoint_path}")

                # Map model types to config file paths
                # Config files are in: venv/Lib/site-packages/sam2/configs/sam2.1/
                config_map = {
                    'hiera_tiny': 'sam2.1_hiera_t.yaml',
                    'hiera_small': 'sam2.1_hiera_s.yaml',
                    'hiera_base_plus': 'sam2.1_hiera_b+.yaml',
                    'hiera_large': 'sam2.1_hiera_l.yaml'
                }

                config_file = config_map.get(model_type, 'sam2.1_hiera_b+.yaml')

                # Get config path relative to sam2 package
                import sam2
                sam2_dir = os.path.dirname(sam2.__file__)
                config_path = os.path.join(sam2_dir, 'configs', 'sam2.1', config_file)

                # Load checkpoint and build model
                sam2_model = build_sam2(
                    config_path,
                    checkpoint_path,
                    device=self.device
                )

                self.predictor = SAM2ImagePredictor(sam2_model)
                self.sam_available = True
                print("✓ SAM 2 model loaded successfully (6x faster than SAM 1.0)")
            else:
                print(f"Warning: SAM 2 checkpoint not found at {checkpoint_path}")
                print("Run: python download_sam2_weights.py")
                self.sam_available = False

        except Exception as e:
            print(f"Could not load SAM 2: {e}")
            import traceback
            traceback.print_exc()
            print("The application will run in demo mode")
            self.sam_available = False

        # Initialize CLIP classifier for object identification
        try:
            from model.clip_classifier import CLIPClassifier
            print("Loading CLIP classifier...")
            self.clip_classifier = CLIPClassifier()
            self.clip_available = True
        except Exception as e:
            print(f"Could not load CLIP classifier: {e}")
            self.clip_available = False

    def _get_config_file(self, model_type):
        """
        Get SAM 2 config file path for model type

        Args:
            model_type: One of 'hiera_tiny', 'hiera_small', 'hiera_base_plus', 'hiera_large'

        Returns:
            str: Config file name
        """
        config_files = {
            'hiera_tiny': 'sam2.1_hiera_t.yaml',
            'hiera_small': 'sam2.1_hiera_s.yaml',
            'hiera_base_plus': 'sam2.1_hiera_b+.yaml',
            'hiera_large': 'sam2.1_hiera_l.yaml'
        }
        return config_files.get(model_type, 'sam2.1_hiera_b+.yaml')

    def get_model_display_name(self):
        """
        Get human-readable model name for UI display

        Returns:
            str: Formatted model name for display
        """
        model_names = {
            'hiera_tiny': 'SAM 2.1 Hiera Tiny',
            'hiera_small': 'SAM 2.1 Hiera Small',
            'hiera_base_plus': 'SAM 2.1 Hiera Base+',
            'hiera_large': 'SAM 2.1 Hiera Large'
        }
        return model_names.get(self.model_type, 'SAM 2.1')

    def predict_single_point_instant(self, image_path, point):
        """
        Instant segmentation + classification for single click (Select Object mode)

        Args:
            image_path: Path to image file
            point: Dictionary with 'x', 'y', 'label' keys (label=1 for foreground)

        Returns:
            {
                'mask': np.array (bool),
                'object_label': str,
                'confidence': float,
                'bbox': [x1, y1, x2, y2],
                'top_3': list of {label, confidence} dicts
            }
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Demo mode if SAM not available
        if not self.sam_available:
            return self._demo_instant_segment(image, point)

        # SAM 2: Set image
        self.predictor.set_image(image_rgb)

        # SAM 2: Predict with single point
        point_coords = np.array([[point['x'], point['y']]])
        point_labels = np.array([point.get('label', 1)])  # Default to foreground

        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )

        if len(masks) == 0 or len(scores) == 0:
            return {
                'mask': np.zeros((image.shape[0], image.shape[1]), dtype=bool),
                'object_label': 'unknown',
                'confidence': 0.0,
                'bbox': [0, 0, 0, 0],
                'top_3': []
            }

        # Select best mask (highest score)
        best_idx = np.argmax(scores)
        mask = masks[best_idx]
        mask_confidence = float(scores[best_idx])

        # CLIP: Classify the segmented region
        if self.clip_available:
            classification = self.clip_classifier.classify_region(image, mask)
            object_label = classification['label']
            top_3 = classification['top_3']
        else:
            object_label = 'object'
            top_3 = [{'label': 'object', 'confidence': mask_confidence}]

        # Compute bounding box
        y_indices, x_indices = np.where(mask)
        if len(y_indices) == 0:
            bbox = [0, 0, 0, 0]
        else:
            bbox = [
                int(x_indices.min()),
                int(y_indices.min()),
                int(x_indices.max()),
                int(y_indices.max())
            ]

        return {
            'mask': mask,
            'object_label': object_label,
            'confidence': mask_confidence,
            'bbox': bbox,
            'top_3': top_3
        }

    def _demo_instant_segment(self, image, point):
        """
        Demo mode for instant segmentation when SAM 2 not available
        """
        height, width = image.shape[:2]
        x, y = point['x'], point['y']

        # Create circular mask around clicked point
        mask = np.zeros((height, width), dtype=bool)
        radius = min(width, height) // 10

        # Draw filled circle
        y_grid, x_grid = np.ogrid[:height, :width]
        circle_mask = (x_grid - x)**2 + (y_grid - y)**2 <= radius**2
        mask[circle_mask] = True

        # Compute bbox
        y_indices, x_indices = np.where(mask)
        bbox = [
            int(x_indices.min()),
            int(y_indices.min()),
            int(x_indices.max()),
            int(y_indices.max())
        ]

        return {
            'mask': mask,
            'object_label': 'demo_object',
            'confidence': 0.85,
            'bbox': bbox,
            'top_3': [
                {'label': 'demo_object', 'confidence': 0.85},
                {'label': 'unknown', 'confidence': 0.10},
                {'label': 'item', 'confidence': 0.05}
            ]
        }

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

    def predict_with_points(self, image_path, output_path, points, confidence_threshold=0.3):
        """
        Perform segmentation using point prompts (click mode)

        Args:
            image_path: Path to input image
            output_path: Path to save result image
            points: List of dictionaries with 'x', 'y', 'label' keys
                   label=1 for foreground, label=0 for background
            confidence_threshold: Minimum confidence threshold for masks

        Returns:
            int: Number of detected objects
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # If SAM is not available, use demo mode
        if not self.sam_available:
            return self._demo_click_mode(image, output_path, points)

        # Set image for prediction
        self.predictor.set_image(image_rgb)

        # Convert points to numpy arrays
        point_coords = np.array([[p['x'], p['y']] for p in points])
        point_labels = np.array([p['label'] for p in points])

        # Predict masks
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )

        # Filter masks by confidence threshold
        valid_masks = [(mask, score) for mask, score in zip(masks, scores) if score >= confidence_threshold]

        if not valid_masks:
            # No valid masks, save original image
            cv2.imwrite(output_path, image)
            return 0

        # Use the best mask (highest score)
        best_mask, best_score = max(valid_masks, key=lambda x: x[1])

        # Visualize
        overlay = self._visualize_masks([best_mask], image, scores=[best_score])

        # Save result
        cv2.imwrite(output_path, overlay)

        return len(points)

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

    def _visualize_masks(self, masks, image, scores=None):
        """
        Visualize masks on image with colored overlay and contours

        Args:
            masks: List of binary masks
            image: Original BGR image
            scores: Optional list of confidence scores

        Returns:
            Image with mask overlay
        """
        overlay = image.copy()

        for i, mask in enumerate(masks):
            # Get color for this mask
            color = self._get_color(i)

            # Create colored overlay
            colored_mask = np.zeros_like(image)
            colored_mask[mask] = color

            # Blend with image
            overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.4, 0)

            # Draw contour
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)

            # Add confidence score if available
            if scores and i < len(scores):
                # Find centroid for text placement
                M = cv2.moments(mask_uint8)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    score_text = f"{scores[i]:.2f}"
                    cv2.putText(overlay, score_text, (cx-20, cy),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return overlay

    def _demo_click_mode(self, image, output_path, points):
        """
        Demo mode for click-based detection when SAM is not available
        """
        overlay = image.copy()
        height, width = image.shape[:2]

        # Draw circular masks around clicked points
        for i, point in enumerate(points):
            x, y = point['x'], point['y']
            radius = min(width, height) // 10  # Adaptive radius

            # Create circular mask
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(mask, (x, y), radius, 255, -1)

            # Create colored overlay
            color = self._get_color(i)
            colored_mask = np.zeros_like(image)
            colored_mask[mask == 255] = color

            # Blend
            overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.4, 0)

            # Draw contour
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)

            # Draw click point
            cv2.circle(overlay, (x, y), 5, (0, 0, 255), -1)
            cv2.circle(overlay, (x, y), 5, (255, 255, 255), 2)

        # Add demo mode text
        cv2.putText(overlay, "Demo Mode - Click Detection", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 52, 115), 2)

        cv2.imwrite(output_path, overlay)
        return len(points)

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
