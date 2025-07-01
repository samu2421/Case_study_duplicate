"""
Image processing utilities for the Virtual Glasses Try-On system.
Handles resizing, augmentation, cropping, and other image operations.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from pathlib import Path
from typing import Tuple, Optional, List, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Handles all image processing operations for selfies and glasses."""
    
    def __init__(self, target_size: Tuple[int, int] = (512, 512)):
        """Initialize image processor with target size."""
        self.target_size = target_size
        self.glasses_transforms = self._get_glasses_transforms()
        self.selfie_transforms = self._get_selfie_transforms()
        
    def _get_selfie_transforms(self) -> A.Compose:
        """Get augmentation transforms for selfie images."""
        return A.Compose([
            A.Resize(height=self.target_size[0], width=self.target_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    def _get_glasses_transforms(self) -> A.Compose:
        """Get transforms for glasses images (minimal augmentation to preserve transparency)."""
        return A.Compose([
            A.Resize(height=self.target_size[0], width=self.target_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406, 0.0], std=[0.229, 0.224, 0.225, 1.0]),  # Include alpha channel
            ToTensorV2()
        ])
    
    def load_image(self, image_path: Union[str, Path, bytes]) -> Optional[np.ndarray]:
        """Load image from file path or bytes."""
        try:
            if isinstance(image_path, bytes):
                # Load from bytes (from database)
                nparr = np.frombuffer(image_path, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                # Load from file path
                image_path = Path(image_path)
                if not image_path.exists():
                    logger.error(f"Image file not found: {image_path}")
                    return None
                
                image = cv2.imread(str(image_path))
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return image
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            return None
    
    def load_glasses_image(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """Load glasses image with alpha channel preservation."""
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                logger.error(f"Glasses image not found: {image_path}")
                return None
            
            # Use PIL to handle transparency better
            pil_image = Image.open(image_path).convert("RGBA")
            image = np.array(pil_image)
            
            return image
        except Exception as e:
            logger.error(f"Failed to load glasses image: {e}")
            return None
    
    def resize_image(self, image: np.ndarray, size: Tuple[int, int] = None) -> np.ndarray:
        """Resize image maintaining aspect ratio."""
        if size is None:
            size = self.target_size
        
        h, w = image.shape[:2]
        target_h, target_w = size
        
        # Calculate scaling factor to maintain aspect ratio
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Create canvas with target size
        if len(image.shape) == 3:
            canvas = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
        else:
            canvas = np.zeros((target_h, target_w), dtype=image.dtype)
        
        # Center the resized image
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        
        return canvas
    
    def remove_glasses_background(self, glasses_image: np.ndarray) -> np.ndarray:
        """Remove background from glasses image and ensure proper transparency."""
        try:
            # Convert to RGBA if not already
            if glasses_image.shape[2] == 3:
                glasses_image = cv2.cvtColor(glasses_image, cv2.COLOR_RGB2RGBA)
            
            # Create mask for background removal
            # Assuming white/light backgrounds are common
            gray = cv2.cvtColor(glasses_image[:, :, :3], cv2.COLOR_RGB2GRAY)
            
            # Threshold to identify background (white/light areas)
            _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Apply mask to alpha channel
            glasses_image[:, :, 3] = mask
            
            return glasses_image
        except Exception as e:
            logger.error(f"Failed to remove glasses background: {e}")
            return glasses_image
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image values to [0, 1] range."""
        return image.astype(np.float32) / 255.0
    
    def denormalize_image(self, image: np.ndarray) -> np.ndarray:
        """Convert normalized image back to [0, 255] range."""
        return (image * 255).astype(np.uint8)
    
    def detect_face_region(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect face region using OpenCV's Haar cascades."""
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            
            # Load face cascade classifier
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) > 0:
                # Return the largest face
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                return tuple(largest_face)
            
            return None
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return None
    
    def crop_face_region(self, image: np.ndarray, face_bbox: Tuple[int, int, int, int], 
                        expand_ratio: float = 0.3) -> np.ndarray:
        """Crop face region with some padding."""
        x, y, w, h = face_bbox
        
        # Expand the bounding box
        expand_w = int(w * expand_ratio)
        expand_h = int(h * expand_ratio)
        
        x1 = max(0, x - expand_w)
        y1 = max(0, y - expand_h)
        x2 = min(image.shape[1], x + w + expand_w)
        y2 = min(image.shape[0], y + h + expand_h)
        
        return image[y1:y2, x1:x2]
    
    def enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """Enhance image quality using PIL."""
        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(image)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(pil_image)
            pil_image = enhancer.enhance(1.2)
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(1.1)
            
            # Convert back to numpy
            return np.array(pil_image)
        except Exception as e:
            logger.error(f"Image enhancement failed: {e}")
            return image
    
    def calculate_image_quality_score(self, image: np.ndarray) -> float:
        """Calculate image quality score based on various metrics."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            
            # Calculate sharpness using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalize to 0-1 range (empirically determined threshold)
            quality_score = min(laplacian_var / 1000.0, 1.0)
            
            return quality_score
        except Exception as e:
            logger.error(f"Quality score calculation failed: {e}")
            return 0.0
    
    def save_image(self, image: np.ndarray, output_path: Union[str, Path]) -> bool:
        """Save image to file."""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert RGB to BGR for OpenCV
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(output_path), image_bgr)
            elif len(image.shape) == 3 and image.shape[2] == 4:
                # Save with alpha channel using PIL
                pil_image = Image.fromarray(image, 'RGBA')
                pil_image.save(output_path)
            else:
                cv2.imwrite(str(output_path), image)
            
            return True
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            return False
    
    def image_to_bytes(self, image: np.ndarray, format: str = 'PNG') -> bytes:
        """Convert image to bytes for database storage."""
        try:
            pil_image = Image.fromarray(image)
            import io
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format=format)
            return img_byte_arr.getvalue()
        except Exception as e:
            logger.error(f"Failed to convert image to bytes: {e}")
            return b''

# Global image processor instance
image_processor = ImageProcessor()