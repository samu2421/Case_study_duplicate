"""
Image processing utilities for the Virtual Glasses Try-On project
"""
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from pathlib import Path
from typing import Tuple, Optional, Union, List
import logging
import requests
from io import BytesIO

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageProcessor:
    """Utility class for common image processing operations"""
    
    # Standard image sizes for the project
    STANDARD_SIZES = {
        'thumbnail': (150, 150),
        'small': (300, 300),
        'medium': (512, 512),
        'large': (1024, 1024)
    }
    
    # Supported image formats
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    @staticmethod
    def load_image(image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Load an image from file path
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Image as numpy array in BGR format, or None if failed
        """
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                logger.error(f"Image file not found: {image_path}")
                return None
            
            # Load image using OpenCV
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            
            logger.debug(f"Loaded image: {image_path}, shape: {image.shape}")
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    @staticmethod
    def load_image_from_url(url: str) -> Optional[np.ndarray]:
        """
        Load an image from URL
        
        Args:
            url: URL of the image
            
        Returns:
            Image as numpy array in BGR format, or None if failed
        """
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Convert to PIL Image first
            pil_image = Image.open(BytesIO(response.content))
            
            # Convert to OpenCV format
            image_array = np.array(pil_image)
            
            # Convert RGB to BGR if needed
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            logger.debug(f"Loaded image from URL: {url}, shape: {image_array.shape}")
            return image_array
            
        except Exception as e:
            logger.error(f"Error loading image from URL {url}: {e}")
            return None
    
    @staticmethod
    def save_image(image: np.ndarray, output_path: Union[str, Path], 
                   quality: int = 95) -> bool:
        """
        Save an image to file
        
        Args:
            image: Image as numpy array
            output_path: Path to save the image
            quality: JPEG quality (0-100)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Set encoding parameters based on file extension
            ext = output_path.suffix.lower()
            if ext in ['.jpg', '.jpeg']:
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            elif ext == '.png':
                encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
            else:
                encode_params = []
            
            success = cv2.imwrite(str(output_path), image, encode_params)
            
            if success:
                logger.debug(f"Saved image: {output_path}")
                return True
            else:
                logger.error(f"Failed to save image: {output_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error saving image to {output_path}: {e}")
            return False
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                     maintain_aspect_ratio: bool = True) -> np.ndarray:
        """
        Resize an image to target size
        
        Args:
            image: Input image as numpy array
            target_size: Target size as (width, height)
            maintain_aspect_ratio: Whether to maintain aspect ratio
            
        Returns:
            Resized image
        """
        height, width = image.shape[:2]
        target_width, target_height = target_size
        
        if maintain_aspect_ratio:
            # Calculate the scaling factor to maintain aspect ratio
            scale = min(target_width / width, target_height / height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize the image
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Create a blank canvas with target size
            if len(image.shape) == 3:
                canvas = np.zeros((target_height, target_width, image.shape[2]), dtype=image.dtype)
            else:
                canvas = np.zeros((target_height, target_width), dtype=image.dtype)
            
            # Center the resized image on the canvas
            y_offset = (target_height - new_height) // 2
            x_offset = (target_width - new_width) // 2
            canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
            
            return canvas
        else:
            # Direct resize without maintaining aspect ratio
            return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """
        Normalize image to 0-1 range
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        return image.astype(np.float32) / 255.0
    
    @staticmethod
    def denormalize_image(image: np.ndarray) -> np.ndarray:
        """
        Denormalize image from 0-1 range to 0-255
        
        Args:
            image: Normalized image
            
        Returns:
            Denormalized image
        """
        return (image * 255.0).astype(np.uint8)
    
    @staticmethod
    def enhance_image(image: np.ndarray, brightness: float = 1.0, 
                     contrast: float = 1.0, saturation: float = 1.0) -> np.ndarray:
        """
        Enhance image brightness, contrast, and saturation
        
        Args:
            image: Input image in BGR format
            brightness: Brightness factor (1.0 = no change)
            contrast: Contrast factor (1.0 = no change)
            saturation: Saturation factor (1.0 = no change)
            
        Returns:
            Enhanced image
        """
        # Convert BGR to RGB for PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Apply enhancements
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(brightness)
        
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(contrast)
        
        if saturation != 1.0:
            enhancer = ImageEnhance.Color(pil_image)
            pil_image = enhancer.enhance(saturation)
        
        # Convert back to BGR
        enhanced_rgb = np.array(pil_image)
        enhanced_bgr = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)
        
        return enhanced_bgr
    
    @staticmethod
    def create_alpha_channel(image: np.ndarray, background_color: Tuple[int, int, int] = (255, 255, 255),
                           tolerance: int = 30) -> np.ndarray:
        """
        Create alpha channel by removing background color
        
        Args:
            image: Input image in BGR format
            background_color: Background color to remove (B, G, R)
            tolerance: Color tolerance for background removal
            
        Returns:
            Image with alpha channel (BGRA)
        """
        # Create alpha channel
        alpha = np.ones(image.shape[:2], dtype=np.uint8) * 255
        
        # Calculate distance from background color
        diff = np.abs(image.astype(np.float32) - np.array(background_color))
        distance = np.sqrt(np.sum(diff ** 2, axis=2))
        
        # Set alpha to 0 where distance is within tolerance
        alpha[distance <= tolerance] = 0
        
        # Combine image with alpha channel
        bgra_image = np.dstack([image, alpha])
        
        return bgra_image
    
    @staticmethod
    def detect_faces(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in an image using OpenCV's Haar cascades
        
        Args:
            image: Input image in BGR format
            
        Returns:
            List of face bounding boxes as (x, y, width, height)
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Load face cascade classifier
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            return faces.tolist() if len(faces) > 0 else []
            
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    @staticmethod
    def get_image_info(image_path: Union[str, Path]) -> Optional[dict]:
        """
        Get basic information about an image file
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with image information or None if failed
        """
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                return None
            
            # Get file size
            file_size = image_path.stat().st_size
            
            # Load image to get dimensions
            image = cv2.imread(str(image_path))
            if image is None:
                return None
            
            height, width, channels = image.shape
            
            return {
                'filename': image_path.name,
                'file_path': str(image_path),
                'file_size_bytes': file_size,
                'file_size_kb': file_size // 1024,
                'width': width,
                'height': height,
                'channels': channels,
                'format': image_path.suffix.lower()
            }
            
        except Exception as e:
            logger.error(f"Error getting image info for {image_path}: {e}")
            return None
    
    @staticmethod
    def validate_image(image_path: Union[str, Path]) -> bool:
        """
        Validate if a file is a valid image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if valid image, False otherwise
        """
        try:
            image_path = Path(image_path)
            
            # Check file extension
            if image_path.suffix.lower() not in ImageProcessor.SUPPORTED_FORMATS:
                return False
            
            # Try to load the image
            image = cv2.imread(str(image_path))
            return image is not None
            
        except Exception:
            return False


# Convenience functions for common operations
def resize_to_standard(image: np.ndarray, size_name: str = 'medium') -> np.ndarray:
    """Resize image to standard project size"""
    if size_name not in ImageProcessor.STANDARD_SIZES:
        raise ValueError(f"Unknown size: {size_name}. Available: {list(ImageProcessor.STANDARD_SIZES.keys())}")
    
    target_size = ImageProcessor.STANDARD_SIZES[size_name]
    return ImageProcessor.resize_image(image, target_size)

def load_and_preprocess_image(image_path: Union[str, Path], 
                            target_size: str = 'medium',
                            normalize: bool = False) -> Optional[np.ndarray]:
    """Load and preprocess image in one step"""
    image = ImageProcessor.load_image(image_path)
    if image is None:
        return None
    
    # Resize to standard size
    image = resize_to_standard(image, target_size)
    
    # Normalize if requested
    if normalize:
        image = ImageProcessor.normalize_image(image)
    
    return image

# Example usage and testing
if __name__ == "__main__":
    print("Testing ImageProcessor...")
    
    # Test creating a dummy image
    dummy_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    print(f"Created dummy image with shape: {dummy_image.shape}")
    
    # Test resizing
    resized = ImageProcessor.resize_image(dummy_image, (512, 512))
    print(f"Resized image shape: {resized.shape}")
    
    # Test standard sizes
    for size_name, size in ImageProcessor.STANDARD_SIZES.items():
        resized_std = resize_to_standard(dummy_image, size_name)
        print(f"{size_name} size: {resized_std.shape}")
    
    # Test normalization
    normalized = ImageProcessor.normalize_image(dummy_image)
    print(f"Normalized image range: {normalized.min():.3f} - {normalized.max():.3f}")
    
    # Test denormalization
    denormalized = ImageProcessor.denormalize_image(normalized)
    print(f"Denormalized image range: {denormalized.min()} - {denormalized.max()}")
    
    print("✅ ImageProcessor test completed")