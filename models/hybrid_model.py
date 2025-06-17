# models/hybrid_model.py
"""
Hybrid model combining SAM and DINOv2 for virtual glasses try-on
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from image_processing.utils.image_utils import ImageProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HybridVirtualTryOnModel(nn.Module):
    """
    Hybrid model combining SAM (Segment Anything) and DINOv2 for virtual glasses try-on
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize the hybrid model
        
        Args:
            device: Device to run the model on ('cpu' or 'cuda')
        """
        super().__init__()
        
        self.device = device
        self.image_processor = ImageProcessor()
        
        # Model components
        self.sam_model = None
        self.sam_predictor = None
        self.dino_model = None
        
        # Model parameters
        self.input_size = (512, 512)
        self.face_embedding_dim = 768  # DINOv2 embedding dimension
        self.glasses_embedding_dim = 768
        
        # Face landmark detection
        self.face_cascade = None
        self.eye_cascade = None
        
        logger.info(f"HybridVirtualTryOnModel initialized on device: {device}")
    
    def load_sam_model(self, model_type: str = "vit_h", checkpoint_path: Optional[str] = None):
        """
        Load SAM (Segment Anything Model)
        
        Args:
            model_type: Type of SAM model ('vit_h', 'vit_l', 'vit_b')
            checkpoint_path: Path to SAM checkpoint
        """
        try:
            from segment_anything import sam_model_registry, SamPredictor
            
            if checkpoint_path is None:
                # Download default checkpoint
                logger.info("Downloading SAM checkpoint...")
                checkpoint_path = self._download_sam_checkpoint(model_type)
            
            # Load SAM model
            self.sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
            self.sam_model.to(self.device)
            
            # Initialize predictor
            self.sam_predictor = SamPredictor(self.sam_model)
            
            logger.info(f" SAM model loaded: {model_type}")
            
        except ImportError:
            logger.warning("SAM not available. Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")
            self._initialize_fallback_segmentation()
        except Exception as e:
            logger.error(f"Failed to load SAM model: {e}")
            self._initialize_fallback_segmentation()
    
    def load_dino_model(self, model_name: str = "dinov2_vitb14"):
        """
        Load DINOv2 model
        
        Args:
            model_name: DINOv2 model name
        """
        try:
            # Load DINOv2 from torch hub
            self.dino_model = torch.hub.load('facebookresearch/dinov2', model_name)
            self.dino_model.to(self.device)
            self.dino_model.eval()
            
            logger.info(f" DINOv2 model loaded: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to load DINOv2 model: {e}")
            self._initialize_fallback_feature_extraction()
    
    def _download_sam_checkpoint(self, model_type: str) -> str:
        """Download SAM checkpoint"""
        import requests
        
        checkpoint_urls = {
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth", 
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
        }
        
        if model_type not in checkpoint_urls:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create models directory
        models_dir = Path(__file__).parent / "sam"
        models_dir.mkdir(exist_ok=True)
        
        checkpoint_path = models_dir / f"sam_{model_type}.pth"
        
        if not checkpoint_path.exists():
            logger.info(f"Downloading SAM {model_type} checkpoint...")
            response = requests.get(checkpoint_urls[model_type])
            response.raise_for_status()
            
            with open(checkpoint_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Downloaded: {checkpoint_path}")
        
        return str(checkpoint_path)
    
    def _initialize_fallback_segmentation(self):
        """Initialize fallback segmentation using OpenCV"""
        logger.info("Using OpenCV fallback for segmentation")
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        except Exception as e:
            logger.warning(f"Failed to load OpenCV cascades: {e}")
    
    def _initialize_fallback_feature_extraction(self):
        """Initialize fallback feature extraction"""
        logger.info("Using simple feature extraction fallback")
        # Use a simple CNN-based feature extractor
        self.fallback_feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 768)  # Match DINOv2 embedding dimension
        ).to(self.device)
    
    def segment_face_region(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Segment face region using SAM or fallback method
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary with segmentation results
        """
        if self.sam_predictor is not None:
            return self._segment_with_sam(image)
        else:
            return self._segment_with_opencv(image)
    
    def _segment_with_sam(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Segment using SAM model"""
        try:
            # Set image for SAM predictor
            self.sam_predictor.set_image(image)
            
            # Detect face first to get region of interest
            face_detection = self._detect_face_opencv(image)
            
            if face_detection is None:
                return {'face_mask': np.zeros(image.shape[:2], dtype=np.uint8)}
            
            # Use face center as prompt point for SAM
            face_center = face_detection['face_center']
            input_point = np.array([[face_center[0], face_center[1]]])
            input_label = np.array([1])
            
            # Predict mask
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            
            # Choose best mask
            best_mask_idx = np.argmax(scores)
            face_mask = masks[best_mask_idx].astype(np.uint8) * 255
            
            return {
                'face_mask': face_mask,
                'face_region': face_detection,
                'confidence': scores[best_mask_idx]
            }
            
        except Exception as e:
            logger.error(f"SAM segmentation failed: {e}")
            return self._segment_with_opencv(image)
    
    def _segment_with_opencv(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Fallback segmentation using OpenCV"""
        try:
            face_detection = self._detect_face_opencv(image)
            
            if face_detection is None:
                return {'face_mask': np.zeros(image.shape[:2], dtype=np.uint8)}
            
            # Create simple face mask from detection
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            x, y, w, h = face_detection['face_rect']
            
            # Create elliptical mask for more natural face shape
            center = (x + w//2, y + h//2)
            axes = (w//2, h//2)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
            
            return {
                'face_mask': mask,
                'face_region': face_detection,
                'confidence': 0.8  # Default confidence for OpenCV
            }
            
        except Exception as e:
            logger.error(f"OpenCV segmentation failed: {e}")
            return {'face_mask': np.zeros(image.shape[:2], dtype=np.uint8)}
    
    def _detect_face_opencv(self, image: np.ndarray) -> Optional[Dict]:
        """Detect face using OpenCV"""
        if self.face_cascade is None:
            return None
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
        
        if len(faces) == 0:
            return None
        
        # Use largest face
        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = face
        
        # Detect eyes within face region
        face_roi = gray[y:y+h, x:x+w]
        eyes = []
        if self.eye_cascade is not None:
            eyes = self.eye_cascade.detectMultiScale(face_roi, 1.1, 5)
        
        # Estimate eye positions
        if len(eyes) >= 2:
            # Use detected eyes
            left_eye = eyes[0]
            right_eye = eyes[1]
            left_eye_center = (x + left_eye[0] + left_eye[2]//2, y + left_eye[1] + left_eye[3]//2)
            right_eye_center = (x + right_eye[0] + right_eye[2]//2, y + right_eye[1] + right_eye[3]//2)
        else:
            # Estimate eye positions
            eye_y = y + int(h * 0.35)
            left_eye_center = (x + int(w * 0.35), eye_y)
            right_eye_center = (x + int(w * 0.65), eye_y)
        
        return {
            'face_rect': (x, y, w, h),
            'face_center': (x + w//2, y + h//2),
            'left_eye': left_eye_center,
            'right_eye': right_eye_center,
            'eye_distance': abs(right_eye_center[0] - left_eye_center[0]),
            'face_width': w,
            'face_height': h
        }
    
    def extract_features(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> torch.Tensor:
        """
        Extract features using DINOv2 or fallback method
        
        Args:
            image: Input image
            mask: Optional mask to focus on specific region
            
        Returns:
            Feature tensor
        """
        if self.dino_model is not None:
            return self._extract_features_dino(image, mask)
        else:
            return self._extract_features_fallback(image, mask)
    
    def _extract_features_dino(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> torch.Tensor:
        """Extract features using DINOv2"""
        try:
            # Preprocess image for DINOv2
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply mask if provided
            if mask is not None:
                image_rgb = image_rgb * (mask[:, :, np.newaxis] / 255.0)
            
            # Resize and normalize
            image_resized = cv2.resize(image_rgb, (224, 224))  # DINOv2 input size
            image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
            
            # Normalize with ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_tensor = (image_tensor - mean) / std
            
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.dino_model(image_tensor)
            
            return features.squeeze(0)  # Remove batch dimension
            
        except Exception as e:
            logger.error(f"DINOv2 feature extraction failed: {e}")
            return self._extract_features_fallback(image, mask)
    
    def _extract_features_fallback(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> torch.Tensor:
        """Extract features using fallback method"""
        try:
            # Preprocess image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply mask if provided
            if mask is not None:
                image_rgb = image_rgb * (mask[:, :, np.newaxis] / 255.0)
            
            # Resize and normalize
            image_resized = cv2.resize(image_rgb, (224, 224))
            image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            # Extract features using fallback network
            with torch.no_grad():
                features = self.fallback_feature_extractor(image_tensor)
            
            return features.squeeze(0)
            
        except Exception as e:
            logger.error(f"Fallback feature extraction failed: {e}")
            return torch.zeros(768).to(self.device)
    
    def align_glasses(self, glasses_image: np.ndarray, face_info: Dict,
                     target_size: Tuple[int, int]) -> np.ndarray:
        """
        Align glasses to face using extracted features and face information
        
        Args:
            glasses_image: Glasses image with alpha channel
            face_info: Face detection information
            target_size: Target size for alignment
            
        Returns:
            Aligned glasses image
        """
        try:
            # Calculate glasses positioning
            eye_distance = face_info['eye_distance']
            glasses_width = int(eye_distance * 1.2)  # Slightly wider than eye distance
            
            # Calculate aspect ratio and height
            current_height, current_width = glasses_image.shape[:2]
            aspect_ratio = current_height / current_width
            glasses_height = int(glasses_width * aspect_ratio)
            
            # Resize glasses
            glasses_resized = cv2.resize(glasses_image, (glasses_width, glasses_height))
            
            # Calculate position
            left_eye = face_info['left_eye']
            right_eye = face_info['right_eye']
            glasses_center_x = (left_eye[0] + right_eye[0]) // 2
            glasses_center_y = (left_eye[1] + right_eye[1]) // 2 - 10  # Slightly above eyes
            
            # Create canvas
            canvas = np.zeros((*target_size, 4), dtype=np.uint8)
            
            # Calculate placement position
            start_x = max(0, glasses_center_x - glasses_width // 2)
            start_y = max(0, glasses_center_y - glasses_height // 2)
            end_x = min(target_size[1], start_x + glasses_width)
            end_y = min(target_size[0], start_y + glasses_height)
            
            # Adjust glasses size if it doesn't fit
            actual_width = end_x - start_x
            actual_height = end_y - start_y
            
            if actual_width < glasses_width or actual_height < glasses_height:
                glasses_resized = cv2.resize(glasses_resized, (actual_width, actual_height))
            
            # Place glasses on canvas
            canvas[start_y:end_y, start_x:end_x] = glasses_resized[:actual_height, :actual_width]
            
            return canvas
            
        except Exception as e:
            logger.error(f"Glasses alignment failed: {e}")
            return glasses_image
    
    def forward(self, selfie_image: np.ndarray, glasses_image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Forward pass of the hybrid model
        
        Args:
            selfie_image: Selfie image as numpy array
            glasses_image: Glasses image as numpy array
            
        Returns:
            Dictionary with results including final composite image
        """
        try:
            # Step 1: Segment face region
            segmentation_result = self.segment_face_region(selfie_image)
            face_mask = segmentation_result['face_mask']
            face_info = segmentation_result.get('face_region')
            
            if face_info is None:
                logger.warning("No face detected, using original image")
                return {'result': selfie_image, 'face_detected': False}
            
            # Step 2: Extract features from face and glasses
            face_features = self.extract_features(selfie_image, face_mask)
            glasses_features = self.extract_features(glasses_image)
            
            # Step 3: Align glasses to face
            aligned_glasses = self.align_glasses(
                glasses_image, 
                face_info, 
                selfie_image.shape[:2]
            )
            
            # Step 4: Composite final image
            result_image = self._composite_images(selfie_image, aligned_glasses)
            
            return {
                'result': result_image,
                'face_mask': face_mask,
                'face_features': face_features,
                'glasses_features': glasses_features,
                'aligned_glasses': aligned_glasses,
                'face_detected': True,
                'confidence': segmentation_result.get('confidence', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            return {'result': selfie_image, 'error': str(e)}
    
    def _composite_images(self, selfie: np.ndarray, glasses: np.ndarray) -> np.ndarray:
        """
        Composite selfie and glasses images
        
        Args:
            selfie: Selfie image
            glasses: Glasses image with alpha channel
            
        Returns:
            Composited image
        """
        try:
            result = selfie.copy()
            
            # Ensure glasses has alpha channel
            if glasses.shape[2] == 3:
                alpha = np.ones((glasses.shape[0], glasses.shape[1], 1), dtype=glasses.dtype) * 255
                glasses = np.concatenate([glasses, alpha], axis=2)
            
            # Extract alpha channel
            alpha = glasses[:, :, 3] / 255.0
            glasses_rgb = glasses[:, :, :3]
            
            # Apply alpha blending
            for c in range(3):
                result[:, :, c] = (1 - alpha) * result[:, :, c] + alpha * glasses_rgb[:, :, c]
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Image compositing failed: {e}")
            return selfie


# Convenience functions
def create_hybrid_model(device: str = 'cpu', load_models: bool = True) -> HybridVirtualTryOnModel:
    """
    Create and initialize hybrid model
    
    Args:
        device: Device to run on
        load_models: Whether to load SAM and DINOv2 models
        
    Returns:
        Initialized hybrid model
    """
    model = HybridVirtualTryOnModel(device=device)
    
    if load_models:
        try:
            model.load_sam_model("vit_b")  # Use smaller model for faster loading
            model.load_dino_model("dinov2_vitb14")
        except Exception as e:
            logger.warning(f"Failed to load some models: {e}")
    
    return model

# Example usage and testing
if __name__ == "__main__":
    print("Testing HybridVirtualTryOnModel...")
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = create_hybrid_model(device=device, load_models=False)  # Test without loading heavy models
    
    # Create dummy images for testing
    dummy_selfie = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    dummy_glasses = np.random.randint(0, 255, (200, 400, 4), dtype=np.uint8)
    
    print(f"Dummy selfie shape: {dummy_selfie.shape}")
    print(f"Dummy glasses shape: {dummy_glasses.shape}")
    
    # Test forward pass
    results = model.forward(dummy_selfie, dummy_glasses)
    
    print(f"Results keys: {list(results.keys())}")
    print(f"Face detected: {results.get('face_detected', False)}")
    print(f"Result image shape: {results['result'].shape}")
    
    print(" HybridVirtualTryOnModel test completed")