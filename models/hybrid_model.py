"""
Hybrid model combining Meta's SAM and DINOv2 for Virtual Glasses Try-On.
Uses SAM for precise facial segmentation and DINOv2 for feature extraction and alignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path

# Import Meta AI models
from segment_anything import sam_model_registry, SamPredictor
import timm

logger = logging.getLogger(__name__)

class SAMProcessor:
    """Handles Segment Anything Model operations for face segmentation."""
    
    def __init__(self, model_type: str = "vit_h", device: str = "auto"):
        """Initialize SAM processor."""
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.sam_model = None
        self.predictor = None
        self._load_model()
    
    def _load_model(self):
        """Load the SAM model."""
        try:
            # Download checkpoint if not exists
            checkpoint_paths = {
                "vit_h": "sam_vit_h_4b8939.pth",
                "vit_l": "sam_vit_l_0b3195.pth",
                "vit_b": "sam_vit_b_01ec64.pth"
            }
            
            checkpoint_path = checkpoint_paths[self.model_type]
            
            # Check if checkpoint exists locally
            if not Path(checkpoint_path).exists():
                logger.info(f"Downloading SAM checkpoint: {checkpoint_path}")
                import urllib.request
                urls = {
                    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
                }
                urllib.request.urlretrieve(urls[self.model_type], checkpoint_path)
            
            # Load model
            self.sam_model = sam_model_registry[self.model_type](checkpoint=checkpoint_path)
            self.sam_model.to(self.device)
            self.predictor = SamPredictor(self.sam_model)
            
            logger.info(f"SAM model {self.model_type} loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load SAM model: {e}")
            raise
    
    def segment_face(self, image: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, np.ndarray]:
        """Segment face regions using SAM."""
        try:
            # Set image for SAM
            self.predictor.set_image(image)
            
            if face_bbox is None:
                # Use center point as prompt if no bbox provided
                h, w = image.shape[:2]
                input_point = np.array([[w//2, h//2]])
                input_label = np.array([1])
            else:
                # Use bbox center and corners as prompts
                x, y, w, h = face_bbox
                input_point = np.array([
                    [x + w//2, y + h//2],  # Center
                    [x + w//4, y + h//4],  # Top-left region
                    [x + 3*w//4, y + 3*h//4]  # Bottom-right region
                ])
                input_label = np.array([1, 1, 1])
            
            # Generate masks
            masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True
            )
            
            # Select best mask
            best_mask_idx = np.argmax(scores)
            best_mask = masks[best_mask_idx]
            
            # Generate additional masks for different face regions
            eye_regions = self._get_eye_regions(image, face_bbox)
            
            return {
                'full_face_mask': best_mask,
                'eye_regions': eye_regions,
                'all_masks': masks,
                'scores': scores
            }
            
        except Exception as e:
            logger.error(f"Face segmentation failed: {e}")
            return {'full_face_mask': np.zeros(image.shape[:2], dtype=bool)}
    
    def _get_eye_regions(self, image: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]]) -> Dict[str, np.ndarray]:
        """Extract eye regions for glasses alignment."""
        try:
            if face_bbox is None:
                h, w = image.shape[:2]
                face_bbox = (w//4, h//4, w//2, h//2)
            
            x, y, bbox_w, bbox_h = face_bbox
            
            # Estimate eye positions based on facial proportions
            left_eye_x = x + int(bbox_w * 0.25)
            right_eye_x = x + int(bbox_w * 0.75)
            eyes_y = y + int(bbox_h * 0.4)
            
            eye_region_size = min(bbox_w // 6, bbox_h // 6, 30)
            
            # Generate masks for eye regions
            left_eye_point = np.array([[left_eye_x, eyes_y]])
            right_eye_point = np.array([[right_eye_x, eyes_y]])
            
            self.predictor.set_image(image)
            
            # Left eye mask
            left_masks, left_scores, _ = self.predictor.predict(
                point_coords=left_eye_point,
                point_labels=np.array([1]),
                multimask_output=True
            )
            left_eye_mask = left_masks[np.argmax(left_scores)]
            
            # Right eye mask
            right_masks, right_scores, _ = self.predictor.predict(
                point_coords=right_eye_point,
                point_labels=np.array([1]),
                multimask_output=True
            )
            right_eye_mask = right_masks[np.argmax(right_scores)]
            
            return {
                'left_eye_mask': left_eye_mask,
                'right_eye_mask': right_eye_mask,
                'left_eye_center': (left_eye_x, eyes_y),
                'right_eye_center': (right_eye_x, eyes_y)
            }
            
        except Exception as e:
            logger.warning(f"Eye region extraction failed: {e}")
            return {}

class DINOv2Processor:
    """Handles DINOv2 operations for feature extraction and understanding."""
    
    def __init__(self, model_name: str = "dinov2_vitb14", device: str = "auto"):
        """Initialize DINOv2 processor."""
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the DINOv2 model."""
        try:
            # Load DINOv2 from timm
            self.model = timm.create_model(
                'vit_base_patch14_dinov2.lvd142m',
                pretrained=True,
                num_classes=0  # Remove classification head
            )
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"DINOv2 model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load DINOv2 model: {e}")
            raise
    
    def extract_features(self, image: np.ndarray) -> torch.Tensor:
        """Extract features from image using DINOv2."""
        try:
            # Preprocess image
            processed_image = self._preprocess_image(image)
            
            # Extract features
            with torch.no_grad():
                features = self.model(processed_image.unsqueeze(0).to(self.device))
            
            return features.squeeze(0)  # Remove batch dimension
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return torch.zeros(768)  # Default feature size for DINOv2-base
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for DINOv2."""
        # Resize to 224x224 (DINOv2 input size)
        resized = cv2.resize(image, (224, 224))
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std
        
        # Convert to tensor and rearrange dimensions
        tensor = torch.from_numpy(normalized).permute(2, 0, 1)
        
        return tensor
    
    def compute_similarity(self, features1: torch.Tensor, features2: torch.Tensor) -> float:
        """Compute cosine similarity between two feature vectors."""
        try:
            # Normalize features
            features1_norm = F.normalize(features1, p=2, dim=0)
            features2_norm = F.normalize(features2, p=2, dim=0)
            
            # Compute cosine similarity
            similarity = torch.dot(features1_norm, features2_norm).item()
            
            return similarity
            
        except Exception as e:
            logger.error(f"Similarity computation failed: {e}")
            return 0.0

class GlassesAlignmentModule(nn.Module):
    """Neural module for glasses alignment and positioning."""
    
    def __init__(self, feature_dim: int = 768):
        """Initialize alignment module."""
        super().__init__()
        
        self.feature_dim = feature_dim
        
        # Transformation prediction network
        self.transform_net = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),  # Face + glasses features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 6)  # Affine transformation parameters
        )
        
        # Blending weights prediction
        self.blend_net = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, face_features: torch.Tensor, glasses_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict transformation and blending parameters."""
        # Concatenate features
        combined_features = torch.cat([face_features, glasses_features], dim=-1)
        
        # Predict transformation parameters
        transform_params = self.transform_net(combined_features)
        
        # Predict blending weight
        blend_weight = self.blend_net(combined_features)
        
        return {
            'transform_params': transform_params,
            'blend_weight': blend_weight
        }

class HybridVirtualTryOnModel:
    """Main hybrid model combining SAM and DINOv2 for virtual glasses try-on."""
    
    def __init__(self, device: str = "auto"):
        """Initialize the hybrid model."""
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.sam_processor = SAMProcessor(device=self.device)
        self.dino_processor = DINOv2Processor(device=self.device)
        self.alignment_module = GlassesAlignmentModule().to(self.device)
        
        # Load pre-trained alignment module if available
        self._load_alignment_weights()
        
        logger.info("Hybrid Virtual Try-On model initialized successfully")
    
    def _load_alignment_weights(self):
        """Load pre-trained weights for alignment module."""
        try:
            checkpoint_path = Path("models/alignment_weights.pth")
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.alignment_module.load_state_dict(checkpoint)
                logger.info("Loaded pre-trained alignment module weights")
            else:
                logger.info("No pre-trained alignment weights found, using random initialization")
        except Exception as e:
            logger.warning(f"Failed to load alignment weights: {e}")
    
    def process_selfie(self, selfie_image: np.ndarray, face_bbox: Optional[Tuple[int, int, int, int]] = None) -> Dict:
        """Process selfie image to extract face features and segmentation."""
        try:
            # Segment face using SAM
            segmentation_result = self.sam_processor.segment_face(selfie_image, face_bbox)
            
            # Extract features using DINOv2
            face_features = self.dino_processor.extract_features(selfie_image)
            
            # Extract eye region features separately
            eye_regions = segmentation_result.get('eye_regions', {})
            eye_features = {}
            
            if 'left_eye_center' in eye_regions and 'right_eye_center' in eye_regions:
                # Extract features from eye regions
                left_eye_patch = self._extract_eye_patch(selfie_image, eye_regions['left_eye_center'])
                right_eye_patch = self._extract_eye_patch(selfie_image, eye_regions['right_eye_center'])
                
                if left_eye_patch is not None:
                    eye_features['left'] = self.dino_processor.extract_features(left_eye_patch)
                if right_eye_patch is not None:
                    eye_features['right'] = self.dino_processor.extract_features(right_eye_patch)
            
            return {
                'face_features': face_features,
                'segmentation': segmentation_result,
                'eye_features': eye_features,
                'original_image': selfie_image
            }
            
        except Exception as e:
            logger.error(f"Selfie processing failed: {e}")
            return {}
    
    def process_glasses(self, glasses_image: np.ndarray) -> Dict:
        """Process glasses image to extract features."""
        try:
            # Extract features using DINOv2
            glasses_features = self.dino_processor.extract_features(glasses_image)
            
            return {
                'glasses_features': glasses_features,
                'original_image': glasses_image
            }
            
        except Exception as e:
            logger.error(f"Glasses processing failed: {e}")
            return {}
    
    def virtual_try_on(self, selfie_data: Dict, glasses_data: Dict) -> np.ndarray:
        """Perform virtual glasses try-on."""
        try:
            face_features = selfie_data['face_features']
            glasses_features = glasses_data['glasses_features']
            selfie_image = selfie_data['original_image']
            glasses_image = glasses_data['original_image']
            
            # Predict alignment parameters
            with torch.no_grad():
                alignment_result = self.alignment_module(face_features, glasses_features)
            
            transform_params = alignment_result['transform_params'].cpu().numpy()
            blend_weight = alignment_result['blend_weight'].cpu().numpy()
            
            # Apply transformation to glasses
            transformed_glasses = self._apply_transformation(glasses_image, transform_params)
            
            # Blend glasses with selfie
            result_image = self._blend_images(
                selfie_image, 
                transformed_glasses, 
                selfie_data['segmentation'],
                blend_weight
            )
            
            return result_image
            
        except Exception as e:
            logger.error(f"Virtual try-on failed: {e}")
            return selfie_data['original_image']
    
    def _extract_eye_patch(self, image: np.ndarray, eye_center: Tuple[int, int], patch_size: int = 64) -> Optional[np.ndarray]:
        """Extract eye patch from image."""
        try:
            x, y = eye_center
            half_size = patch_size // 2
            
            x1 = max(0, x - half_size)
            y1 = max(0, y - half_size)
            x2 = min(image.shape[1], x + half_size)
            y2 = min(image.shape[0], y + half_size)
            
            patch = image[y1:y2, x1:x2]
            
            # Resize to standard size
            if patch.shape[0] > 0 and patch.shape[1] > 0:
                patch = cv2.resize(patch, (patch_size, patch_size))
                return patch
            
            return None
            
        except Exception as e:
            logger.warning(f"Eye patch extraction failed: {e}")
            return None
    
    def _apply_transformation(self, glasses_image: np.ndarray, transform_params: np.ndarray) -> np.ndarray:
        """Apply affine transformation to glasses image."""
        try:
            # Reshape transform parameters to 2x3 matrix
            transform_matrix = transform_params.reshape(2, 3)
            
            # Apply affine transformation
            h, w = glasses_image.shape[:2]
            transformed = cv2.warpAffine(glasses_image, transform_matrix, (w, h))
            
            return transformed
            
        except Exception as e:
            logger.error(f"Transformation application failed: {e}")
            return glasses_image
    
    def _blend_images(self, selfie: np.ndarray, glasses: np.ndarray, 
                      segmentation: Dict, blend_weight: float) -> np.ndarray:
        """Blend glasses with selfie using segmentation masks."""
        try:
            # Ensure images have the same size
            if selfie.shape[:2] != glasses.shape[:2]:
                glasses = cv2.resize(glasses, (selfie.shape[1], selfie.shape[0]))
            
            # Get face mask
            face_mask = segmentation.get('full_face_mask', np.ones(selfie.shape[:2], dtype=bool))
            
            # Create glasses mask (assuming alpha channel or transparency)
            if glasses.shape[2] == 4:
                glasses_mask = glasses[:, :, 3] > 0
                glasses_rgb = glasses[:, :, :3]
            else:
                # Simple threshold for transparency detection
                gray_glasses = cv2.cvtColor(glasses, cv2.COLOR_RGB2GRAY)
                glasses_mask = gray_glasses > 10  # Threshold for non-black pixels
                glasses_rgb = glasses
            
            # Combine masks
            combined_mask = face_mask & glasses_mask
            
            # Blend images
            result = selfie.copy().astype(np.float32)
            glasses_rgb = glasses_rgb.astype(np.float32)
            
            # Apply blending in masked regions
            for c in range(3):  # RGB channels
                result[:, :, c] = np.where(
                    combined_mask,
                    result[:, :, c] * (1 - blend_weight) + glasses_rgb[:, :, c] * blend_weight,
                    result[:, :, c]
                )
            
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Image blending failed: {e}")
            return selfie
    
    def save_model(self, save_path: Path):
        """Save the alignment module weights."""
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save(self.alignment_module.state_dict(), save_path)
            logger.info(f"Model saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def train_alignment_module(self, train_dataloader, val_dataloader, epochs: int = 10):
        """Train the alignment module (placeholder for future implementation)."""
        logger.info("Training functionality will be implemented in the training script")
        pass

# Global hybrid model instance
hybrid_model = None

def get_hybrid_model() -> HybridVirtualTryOnModel:
    """Get or create the global hybrid model instance."""
    global hybrid_model
    if hybrid_model is None:
        hybrid_model = HybridVirtualTryOnModel()
    return hybrid_model