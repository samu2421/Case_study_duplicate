#!/usr/bin/env python3
"""
Simple demo script that works without complex pipeline dependencies.
Creates a basic virtual try-on demonstration.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from database.config import db_manager
from image_processing.utils.image_utils import image_processor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleTryOnDemo:
    """Simple demo class that doesn't require complex model loading."""
    
    def __init__(self):
        """Initialize simple demo."""
        self.demo_dir = Path(__file__).parent / "demo"
        self.output_dir = self.demo_dir / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def create_synthetic_selfie(self) -> np.ndarray:
        """Create a synthetic selfie for demo."""
        # Create base image
        image = np.full((512, 512, 3), [240, 240, 240], dtype=np.uint8)
        
        # Add face
        center = (256, 280)
        axes = (120, 150)
        face_color = (220, 190, 160)
        cv2.ellipse(image, center, axes, 0, 0, 360, face_color, -1)
        
        # Add eyes (important for glasses alignment)
        left_eye = (200, 240)
        right_eye = (312, 240)
        cv2.circle(image, left_eye, 15, (80, 60, 40), -1)
        cv2.circle(image, right_eye, 15, (80, 60, 40), -1)
        cv2.circle(image, left_eye, 8, (20, 20, 20), -1)
        cv2.circle(image, right_eye, 8, (20, 20, 20), -1)
        
        # Add eyebrows
        cv2.ellipse(image, (200, 220), (20, 8), 0, 0, 180, (100, 70, 50), -1)
        cv2.ellipse(image, (312, 220), (20, 8), 0, 0, 180, (100, 70, 50), -1)
        
        # Add nose
        nose_points = np.array([[256, 260], [250, 290], [262, 290]], np.int32)
        cv2.fillPoly(image, [nose_points], (200, 170, 140))
        
        # Add mouth
        cv2.ellipse(image, (256, 320), (25, 12), 0, 0, 180, (150, 100, 100), -1)
        
        return image
    
    def create_synthetic_glasses(self, style: str = "regular") -> np.ndarray:
        """Create synthetic glasses with different styles."""
        # Create RGBA image
        image = np.zeros((512, 512, 4), dtype=np.uint8)
        
        if style == "regular":
            # Regular prescription glasses
            frame_color = (50, 50, 50, 255)
            # Left lens
            cv2.circle(image, (200, 240), 70, frame_color, 6)
            # Right lens
            cv2.circle(image, (312, 240), 70, frame_color, 6)
            # Bridge
            cv2.line(image, (270, 240), (242, 240), frame_color, 4)
            # Arms
            cv2.line(image, (130, 240), (80, 230), frame_color, 4)
            cv2.line(image, (382, 240), (432, 230), frame_color, 4)
            
        elif style == "sunglasses":
            # Dark sunglasses
            frame_color = (20, 20, 20, 255)
            lens_color = (40, 40, 40, 180)
            # Left lens (filled)
            cv2.circle(image, (200, 240), 70, lens_color, -1)
            cv2.circle(image, (200, 240), 70, frame_color, 6)
            # Right lens (filled)
            cv2.circle(image, (312, 240), 70, lens_color, -1)
            cv2.circle(image, (312, 240), 70, frame_color, 6)
            # Bridge
            cv2.line(image, (270, 240), (242, 240), frame_color, 6)
            # Arms
            cv2.line(image, (130, 240), (80, 230), frame_color, 6)
            cv2.line(image, (382, 240), (432, 230), frame_color, 6)
            
        elif style == "cat_eye":
            # Cat eye glasses
            frame_color = (100, 50, 150, 255)
            # Left lens (cat eye shape)
            points_left = np.array([[130, 240], [200, 220], [270, 240], [200, 260]], np.int32)
            cv2.polylines(image, [points_left], True, frame_color, 6)
            # Right lens (cat eye shape)
            points_right = np.array([[242, 240], [312, 220], [382, 240], [312, 260]], np.int32)
            cv2.polylines(image, [points_right], True, frame_color, 6)
            # Bridge
            cv2.line(image, (270, 240), (242, 240), frame_color, 4)
            # Arms
            cv2.line(image, (130, 240), (80, 230), frame_color, 4)
            cv2.line(image, (382, 240), (432, 230), frame_color, 4)
        
        return image
    
    def simple_overlay(self, face_image: np.ndarray, glasses_image: np.ndarray) -> np.ndarray:
        """Simple glasses overlay without complex alignment."""
        # Ensure both images are same size
        if face_image.shape[:2] != glasses_image.shape[:2]:
            glasses_image = cv2.resize(glasses_image, (face_image.shape[1], face_image.shape[0]))
        
        # Convert face image to RGBA if needed
        if face_image.shape[2] == 3:
            face_rgba = cv2.cvtColor(face_image, cv2.COLOR_RGB2RGBA)
        else:
            face_rgba = face_image.copy()
        
        # Blend glasses onto face
        result = face_rgba.copy().astype(np.float32)
        glasses_float = glasses_image.astype(np.float32)
        
        # Simple alpha blending
        alpha = glasses_float[:, :, 3:4] / 255.0
        alpha = np.repeat(alpha, 3, axis=2)
        
        result[:, :, :3] = (1 - alpha) * result[:, :, :3] + alpha * glasses_float[:, :, :3]
        
        return result.astype(np.uint8)[:, :, :3]  # Return RGB only
    
    def create_demo_combinations(self) -> bool:
        """Create demo combinations and save results."""
        try:
            logger.info("Creating demo combinations...")
            
            # Create selfie
            selfie = self.create_synthetic_selfie()
            
            # Create different glasses styles
            glasses_styles = {
                "regular": "Regular Glasses",
                "sunglasses": "Sunglasses", 
                "cat_eye": "Cat Eye Glasses"
            }
            
            results = []
            
            for style, name in glasses_styles.items():
                logger.info(f"Creating combination: {name}")
                
                # Create glasses
                glasses = self.create_synthetic_glasses(style)
                
                # Create try-on result
                result = self.simple_overlay(selfie, glasses)
                
                # Save individual result
                output_path = self.output_dir / f"tryon_{style}.jpg"
                image_processor.save_image(result, output_path)
                
                results.append((selfie, result, name))
            
            # Create comparison grid
            self.create_comparison_grid(results)
            
            logger.info(f"✅ Demo combinations created in {self.output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create demo combinations: {e}")
            return False
    
    def create_comparison_grid(self, results: list):
        """Create a comparison grid showing original vs try-on results."""
        try:
            num_combinations = len(results)
            img_size = 256
            
            # Create grid (2 rows: original, try-on results)
            grid_width = num_combinations * img_size
            grid_height = 2 * img_size
            grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
            
            for i, (original, result, name) in enumerate(results):
                # Resize images
                orig_resized = cv2.resize(original, (img_size, img_size))
                result_resized = cv2.resize(result, (img_size, img_size))
                
                # Place in grid
                x_start = i * img_size
                
                # Original (top row)
                grid[0:img_size, x_start:x_start + img_size] = orig_resized
                
                # Try-on result (bottom row)
                grid[img_size:2*img_size, x_start:x_start + img_size] = result_resized
                
                # Add text labels
                cv2.putText(grid, "Original", (x_start + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(grid, name, (x_start + 10, img_size + 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Save comparison grid
            grid_path = self.output_dir / "comparison_grid.jpg"
            image_processor.save_image(grid, grid_path)
            
            logger.info(f"✅ Comparison grid saved to {grid_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create comparison grid: {e}")
    
    def run_database_demo(self) -> bool:
        """Try to run demo with database data if available."""
        try:
            logger.info("Attempting database demo...")
            
            # Check if database has data
            if not db_manager.connect():
                logger.warning("Database not available, using synthetic data")
                return False
            
            # Check for usable selfies
            selfies_query = """
            SELECT COUNT(*) as count FROM diffusion.selfies 
            WHERE image_data IS NOT NULL LIMIT 1;
            """
            selfies_result = db_manager.execute_query(selfies_query)
            
            # Check for usable glasses
            glasses_query = """
            SELECT COUNT(*) as count FROM diffusion.processed_glasses 
            WHERE image_data IS NOT NULL LIMIT 1;
            """
            glasses_result = db_manager.execute_query(glasses_query)
            
            has_selfies = len(selfies_result) > 0 and selfies_result.iloc[0]['count'] > 0
            has_glasses = len(glasses_result) > 0 and glasses_result.iloc[0]['count'] > 0
            
            if has_selfies and has_glasses:
                logger.info("✅ Database has data for demo")
                return True
            else:
                logger.info("⚠️ Database lacks sufficient data, using synthetic")
                return False
                
        except Exception as e:
            logger.info(f"Database demo not available: {e}")
            return False

def main():
    """Main demo function."""
    try:
        logger.info("🎯 SIMPLE VIRTUAL GLASSES TRY-ON DEMO")
        logger.info("=" * 50)
        
        demo = SimpleTryOnDemo()
        
        # Try database demo first
        logger.info("--- CHECKING FOR DATABASE DATA ---")
        has_db_data = demo.run_database_demo()
        
        if has_db_data:
            logger.info("Using database data for demo...")
            # Try to run full demo
            try:
                from demo.demo_tryon import VirtualTryOnDemo
                full_demo = VirtualTryOnDemo()
                results = full_demo.run_demo_batch(num_selfies=2, num_glasses=2)
                
                if results['success'] > 0:
                    logger.info("✅ Full database demo successful!")
                    return 0
                else:
                    logger.info("Database demo failed, falling back to synthetic")
            except Exception as e:
                logger.info(f"Full demo failed: {e}, using synthetic")
        
        # Fallback to synthetic demo
        logger.info("--- CREATING SYNTHETIC DEMO ---")
        if demo.create_demo_combinations():
            logger.info("✅ Simple demo completed successfully!")
            logger.info(f"📁 Results saved to: {demo.output_dir}")
            logger.info("📸 View comparison_grid.jpg to see results")
            return 0
        else:
            logger.error("❌ Simple demo failed")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())