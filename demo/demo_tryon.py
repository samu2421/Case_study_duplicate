"""
Virtual Glasses Try-On Demo Script
This script demonstrates the basic virtual try-on functionality
"""
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import logging
import argparse

# Import project modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.database_config import DatabaseManager
from image_processing.utils.path_utils import ProjectPaths
from image_processing.utils.image_utils import ImageProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VirtualTryOnDemo:
    """Demo class for virtual glasses try-on"""
    
    def __init__(self):
        """Initialize the demo system"""
        self.db_manager = DatabaseManager()
        self.paths = ProjectPaths()
        self.image_processor = ImageProcessor()
        
        # Ensure demo directories exist
        self.paths.demo_dir.mkdir(exist_ok=True)
        self.paths.demo_selfies_dir.mkdir(exist_ok=True)
        self.paths.demo_output_dir.mkdir(exist_ok=True)
        
        # Demo parameters
        self.glasses_scale_factor = 0.8  # How much to scale glasses relative to face
        self.glasses_y_offset = -20  # Vertical offset for glasses positioning
        self.blend_alpha = 0.9  # Alpha blending factor for glasses
        
        logger.info("VirtualTryOnDemo initialized")
    
    def detect_face_landmarks(self, image: np.ndarray) -> Optional[Dict]:
        """
        Detect face landmarks using OpenCV
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with face detection results
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Load face cascade
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
            
            if len(faces) == 0:
                logger.warning("No face detected in image")
                return None
            
            # Use the largest face
            face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = face
            
            # Estimate eye positions (approximate)
            eye_y = y + h * 0.35  # Eyes are roughly 35% down from top of face
            left_eye_x = x + w * 0.35  # Left eye at 35% from left
            right_eye_x = x + w * 0.65  # Right eye at 65% from left
            
            face_info = {
                'face_rect': (x, y, w, h),
                'face_center': (x + w // 2, y + h // 2),
                'left_eye': (int(left_eye_x), int(eye_y)),
                'right_eye': (int(right_eye_x), int(eye_y)),
                'eye_distance': abs(right_eye_x - left_eye_x),
                'face_width': w,
                'face_height': h
            }
            
            return face_info
            
        except Exception as e:
            logger.error(f"Error detecting face landmarks: {e}")
            return None
    
    def prepare_glasses(self, glasses_image: np.ndarray, 
                       target_width: int) -> np.ndarray:
        """
        Prepare glasses image for overlay
        
        Args:
            glasses_image: Glasses image with alpha channel
            target_width: Target width for glasses
            
        Returns:
            Prepared glasses image
        """
        try:
            # Ensure glasses image has alpha channel
            if glasses_image.shape[2] == 3:
                # Add alpha channel
                alpha = np.ones((glasses_image.shape[0], glasses_image.shape[1], 1), dtype=glasses_image.dtype) * 255
                glasses_image = np.concatenate([glasses_image, alpha], axis=2)
            
            # Calculate new dimensions maintaining aspect ratio
            current_height, current_width = glasses_image.shape[:2]
            aspect_ratio = current_height / current_width
            target_height = int(target_width * aspect_ratio)
            
            # Resize glasses
            glasses_resized = cv2.resize(glasses_image, (target_width, target_height), interpolation=cv2.INTER_AREA)
            
            return glasses_resized
            
        except Exception as e:
            logger.error(f"Error preparing glasses: {e}")
            return glasses_image
    
    def overlay_glasses(self, selfie: np.ndarray, glasses: np.ndarray,
                       face_info: Dict) -> np.ndarray:
        """
        Overlay glasses on selfie
        
        Args:
            selfie: Selfie image
            glasses: Glasses image with alpha channel
            face_info: Face detection information
            
        Returns:
            Image with glasses overlaid
        """
        try:
            # Calculate glasses size based on face width
            glasses_width = int(face_info['face_width'] * self.glasses_scale_factor)
            
            # Prepare glasses
            glasses_prepared = self.prepare_glasses(glasses, glasses_width)
            
            # Calculate position for glasses
            glasses_height = glasses_prepared.shape[0]
            
            # Position glasses centered on eyes with offset
            center_x = (face_info['left_eye'][0] + face_info['right_eye'][0]) // 2
            center_y = face_info['left_eye'][1] + self.glasses_y_offset
            
            # Calculate top-left corner for glasses placement
            glasses_x = center_x - glasses_width // 2
            glasses_y = center_y - glasses_height // 2
            
            # Ensure glasses fit within image bounds
            glasses_x = max(0, min(glasses_x, selfie.shape[1] - glasses_width))
            glasses_y = max(0, min(glasses_y, selfie.shape[0] - glasses_height))
            
            # Adjust glasses size if needed to fit
            max_width = selfie.shape[1] - glasses_x
            max_height = selfie.shape[0] - glasses_y
            
            if glasses_width > max_width or glasses_height > max_height:
                # Recalculate size
                scale_factor = min(max_width / glasses_width, max_height / glasses_height)
                new_width = int(glasses_width * scale_factor)
                new_height = int(glasses_height * scale_factor)
                glasses_prepared = cv2.resize(glasses_prepared, (new_width, new_height))
                glasses_width, glasses_height = new_width, new_height
            
            # Create result image
            result = selfie.copy()
            
            # Extract alpha channel from glasses
            if glasses_prepared.shape[2] == 4:
                glasses_bgr = glasses_prepared[:, :, :3]
                glasses_alpha = glasses_prepared[:, :, 3] / 255.0
            else:
                glasses_bgr = glasses_prepared
                glasses_alpha = np.ones((glasses_height, glasses_width), dtype=np.float32)
            
            # Apply alpha blending
            for c in range(3):  # For each color channel
                result[glasses_y:glasses_y+glasses_height, glasses_x:glasses_x+glasses_width, c] = \
                    (1 - glasses_alpha * self.blend_alpha) * result[glasses_y:glasses_y+glasses_height, glasses_x:glasses_x+glasses_width, c] + \
                    (glasses_alpha * self.blend_alpha) * glasses_bgr[:, :, c]
            
            return result
            
        except Exception as e:
            logger.error(f"Error overlaying glasses: {e}")
            return selfie
    
    def try_on_glasses(self, selfie_path: Path, glasses_path: Path,
                      output_path: Optional[Path] = None) -> bool:
        """
        Perform virtual try-on
        
        Args:
            selfie_path: Path to selfie image
            glasses_path: Path to glasses image
            output_path: Path to save result (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load images
            selfie = self.image_processor.load_image(selfie_path)
            glasses = self.image_processor.load_image(glasses_path)
            
            if selfie is None:
                logger.error(f"Failed to load selfie: {selfie_path}")
                return False
            
            if glasses is None:
                logger.error(f"Failed to load glasses: {glasses_path}")
                return False
            
            logger.info(f"Loaded selfie: {selfie.shape}, glasses: {glasses.shape}")
            
            # Detect face in selfie
            face_info = self.detect_face_landmarks(selfie)
            if face_info is None:
                logger.warning("No face detected, using fallback positioning")
                # Create fallback face info for center positioning
                h, w = selfie.shape[:2]
                face_info = {
                    'face_rect': (w//4, h//4, w//2, h//2),
                    'face_center': (w//2, h//2),
                    'left_eye': (w//2 - w//8, h//2 - h//8),
                    'right_eye': (w//2 + w//8, h//2 - h//8),
                    'eye_distance': w//4,
                    'face_width': w//2,
                    'face_height': h//2
                }
                logger.info("Using fallback face positioning for demo")
            else:
                logger.info(f"Face detected: {face_info['face_rect']}")
            
            # Perform try-on
            result = self.overlay_glasses(selfie, glasses, face_info)
            
            # Save result
            if output_path is None:
                output_path = self.paths.demo_output_dir / f"tryon_{selfie_path.stem}_{glasses_path.stem}.jpg"
            
            success = self.image_processor.save_image(result, output_path)
            
            if success:
                logger.info(f"Virtual try-on completed: {output_path}")
                return True
            else:
                logger.error("Failed to save result image")
                return False
            
        except Exception as e:
            logger.error(f"Error in virtual try-on: {e}")
            return False
    
    def demo_with_database_glasses(self, selfie_path: Path,
                                 glasses_title: Optional[str] = None) -> bool:
        """
        Demo using glasses from database
        
        Args:
            selfie_path: Path to selfie image
            glasses_title: Specific glasses title to use (optional)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get glasses from database
            glasses_df = self.db_manager.get_glasses_data(limit=10)
            
            if glasses_df.empty:
                logger.error("No glasses found in database")
                return False
            
            # Select glasses
            if glasses_title:
                selected_glasses = glasses_df[glasses_df['title'] == glasses_title]
                if selected_glasses.empty:
                    logger.error(f"Glasses with title '{glasses_title}' not found")
                    return False
                glasses_row = selected_glasses.iloc[0]
            else:
                # Use first available glasses
                glasses_row = glasses_df.iloc[0]
            
            logger.info(f"Using glasses: {glasses_row['title']}")
            
            # Download glasses image
            main_image_url = glasses_row['main_image']
            glasses_image = self.image_processor.load_image_from_url(main_image_url)
            
            if glasses_image is None:
                logger.error("Failed to download glasses image")
                return False
            
            # Save glasses temporarily
            temp_glasses_path = self.paths.demo_dir / "temp_glasses.png"
            self.image_processor.save_image(glasses_image, temp_glasses_path)
            
            # Perform try-on
            result = self.try_on_glasses(selfie_path, temp_glasses_path)
            
            # Clean up
            if temp_glasses_path.exists():
                temp_glasses_path.unlink()
            
            return result
            
        except Exception as e:
            logger.error(f"Error in database demo: {e}")
            return False
    
    def create_sample_selfie(self) -> Path:
        """
        Create a sample selfie for demo with more realistic face features
        
        Returns:
            Path to created sample selfie
        """
        # Create a more realistic face image
        sample_image = np.ones((512, 512, 3), dtype=np.uint8) * 220  # Light background
        
        # Create a more detailed face that OpenCV can detect
        center_x, center_y = 256, 200
        face_width, face_height = 160, 200
        
        # Face oval (using ellipse)
        cv2.ellipse(sample_image, (center_x, center_y), (face_width//2, face_height//2), 
                   0, 0, 360, (210, 180, 140), -1)  # Skin color
        
        # Face shadow/contour
        cv2.ellipse(sample_image, (center_x, center_y), (face_width//2-5, face_height//2-5), 
                   0, 0, 360, (200, 170, 130), 3)
        
        # Eyes (more realistic)
        eye_y = center_y - 30
        left_eye_x = center_x - 35
        right_eye_x = center_x + 35
        
        # Eye whites
        cv2.ellipse(sample_image, (left_eye_x, eye_y), (15, 10), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(sample_image, (right_eye_x, eye_y), (15, 10), 0, 0, 360, (255, 255, 255), -1)
        
        # Pupils
        cv2.circle(sample_image, (left_eye_x, eye_y), 8, (20, 20, 20), -1)
        cv2.circle(sample_image, (right_eye_x, eye_y), 8, (20, 20, 20), -1)
        
        # Eye reflections
        cv2.circle(sample_image, (left_eye_x-3, eye_y-3), 3, (255, 255, 255), -1)
        cv2.circle(sample_image, (right_eye_x-3, eye_y-3), 3, (255, 255, 255), -1)
        
        # Eyebrows
        cv2.ellipse(sample_image, (left_eye_x, eye_y-20), (20, 8), 0, 0, 180, (80, 60, 40), -1)
        cv2.ellipse(sample_image, (right_eye_x, eye_y-20), (20, 8), 0, 0, 180, (80, 60, 40), -1)
        
        # Nose
        nose_points = np.array([
            [center_x-8, center_y-10],
            [center_x+8, center_y-10],
            [center_x+12, center_y+10],
            [center_x-12, center_y+10]
        ], np.int32)
        cv2.fillPoly(sample_image, [nose_points], (190, 160, 120))
        
        # Nostrils
        cv2.ellipse(sample_image, (center_x-6, center_y+8), (3, 2), 0, 0, 360, (150, 120, 80), -1)
        cv2.ellipse(sample_image, (center_x+6, center_y+8), (3, 2), 0, 0, 360, (150, 120, 80), -1)
        
        # Mouth
        cv2.ellipse(sample_image, (center_x, center_y+50), (25, 12), 0, 0, 360, (150, 80, 80), -1)
        cv2.ellipse(sample_image, (center_x, center_y+50), (20, 8), 0, 0, 360, (120, 60, 60), -1)
        
        # Hair
        cv2.ellipse(sample_image, (center_x, center_y-80), (90, 60), 0, 0, 180, (60, 40, 20), -1)
        
        # Add some texture/noise to make it more realistic
        noise = np.random.normal(0, 5, sample_image.shape).astype(np.int16)
        sample_image = np.clip(sample_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Save sample
        sample_path = self.paths.demo_selfies_dir / "sample_selfie.jpg"
        self.image_processor.save_image(sample_image, sample_path)
        
        logger.info(f"Created sample selfie: {sample_path}")
        return sample_path

    def debug_face_detection(self, image_path: Path) -> bool:
        """
        Debug face detection by saving an image with detected faces marked
        
        Args:
            image_path: Path to image to debug
            
        Returns:
            True if faces detected, False otherwise
        """
        try:
            image = self.image_processor.load_image(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return False
            
            # Detect faces
            face_info = self.detect_face_landmarks(image)
            
            # Create debug image
            debug_image = image.copy()
            
            if face_info:
                # Draw face rectangle
                x, y, w, h = face_info['face_rect']
                cv2.rectangle(debug_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Draw eye positions
                cv2.circle(debug_image, face_info['left_eye'], 5, (255, 0, 0), -1)
                cv2.circle(debug_image, face_info['right_eye'], 5, (255, 0, 0), -1)
                
                # Draw face center
                cv2.circle(debug_image, face_info['face_center'], 3, (0, 0, 255), -1)
                
                # Add text
                cv2.putText(debug_image, f"Face: {w}x{h}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(debug_image, f"Eyes: {face_info['eye_distance']}px", 
                           (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                logger.info(f"Face detected: {face_info['face_rect']}")
            else:
                cv2.putText(debug_image, "NO FACE DETECTED", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                logger.warning("No face detected")
            
            # Save debug image
            debug_path = self.paths.demo_output_dir / f"debug_{image_path.stem}.jpg"
            self.image_processor.save_image(debug_image, debug_path)
            logger.info(f"Debug image saved: {debug_path}")
            
            return face_info is not None
            
        except Exception as e:
            logger.error(f"Error in face detection debug: {e}")
            return False
        """
        Create a sample selfie for demo with more realistic face features
        
        Returns:
            Path to created sample selfie
        """
        # Create a more realistic face image
        sample_image = np.ones((512, 512, 3), dtype=np.uint8) * 220  # Light background
        
        # Create a more detailed face that OpenCV can detect
        center_x, center_y = 256, 200
        face_width, face_height = 160, 200
        
        # Face oval (using ellipse)
        cv2.ellipse(sample_image, (center_x, center_y), (face_width//2, face_height//2), 
                   0, 0, 360, (210, 180, 140), -1)  # Skin color
        
        # Face shadow/contour
        cv2.ellipse(sample_image, (center_x, center_y), (face_width//2-5, face_height//2-5), 
                   0, 0, 360, (200, 170, 130), 3)
        
        # Eyes (more realistic)
        eye_y = center_y - 30
        left_eye_x = center_x - 35
        right_eye_x = center_x + 35
        
        # Eye whites
        cv2.ellipse(sample_image, (left_eye_x, eye_y), (15, 10), 0, 0, 360, (255, 255, 255), -1)
        cv2.ellipse(sample_image, (right_eye_x, eye_y), (15, 10), 0, 0, 360, (255, 255, 255), -1)
        
        # Pupils
        cv2.circle(sample_image, (left_eye_x, eye_y), 8, (20, 20, 20), -1)
        cv2.circle(sample_image, (right_eye_x, eye_y), 8, (20, 20, 20), -1)
        
        # Eye reflections
        cv2.circle(sample_image, (left_eye_x-3, eye_y-3), 3, (255, 255, 255), -1)
        cv2.circle(sample_image, (right_eye_x-3, eye_y-3), 3, (255, 255, 255), -1)
        
        # Eyebrows
        cv2.ellipse(sample_image, (left_eye_x, eye_y-20), (20, 8), 0, 0, 180, (80, 60, 40), -1)
        cv2.ellipse(sample_image, (right_eye_x, eye_y-20), (20, 8), 0, 0, 180, (80, 60, 40), -1)
        
        # Nose
        nose_points = np.array([
            [center_x-8, center_y-10],
            [center_x+8, center_y-10],
            [center_x+12, center_y+10],
            [center_x-12, center_y+10]
        ], np.int32)
        cv2.fillPoly(sample_image, [nose_points], (190, 160, 120))
        
        # Nostrils
        cv2.ellipse(sample_image, (center_x-6, center_y+8), (3, 2), 0, 0, 360, (150, 120, 80), -1)
        cv2.ellipse(sample_image, (center_x+6, center_y+8), (3, 2), 0, 0, 360, (150, 120, 80), -1)
        
        # Mouth
        cv2.ellipse(sample_image, (center_x, center_y+50), (25, 12), 0, 0, 360, (150, 80, 80), -1)
        cv2.ellipse(sample_image, (center_x, center_y+50), (20, 8), 0, 0, 360, (120, 60, 60), -1)
        
        # Hair
        cv2.ellipse(sample_image, (center_x, center_y-80), (90, 60), 0, 0, 180, (60, 40, 20), -1)
        
        # Add some texture/noise to make it more realistic
        noise = np.random.normal(0, 5, sample_image.shape).astype(np.int16)
        sample_image = np.clip(sample_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Save sample
        sample_path = self.paths.demo_selfies_dir / "sample_selfie.jpg"
        self.image_processor.save_image(sample_image, sample_path)
        
        logger.info(f"Created sample selfie: {sample_path}")
        return sample_path
    
    def debug_face_detection(self, image_path: Path) -> bool:
        """
        Debug face detection by saving an image with detected faces marked
        
        Args:
            image_path: Path to image to debug
            
        Returns:
            True if faces detected, False otherwise
        """
        try:
            image = self.image_processor.load_image(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return False
            
            # Detect faces
            face_info = self.detect_face_landmarks(image)
            
            # Create debug image
            debug_image = image.copy()
            
            if face_info:
                # Draw face rectangle
                x, y, w, h = face_info['face_rect']
                cv2.rectangle(debug_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Draw eye positions
                cv2.circle(debug_image, face_info['left_eye'], 5, (255, 0, 0), -1)
                cv2.circle(debug_image, face_info['right_eye'], 5, (255, 0, 0), -1)
                
                # Draw face center
                cv2.circle(debug_image, face_info['face_center'], 3, (0, 0, 255), -1)
                
                # Add text
                cv2.putText(debug_image, f"Face: {w}x{h}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(debug_image, f"Eyes: {face_info['eye_distance']}px", 
                           (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                logger.info(f"Face detected: {face_info['face_rect']}")
            else:
                cv2.putText(debug_image, "NO FACE DETECTED", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                logger.warning("No face detected")
            
            # Save debug image
            debug_path = self.paths.demo_output_dir / f"debug_{image_path.stem}.jpg"
            self.image_processor.save_image(debug_image, debug_path)
            logger.info(f"Debug image saved: {debug_path}")
            
            return face_info is not None
            
        except Exception as e:
            logger.error(f"Error in face detection debug: {e}")
            return False


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description='Virtual Glasses Try-On Demo')
    parser.add_argument('--selfie', type=str, help='Path to selfie image')
    parser.add_argument('--glasses', type=str, help='Path to glasses image')
    parser.add_argument('--glasses-title', type=str, help='Glasses title from database')
    parser.add_argument('--create-sample', action='store_true', help='Create sample selfie')
    parser.add_argument('--output', type=str, help='Output path for result')
    parser.add_argument('--debug-face', action='store_true', help='Debug face detection on selfie')
    
    args = parser.parse_args()
    
    # Initialize demo
    demo = VirtualTryOnDemo()
    
    # Create sample selfie if requested
    if args.create_sample:
        sample_path = demo.create_sample_selfie()
        print(f"Sample selfie created: {sample_path}")
        return
    
    # Debug face detection if requested
    if args.debug_face:
        if args.selfie:
            selfie_path = Path(args.selfie)
        else:
            # Look for sample selfie
            sample_path = demo.paths.demo_selfies_dir / "sample_selfie.jpg"
            if sample_path.exists():
                selfie_path = sample_path
            else:
                print("No selfie provided and no sample found. Use --create-sample first.")
                return
        
        face_detected = demo.debug_face_detection(selfie_path)
        print(f"Face detection result: {'Success' if face_detected else 'Failed'}")
        print(f"Check debug image in: {demo.paths.demo_output_dir}")
        return
    
    # Determine selfie path
    if args.selfie:
        selfie_path = Path(args.selfie)
    else:
        # Look for sample selfie
        sample_path = demo.paths.demo_selfies_dir / "sample_selfie.jpg"
        if sample_path.exists():
            selfie_path = sample_path
        else:
            print("No selfie provided and no sample found. Use --create-sample first.")
            return
    
    if not selfie_path.exists():
        print(f"Selfie not found: {selfie_path}")
        return
    
    # Perform try-on
    if args.glasses:
        # Use local glasses file
        glasses_path = Path(args.glasses)
        if not glasses_path.exists():
            print(f"Glasses file not found: {glasses_path}")
            return
        
        output_path = Path(args.output) if args.output else None
        success = demo.try_on_glasses(selfie_path, glasses_path, output_path)
        
    else:
        # Use glasses from database
        success = demo.demo_with_database_glasses(selfie_path, args.glasses_title)
    
    if success:
        print("Virtual try-on completed successfully!")
        print(f"Check output in: {demo.paths.demo_output_dir}")
    else:
        print("Virtual try-on failed")


if __name__ == "__main__":
    # If run directly without arguments, run a simple test
    if len(sys.argv) == 1:
        print("Running simple demo test...")
        
        demo = VirtualTryOnDemo()
        
        # Test database connection
        if demo.db_manager.test_connection():
            print("Database connection successful")
            
            # Create sample selfie
            sample_selfie = demo.create_sample_selfie()
            
            # Try with database glasses
            success = demo.demo_with_database_glasses(sample_selfie)
            
            if success:
                print("Demo completed successfully!")
                print(f"Check output in: {demo.paths.demo_output_dir}")
            else:
                print("Demo failed")
        else:
            print("Database connection failed")
    else:
        main()
