"""
Demo script for Virtual Glasses Try-On system.
Demonstrates loading selfies and glasses from database and performing virtual try-on.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import logging
from typing import Optional, Dict, List
import pandas as pd
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from database.config import db_manager
from image_processing.utils.image_utils import image_processor
from image_processing.preprocess.preprocess_selfies import selfie_preprocessor
from image_processing.preprocess.preprocess_glasses import glasses_preprocessor
from models.hybrid_model import get_hybrid_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VirtualTryOnDemo:
    """Demo class for virtual glasses try-on."""
    
    def __init__(self):
        """Initialize demo."""
        self.demo_dir = Path(__file__).parent
        self.output_dir = self.demo_dir / "output"
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.hybrid_model = get_hybrid_model()
        logger.info("Virtual Try-On Demo initialized")
    
    def setup_database_connection(self) -> bool:
        """Setup database connection."""
        try:
            if not db_manager.connect():
                logger.error("Failed to connect to database")
                return False
            
            logger.info("Database connection established")
            return True
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            return False
    
    def get_sample_selfies(self, limit: int = 5) -> List[Dict]:
        """Get sample selfies from database."""
        try:
            # Get high-quality selfies with faces
            query = f"""
            SELECT id, filename, quality_score, face_detected
            FROM {db_manager.config['schema']}.selfies
            WHERE face_detected = true AND quality_score > 0.4
            ORDER BY quality_score DESC
            LIMIT {limit};
            """
            
            selfies_df = db_manager.execute_query(query)
            
            samples = []
            for _, row in selfies_df.iterrows():
                # Get image data
                image_data = db_manager.get_selfie_image_data(row['id'])
                if image_data is not None:
                    image = image_processor.load_image(image_data)
                    if image is not None:
                        samples.append({
                            'id': row['id'],
                            'filename': row['filename'],
                            'image': image,
                            'quality_score': row['quality_score']
                        })
            
            logger.info(f"Retrieved {len(samples)} sample selfies")
            return samples
            
        except Exception as e:
            logger.error(f"Failed to get sample selfies: {e}")
            return []
    
    def get_sample_glasses(self, limit: int = 5) -> List[Dict]:
        """Get sample glasses from database."""
        try:
            # Get processed glasses
            query = f"""
            SELECT pg.id, pg.original_id, pg.title, pg.glasses_type, pg.transparency_ratio
            FROM {db_manager.config['schema']}.processed_glasses pg
            WHERE pg.has_transparency = true
            ORDER BY pg.transparency_ratio DESC
            LIMIT {limit};
            """
            
            glasses_df = db_manager.execute_query(query)
            
            samples = []
            for _, row in glasses_df.iterrows():
                # Get processed image data
                image_query = f"""
                SELECT image_data FROM {db_manager.config['schema']}.processed_glasses 
                WHERE id = %(glasses_id)s;
                """
                
                result = db_manager.execute_query(image_query, {'glasses_id': row['id']})
                if len(result) > 0 and result.iloc[0]['image_data'] is not None:
                    # Load image from bytes
                    image_bytes = result.iloc[0]['image_data']
                    image = image_processor.load_image(image_bytes)
                    if image is not None:
                        samples.append({
                            'id': row['id'],
                            'title': row['title'],
                            'image': image,
                            'glasses_type': row['glasses_type'],
                            'transparency_ratio': row['transparency_ratio']
                        })
            
            logger.info(f"Retrieved {len(samples)} sample glasses")
            return samples
            
        except Exception as e:
            logger.error(f"Failed to get sample glasses: {e}")
            return []
    
    def perform_single_try_on(self, selfie_data: Dict, glasses_data: Dict) -> Optional[np.ndarray]:
        """Perform virtual try-on for a single selfie-glasses pair."""
        try:
            logger.info(f"Trying on glasses '{glasses_data['title']}' on selfie '{selfie_data['filename']}'")
            
            # Process selfie
            selfie_result = self.hybrid_model.process_selfie(selfie_data['image'])
            if not selfie_result:
                logger.error("Failed to process selfie")
                return None
            
            # Process glasses
            glasses_result = self.hybrid_model.process_glasses(glasses_data['image'])
            if not glasses_result:
                logger.error("Failed to process glasses")
                return None
            
            # Perform virtual try-on
            result_image = self.hybrid_model.virtual_try_on(selfie_result, glasses_result)
            
            return result_image
            
        except Exception as e:
            logger.error(f"Virtual try-on failed: {e}")
            return None
    
    def run_demo_batch(self, num_selfies: int = 3, num_glasses: int = 3) -> Dict[str, int]:
        """Run demo with multiple selfie-glasses combinations."""
        try:
            # Setup database
            if not self.setup_database_connection():
                return {'total': 0, 'success': 0, 'failed': 0}
            
            # Get sample data
            sample_selfies = self.get_sample_selfies(num_selfies)
            sample_glasses = self.get_sample_glasses(num_glasses)
            
            if not sample_selfies or not sample_glasses:
                logger.error("No sample data available")
                return {'total': 0, 'success': 0, 'failed': 0}
            
            # Perform try-on combinations
            total_combinations = len(sample_selfies) * len(sample_glasses)
            success_count = 0
            failed_count = 0
            
            logger.info(f"Running {total_combinations} try-on combinations...")
            
            with tqdm(total=total_combinations, desc="Processing combinations") as pbar:
                for i, selfie_data in enumerate(sample_selfies):
                    for j, glasses_data in enumerate(sample_glasses):
                        try:
                            # Perform try-on
                            result_image = self.perform_single_try_on(selfie_data, glasses_data)
                            
                            if result_image is not None:
                                # Save result
                                output_filename = f"tryon_{i}_{j}_{selfie_data['id'][:8]}_{glasses_data['id'][:8]}.jpg"
                                output_path = self.output_dir / output_filename
                                
                                if image_processor.save_image(result_image, output_path):
                                    success_count += 1
                                    logger.debug(f"Saved result: {output_filename}")
                                else:
                                    failed_count += 1
                            else:
                                failed_count += 1
                                
                        except Exception as e:
                            logger.error(f"Combination {i},{j} failed: {e}")
                            failed_count += 1
                        
                        pbar.update(1)
            
            results = {
                'total': total_combinations,
                'success': success_count,
                'failed': failed_count,
                'success_rate': success_count / total_combinations if total_combinations > 0 else 0
            }
            
            logger.info(f"Demo completed: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Demo batch failed: {e}")
            return {'total': 0, 'success': 0, 'failed': 0}
    
    def create_comparison_grid(self, num_combinations: int = 6) -> bool:
        """Create a comparison grid showing original vs try-on results."""
        try:
            # Get sample data
            sample_selfies = self.get_sample_selfies(2)
            sample_glasses = self.get_sample_glasses(3)
            
            if len(sample_selfies) < 2 or len(sample_glasses) < 3:
                logger.error("Insufficient sample data for comparison grid")
                return False
            
            # Create grid layout
            grid_rows = 2
            grid_cols = 3
            img_size = 256
            
            # Create canvas
            canvas = np.zeros((grid_rows * img_size * 2, grid_cols * img_size, 3), dtype=np.uint8)
            
            combination_idx = 0
            for i in range(grid_rows):
                for j in range(grid_cols):
                    if combination_idx >= num_combinations:
                        break
                    
                    # Get selfie and glasses
                    selfie_idx = i
                    glasses_idx = j
                    
                    if selfie_idx < len(sample_selfies) and glasses_idx < len(sample_glasses):
                        selfie_data = sample_selfies[selfie_idx]
                        glasses_data = sample_glasses[glasses_idx]
                        
                        # Resize original selfie
                        original_resized = cv2.resize(selfie_data['image'], (img_size, img_size))
                        
                        # Perform try-on
                        result_image = self.perform_single_try_on(selfie_data, glasses_data)
                        
                        if result_image is not None:
                            result_resized = cv2.resize(result_image, (img_size, img_size))
                        else:
                            result_resized = np.zeros((img_size, img_size, 3), dtype=np.uint8)
                        
                        # Place in grid
                        row_start = i * img_size * 2
                        col_start = j * img_size
                        
                        # Original image (top)
                        canvas[row_start:row_start + img_size, col_start:col_start + img_size] = original_resized
                        
                        # Try-on result (bottom)
                        canvas[row_start + img_size:row_start + 2 * img_size, col_start:col_start + img_size] = result_resized
                    
                    combination_idx += 1
            
            # Save comparison grid
            grid_path = self.output_dir / "comparison_grid.jpg"
            if image_processor.save_image(canvas, grid_path):
                logger.info(f"Comparison grid saved to {grid_path}")
                return True
            else:
                logger.error("Failed to save comparison grid")
                return False
                
        except Exception as e:
            logger.error(f"Comparison grid creation failed: {e}")
            return False
    
    def run_interactive_demo(self):
        """Run interactive demo with user input."""
        try:
            logger.info("Starting interactive demo...")
            
            if not self.setup_database_connection():
                return
            
            while True:
                print("\n=== Virtual Glasses Try-On Demo ===")
                print("1. Run batch demo")
                print("2. Create comparison grid")
                print("3. List available selfies")
                print("4. List available glasses")
                print("5. Exit")
                
                choice = input("Enter your choice (1-5): ").strip()
                
                if choice == "1":
                    num_selfies = int(input("Number of selfies (default 3): ") or "3")
                    num_glasses = int(input("Number of glasses (default 3): ") or "3")
                    results = self.run_demo_batch(num_selfies, num_glasses)
                    print(f"Demo results: {results}")
                
                elif choice == "2":
                    if self.create_comparison_grid():
                        print("Comparison grid created successfully!")
                    else:
                        print("Failed to create comparison grid")
                
                elif choice == "3":
                    selfies = self.get_sample_selfies(10)
                    print(f"\nAvailable selfies ({len(selfies)}):")
                    for i, selfie in enumerate(selfies):
                        print(f"{i+1}. {selfie['filename']} (Quality: {selfie['quality_score']:.2f})")
                
                elif choice == "4":
                    glasses = self.get_sample_glasses(10)
                    print(f"\nAvailable glasses ({len(glasses)}):")
                    for i, glass in enumerate(glasses):
                        print(f"{i+1}. {glass['title']} (Type: {glass['glasses_type']}, Transparency: {glass['transparency_ratio']:.2f})")
                
                elif choice == "5":
                    print("Exiting demo...")
                    break
                
                else:
                    print("Invalid choice. Please try again.")
                    
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
        except Exception as e:
            logger.error(f"Interactive demo failed: {e}")

def main():
    """Main demo function."""
    logger.info("Starting Virtual Glasses Try-On Demo")
    
    try:
        demo = VirtualTryOnDemo()
        
        # Check if running in interactive mode
        if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
            demo.run_interactive_demo()
        else:
            # Run default batch demo
            print("Running default batch demo...")
            results = demo.run_demo_batch(num_selfies=2, num_glasses=2)
            print(f"\nDemo Results:")
            print(f"Total combinations: {results['total']}")
            print(f"Successful: {results['success']}")
            print(f"Failed: {results['failed']}")
            print(f"Success rate: {results['success_rate']:.2%}")
            
            # Create comparison grid
            print("\nCreating comparison grid...")
            if demo.create_comparison_grid():
                print("Comparison grid created successfully!")
            
            print(f"\nResults saved to: {demo.output_dir}")
    
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())