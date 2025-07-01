"""
Main pipeline runner for the Virtual Glasses Try-On system.
Handles the complete workflow from data download to model inference.
"""

import sys
import argparse
from pathlib import Path
import logging
from typing import Dict, Optional
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from database.config import db_manager
from image_processing.download.dataset_downloader import dataset_downloader
from image_processing.preprocess.preprocess_selfies import selfie_preprocessor
from image_processing.preprocess.preprocess_glasses import glasses_preprocessor
from models.hybrid_model import get_hybrid_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class PipelineRunner:
    """Main pipeline runner for the Virtual Glasses Try-On system."""
    
    def __init__(self):
        """Initialize pipeline runner."""
        self.start_time = time.time()
        self.stages_completed = []
        self.results = {}
        
    def setup_database(self) -> bool:
        """Setup database connection and tables."""
        try:
            logger.info("=== Stage 1: Database Setup ===")
            
            # Connect to database
            if not db_manager.connect():
                logger.error("Failed to connect to database")
                return False
            
            # Create tables if they don't exist
            db_manager.create_selfies_table()
            
            logger.info("Database setup completed successfully")
            self.stages_completed.append("database_setup")
            return True
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            return False
    
    def download_and_store_dataset(self, skip_if_exists: bool = True) -> bool:
        """Download SCUT-FBP5500 dataset and store in database."""
        try:
            logger.info("=== Stage 2: Dataset Download and Storage ===")
            
            # Check if dataset already exists
            if skip_if_exists:
                try:
                    existing_count = db_manager.execute_query(
                        f"SELECT COUNT(*) as count FROM {db_manager.config['schema']}.selfies"
                    )
                    if len(existing_count) > 0 and existing_count.iloc[0]['count'] > 0:
                        logger.info(f"Dataset already exists ({existing_count.iloc[0]['count']} images). Skipping download.")
                        self.results['dataset_download'] = {'total': existing_count.iloc[0]['count'], 'success': existing_count.iloc[0]['count'], 'failed': 0}
                        self.stages_completed.append("dataset_download")
                        return True
                except Exception as e:
                    logger.warning(f"Could not check existing dataset: {e}")
            
            # Download and store dataset
            results = dataset_downloader.download_and_store_dataset()
            self.results['dataset_download'] = results
            
            if results['success'] > 0:
                logger.info(f"Dataset download completed: {results}")
                self.stages_completed.append("dataset_download")
                return True
            else:
                logger.error("Dataset download failed")
                return False
                
        except Exception as e:
            logger.error(f"Dataset download and storage failed: {e}")
            return False
    
    def preprocess_selfies(self, reprocess: bool = False) -> bool:
        """Preprocess selfie images."""
        try:
            logger.info("=== Stage 3: Selfie Preprocessing ===")
            
            # Check if already processed
            if not reprocess:
                try:
                    processed_count = db_manager.execute_query(
                        f"SELECT COUNT(*) as count FROM {db_manager.config['schema']}.selfies WHERE face_detected IS NOT NULL"
                    )
                    if len(processed_count) > 0 and processed_count.iloc[0]['count'] > 100:
                        logger.info(f"Selfies already preprocessed ({processed_count.iloc[0]['count']} images). Skipping.")
                        self.results['selfie_preprocessing'] = {'total_processed': processed_count.iloc[0]['count'], 'total_failed': 0}
                        self.stages_completed.append("selfie_preprocessing")
                        return True
                except Exception as e:
                    logger.warning(f"Could not check preprocessing status: {e}")
            
            # Preprocess selfies in batches
            results = selfie_preprocessor.preprocess_batch_from_database(batch_size=32, quality_filter=True)
            self.results['selfie_preprocessing'] = results
            
            if results['total_processed'] > 0:
                logger.info(f"Selfie preprocessing completed: {results}")
                self.stages_completed.append("selfie_preprocessing")
                return True
            else:
                logger.error("Selfie preprocessing failed")
                return False
                
        except Exception as e:
            logger.error(f"Selfie preprocessing failed: {e}")
            return False
    
    def preprocess_glasses(self, limit: Optional[int] = None, reprocess: bool = False) -> bool:
        """Preprocess glasses images."""
        try:
            logger.info("=== Stage 4: Glasses Preprocessing ===")
            
            # Check if already processed
            if not reprocess:
                try:
                    processed_count = db_manager.execute_query(
                        f"SELECT COUNT(*) as count FROM {db_manager.config['schema']}.processed_glasses"
                    )
                    if len(processed_count) > 0 and processed_count.iloc[0]['count'] > 50:
                        logger.info(f"Glasses already preprocessed ({processed_count.iloc[0]['count']} images). Skipping.")
                        self.results['glasses_preprocessing'] = {'total_processed': processed_count.iloc[0]['count'], 'total_failed': 0}
                        self.stages_completed.append("glasses_preprocessing")
                        return True
                except Exception as e:
                    logger.warning(f"Could not check glasses preprocessing status: {e}")
            
            # Preprocess glasses
            results = glasses_preprocessor.preprocess_glasses_from_database(batch_size=16, limit=limit)
            self.results['glasses_preprocessing'] = results
            
            if results['total_processed'] > 0:
                logger.info(f"Glasses preprocessing completed: {results}")
                self.stages_completed.append("glasses_preprocessing")
                return True
            else:
                logger.error("Glasses preprocessing failed")
                return False
                
        except Exception as e:
            logger.error(f"Glasses preprocessing failed: {e}")
            return False
    
    def initialize_models(self) -> bool:
        """Initialize and load the hybrid model."""
        try:
            logger.info("=== Stage 5: Model Initialization ===")
            
            # Load hybrid model (this will download SAM and DINOv2 if needed)
            hybrid_model = get_hybrid_model()
            
            logger.info("Models initialized successfully")
            self.stages_completed.append("model_initialization")
            return True
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            return False
    
    def run_data_analysis(self) -> bool:
        """Run data quality analysis."""
        try:
            logger.info("=== Stage 6: Data Analysis ===")
            
            # Analyze selfies dataset
            selfie_analysis = selfie_preprocessor.analyze_dataset_quality()
            self.results['selfie_analysis'] = selfie_analysis
            
            # Analyze glasses dataset
            glasses_analysis = glasses_preprocessor.analyze_glasses_dataset()
            self.results['glasses_analysis'] = glasses_analysis
            
            # Log analysis results
            logger.info("=== Selfie Dataset Analysis ===")
            if 'overall_statistics' in selfie_analysis:
                stats = selfie_analysis['overall_statistics']
                logger.info(f"Total images: {stats.get('total_images', 0)}")
                logger.info(f"Images with faces: {stats.get('images_with_faces', 0)}")
                logger.info(f"Average quality score: {stats.get('avg_quality_score', 0):.3f}")
            
            logger.info("=== Glasses Dataset Analysis ===")
            if 'overall_statistics' in glasses_analysis:
                stats = glasses_analysis['overall_statistics']
                logger.info(f"Total glasses: {stats.get('total_glasses', 0)}")
                logger.info(f"With transparency: {stats.get('with_transparency', 0)}")
                logger.info(f"Average transparency ratio: {stats.get('avg_transparency_ratio', 0):.3f}")
            
            self.stages_completed.append("data_analysis")
            return True
            
        except Exception as e:
            logger.error(f"Data analysis failed: {e}")
            return False
    
    def run_demo(self, num_selfies: int = 3, num_glasses: int = 3) -> bool:
        """Run demo to test the complete pipeline."""
        try:
            logger.info("=== Stage 7: Demo Execution ===")
            
            # Import demo (avoid circular imports)
            from demo.demo_tryon import VirtualTryOnDemo
            
            demo = VirtualTryOnDemo()
            results = demo.run_demo_batch(num_selfies=num_selfies, num_glasses=num_glasses)
            self.results['demo'] = results
            
            if results['success'] > 0:
                logger.info(f"Demo completed successfully: {results}")
                self.stages_completed.append("demo")
                return True
            else:
                logger.warning(f"Demo had limited success: {results}")
                return False
                
        except Exception as e:
            logger.error(f"Demo execution failed: {e}")
            return False
    
    def generate_report(self) -> Dict:
        """Generate a comprehensive pipeline report."""
        try:
            end_time = time.time()
            total_duration = end_time - self.start_time
            
            report = {
                'pipeline_info': {
                    'start_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.start_time)),
                    'end_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)),
                    'total_duration_seconds': total_duration,
                    'total_duration_formatted': f"{total_duration//3600:.0f}h {(total_duration%3600)//60:.0f}m {total_duration%60:.0f}s"
                },
                'stages_completed': self.stages_completed,
                'stages_total': 7,
                'completion_rate': len(self.stages_completed) / 7,
                'results': self.results
            }
            
            # Log report summary
            logger.info("=== Pipeline Execution Report ===")
            logger.info(f"Duration: {report['pipeline_info']['total_duration_formatted']}")
            logger.info(f"Stages completed: {len(self.stages_completed)}/7 ({report['completion_rate']:.1%})")
            logger.info(f"Completed stages: {', '.join(self.stages_completed)}")
            
            if 'dataset_download' in self.results:
                dr = self.results['dataset_download']
                logger.info(f"Dataset: {dr['success']}/{dr['total']} images stored")
            
            if 'demo' in self.results:
                demo_r = self.results['demo']
                logger.info(f"Demo: {demo_r['success']}/{demo_r['total']} combinations successful")
            
            return report
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return {}
    
    def run_full_pipeline(self, args: argparse.Namespace) -> Dict:
        """Run the complete pipeline."""
        try:
            logger.info("Starting Virtual Glasses Try-On Pipeline")
            logger.info(f"Arguments: {vars(args)}")
            
            # Stage 1: Database Setup
            if not self.setup_database():
                return self.generate_report()
            
            # Stage 2: Dataset Download
            if not args.skip_download:
                if not self.download_and_store_dataset(skip_if_exists=not args.force_download):
                    return self.generate_report()
            else:
                logger.info("Skipping dataset download (--skip-download)")
                self.stages_completed.append("dataset_download")
            
            # Stage 3: Selfie Preprocessing
            if not args.skip_preprocessing:
                if not self.preprocess_selfies(reprocess=args.force_preprocessing):
                    return self.generate_report()
            else:
                logger.info("Skipping selfie preprocessing (--skip-preprocessing)")
                self.stages_completed.append("selfie_preprocessing")
            
            # Stage 4: Glasses Preprocessing
            if not args.skip_preprocessing:
                if not self.preprocess_glasses(limit=args.glasses_limit, reprocess=args.force_preprocessing):
                    return self.generate_report()
            else:
                logger.info("Skipping glasses preprocessing (--skip-preprocessing)")
                self.stages_completed.append("glasses_preprocessing")
            
            # Stage 5: Model Initialization
            if not self.initialize_models():
                return self.generate_report()
            
            # Stage 6: Data Analysis
            if not self.run_data_analysis():
                logger.warning("Data analysis failed, but continuing...")
            
            # Stage 7: Demo
            if not args.skip_demo:
                if not self.run_demo(num_selfies=args.demo_selfies, num_glasses=args.demo_glasses):
                    logger.warning("Demo failed, but pipeline considered successful")
            else:
                logger.info("Skipping demo (--skip-demo)")
                self.stages_completed.append("demo")
            
            logger.info("Pipeline execution completed")
            return self.generate_report()
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return self.generate_report()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Virtual Glasses Try-On Pipeline Runner")
    
    # Dataset options
    parser.add_argument("--skip-download", action="store_true", 
                       help="Skip dataset download stage")
    parser.add_argument("--force-download", action="store_true",
                       help="Force dataset download even if it exists")
    
    # Preprocessing options
    parser.add_argument("--skip-preprocessing", action="store_true",
                       help="Skip preprocessing stages")
    parser.add_argument("--force-preprocessing", action="store_true",
                       help="Force reprocessing even if already done")
    parser.add_argument("--glasses-limit", type=int, default=None,
                       help="Limit number of glasses to process")
    
    # Demo options
    parser.add_argument("--skip-demo", action="store_true",
                       help="Skip demo execution")
    parser.add_argument("--demo-selfies", type=int, default=3,
                       help="Number of selfies for demo")
    parser.add_argument("--demo-glasses", type=int, default=3,
                       help="Number of glasses for demo")
    
    # General options
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    return parser.parse_args()

def main():
    """Main function."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Set logging level
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Create and run pipeline
        runner = PipelineRunner()
        report = runner.run_full_pipeline(args)
        
        # Print final summary
        print("\n" + "="*60)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*60)
        print(f"Duration: {report.get('pipeline_info', {}).get('total_duration_formatted', 'Unknown')}")
        print(f"Completion: {len(report.get('stages_completed', []))}/7 stages")
        print(f"Success Rate: {report.get('completion_rate', 0):.1%}")
        
        if report.get('completion_rate', 0) >= 0.8:
            print("✅ Pipeline completed successfully!")
            return 0
        else:
            print("❌ Pipeline completed with errors")
            return 1
            
    except KeyboardInterrupt:
        print("\n❌ Pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"❌ Pipeline failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())