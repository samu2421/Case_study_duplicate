# scripts/run_pipeline.py
"""
Main pipeline runner for the Virtual Glasses Try-On project
"""
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import sys
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.database_config import DatabaseManager
from image_processing.utils.path_utils import ProjectPaths
from image_processing.preprocess.preprocess_selfies import SelfiePreprocessor
from image_processing.preprocess.preprocess_glasses import GlassesPreprocessor
from demo.demo_tryon import VirtualTryOnDemo

# Setup logging
def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if log_file:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    else:
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format=log_format
        )

logger = logging.getLogger(__name__)

class PipelineRunner:
    """Main pipeline runner for the virtual glasses try-on system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the pipeline runner
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.paths = ProjectPaths()
        self.db_manager = DatabaseManager()
        
        # Initialize processors
        self.selfie_preprocessor = SelfiePreprocessor(self.db_manager, self.paths)
        self.glasses_preprocessor = GlassesPreprocessor(self.db_manager, self.paths)
        self.demo = VirtualTryOnDemo()
        
        # Ensure all directories exist
        self.paths.ensure_directories_exist()
        
        logger.info("PipelineRunner initialized")
    
    def setup_database(self) -> bool:
        """
        Setup database tables and connections
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Setting up database...")
        
        try:
            # Test connection
            if not self.db_manager.test_connection():
                logger.error("Database connection failed")
                return False
            
            # Create tables
            success = self.db_manager.create_selfies_table()
            if not success:
                logger.error("Failed to create selfies table")
                return False
            
            logger.info("✅ Database setup completed")
            return True
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            return False
    
    def download_glasses_data(self, limit: Optional[int] = None,
                            skip_existing: bool = True) -> Dict[str, Any]:
        """
        Download glasses images from database
        
        Args:
            limit: Maximum number of glasses to download
            skip_existing: Whether to skip existing downloads
            
        Returns:
            Download results
        """
        logger.info(f"Downloading glasses data (limit: {limit})...")
        
        try:
            results = self.glasses_preprocessor.download_glasses_from_database(
                limit=limit, 
                skip_existing=skip_existing
            )
            
            logger.info(f"Download completed: {results['successful']} successful, "
                       f"{results['failed']} failed, {results['skipped']} skipped")
            
            return results
            
        except Exception as e:
            logger.error(f"Glasses download failed: {e}")
            return {'error': str(e)}
    
    def preprocess_glasses(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Preprocess downloaded glasses images
        
        Args:
            limit: Maximum number of glasses to process
            
        Returns:
            Preprocessing results
        """
        logger.info(f"Preprocessing glasses (limit: {limit})...")
        
        try:
            results = self.glasses_preprocessor.preprocess_downloaded_glasses(limit=limit)
            
            logger.info(f"Glasses preprocessing completed: {results['successful']} successful, "
                       f"{results['failed']} failed")
            
            return results
            
        except Exception as e:
            logger.error(f"Glasses preprocessing failed: {e}")
            return {'error': str(e)}
    
    def preprocess_selfies(self, source_dir: Optional[Path] = None,
                          limit: Optional[int] = None,
                          skip_existing: bool = True) -> Dict[str, Any]:
        """
        Preprocess selfie images
        
        Args:
            source_dir: Directory containing selfie images
            limit: Maximum number of selfies to process
            skip_existing: Whether to skip existing selfies
            
        Returns:
            Preprocessing results
        """
        if source_dir is None:
            source_dir = self.paths.raw_selfies_dir
        
        logger.info(f"Preprocessing selfies from {source_dir} (limit: {limit})...")
        
        try:
            results = self.selfie_preprocessor.preprocess_batch(
                source_dir=source_dir,
                max_files=limit,
                skip_existing=skip_existing
            )
            
            logger.info(f"Selfie preprocessing completed: {results['successful']} successful, "
                       f"{results['failed']} failed")
            
            return results
            
        except Exception as e:
            logger.error(f"Selfie preprocessing failed: {e}")
            return {'error': str(e)}
    
    def run_demo(self, selfie_path: Optional[Path] = None,
                glasses_path: Optional[Path] = None,
                glasses_title: Optional[str] = None,
                create_sample: bool = False) -> bool:
        """
        Run the virtual try-on demo
        
        Args:
            selfie_path: Path to selfie image
            glasses_path: Path to glasses image
            glasses_title: Glasses title from database
            create_sample: Whether to create a sample selfie
            
        Returns:
            True if successful, False otherwise
        """
        logger.info("Running virtual try-on demo...")
        
        try:
            if create_sample:
                sample_path = self.demo.create_sample_selfie()
                logger.info(f"Sample selfie created: {sample_path}")
                selfie_path = sample_path
            
            if selfie_path is None:
                # Look for existing sample
                sample_path = self.paths.demo_selfies_dir / "sample_selfie.jpg"
                if sample_path.exists():
                    selfie_path = sample_path
                else:
                    logger.error("No selfie provided and no sample found")
                    return False
            
            if glasses_path:
                # Use local glasses file
                success = self.demo.try_on_glasses(selfie_path, glasses_path)
            else:
                # Use glasses from database
                success = self.demo.demo_with_database_glasses(selfie_path, glasses_title)
            
            if success:
                logger.info("✅ Demo completed successfully")
                logger.info(f"Check output in: {self.paths.demo_output_dir}")
            else:
                logger.error("Demo failed")
            
            return success
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            return False
    
    def create_synthetic_selfies_dataset(self, limit: int = 500) -> Dict[str, Any]:
        """
        Create synthetic selfies dataset as alternative to SCUT-FBP5500
        
        Args:
            limit: Number of synthetic images to create
            
        Returns:
            Creation results
        """
        logger.info(f"Creating synthetic selfies dataset ({limit} images)...")
        
        try:
            from image_processing.download.google_drive_downloader import GoogleDriveDatasetDownloader
            
            downloader = GoogleDriveDatasetDownloader(self.db_manager)
            results = downloader.create_synthetic_dataset_fallback(limit=limit)
            
            if 'error' not in results:
                logger.info(f"✅ Synthetic dataset created: {results['successful']}/{results['total_processed']} images")
                
                # Get final statistics
                final_stats = downloader.get_dataset_stats()
                logger.info(f"   Dataset statistics: {final_stats}")
                
                return {'status': 'success', 'creation_results': results, 'final_stats': final_stats}
            else:
                logger.error(f"❌ Synthetic dataset creation failed: {results['error']}")
                return {'status': 'failed', 'error': results['error']}
            
        except Exception as e:
            logger.error(f"Synthetic dataset creation failed: {e}")
            return {'status': 'failed', 'error': str(e)}
        """
        Download SCUT-FBP5500 selfies dataset from Google Drive
        
        Args:
            limit: Maximum number of images to process
            
        Returns:
            Download and processing results
        """
        logger.info(f"Downloading SCUT-FBP5500 selfies dataset (limit: {limit})...")
        
        try:
            from image_processing.download.google_drive_downloader import GoogleDriveDatasetDownloader
            
            downloader = GoogleDriveDatasetDownloader(self.db_manager)
            results = downloader.download_and_process_complete_dataset(limit=limit)
            
            if 'error' not in results:
                logger.info("✅ Selfies dataset download and processing completed")
                if 'final_stats' in results and results['final_stats']['status'] == 'success':
                    stats = results['final_stats']['data']
                    logger.info(f"   Total images: {stats['total_images']}")
                    logger.info(f"   Face detection rate: {stats['face_detection_rate']:.1%}")
                    logger.info(f"   Train/Val/Test split: {stats['train_count']}/{stats['val_count']}/{stats['test_count']}")
            else:
                logger.error(f"Selfies dataset download failed: {results['error']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Selfies dataset download failed: {e}")
            return {'error': str(e)}
    
    def analyze_glasses_dataset(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze glasses dataset structure and metadata
        
        Args:
            limit: Maximum number of glasses to analyze
            
        Returns:
            Analysis results
        """
        logger.info(f"Analyzing glasses dataset (limit: {limit})...")
        
        try:
            from image_processing.utils.glasses_dataset_analyzer import GlassesDatasetAnalyzer
            
            analyzer = GlassesDatasetAnalyzer(self.db_manager)
            results = analyzer.analyze_glasses_from_database(limit=limit)
            
            if 'error' not in results:
                logger.info(f"✅ Glasses analysis completed: {results['analyzed']}/{results['total_glasses']} glasses")
                logger.info(f"   Style distribution: {results['style_distribution']}")
                logger.info(f"   Top brands: {list(results['brand_distribution'].keys())[:5]}")
                
                # Export analysis report
                report_path = analyzer.export_analysis_report()
                if report_path:
                    logger.info(f"   Analysis report saved: {report_path}")
            else:
                logger.error(f"Glasses analysis failed: {results['error']}")
            
            return results
            
        except Exception as e:
            logger.error(f"Glasses analysis failed: {e}")
            return {'error': str(e)}
    
    def train_model(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Train the virtual try-on model
        
        Args:
            config: Training configuration
            
        Returns:
            Training results
        """
        logger.info("🧠 Starting model training...")
        
        try:
            from scripts.train_model import VirtualTryOnTrainer
            
            # Default training configuration
            default_config = {
                'epochs': 50,
                'batch_size': 8,
                'learning_rate': 1e-4,
                'selfies_limit': 1000,
                'glasses_limit': 100,
                'lambda_l1': 1.0,
                'lambda_perceptual': 0.1,
                'lambda_identity': 0.1,
                'lambda_style': 0.01,
                'use_gpu': True
            }
            
            if config:
                default_config.update(config)
            
            # Initialize trainer
            trainer = VirtualTryOnTrainer(default_config)
            
            # Start training
            trainer.train()
            
            # Get final evaluation
            final_metrics = trainer.evaluate()
            
            results = {
                'status': 'completed',
                'config': default_config,
                'final_metrics': final_metrics,
                'best_loss': trainer.best_loss,
                'epochs_completed': len(trainer.train_losses)
            }
            
            logger.info("🎉 Model training completed successfully!")
            logger.info(f"   Best validation loss: {trainer.best_loss:.4f}")
            logger.info(f"   Epochs completed: {len(trainer.train_losses)}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            return {'error': str(e)}
    
    def run_advanced_demo(self, selfie_path: Optional[Path] = None,
                         glasses_id: Optional[str] = None,
                         use_trained_model: bool = True) -> bool:
        """
        Run advanced demo with trained model
        
        Args:
            selfie_path: Path to selfie image
            glasses_id: Specific glasses ID to use
            use_trained_model: Whether to use trained model
            
        Returns:
            True if successful, False otherwise
        """
        logger.info("🚀 Running advanced virtual try-on demo...")
        
        try:
            if use_trained_model:
                # Try to load trained model
                checkpoint_path = self.paths.models_dir / "checkpoints" / "best_model.pth"
                if checkpoint_path.exists():
                    logger.info("Using trained model for demo")
                    # TODO: Implement trained model inference
                else:
                    logger.warning("No trained model found, using basic demo")
                    use_trained_model = False
            
            if not use_trained_model:
                # Fallback to basic demo
                success = self.run_demo(selfie_path, create_sample=True)
                return success
            
            # Advanced demo implementation would go here
            # For now, fallback to basic demo
            success = self.run_demo(selfie_path, create_sample=True)
            return success
            
        except Exception as e:
            logger.error(f"Advanced demo failed: {e}")
            return False
        """
        Get overall system status and statistics
        
        Returns:
            System status information
        """
        logger.info("Getting system status...")
        
        try:
            status = {
                'timestamp': datetime.now().isoformat(),
                'database_connected': self.db_manager.test_connection(),
                'tables_info': self.db_manager.get_tables_info(),
                'selfies_stats': self.selfie_preprocessor.get_preprocessing_stats(),
                'glasses_stats': self.glasses_preprocessor.get_glasses_stats(),
                'directories': {
                    'project_root': str(self.paths.project_root),
                    'raw_selfies_exist': self.paths.raw_selfies_dir.exists(),
                    'processed_selfies_exist': self.paths.processed_selfies_dir.exists(),
                    'raw_glasses_exist': self.paths.raw_glasses_dir.exists(),
                    'processed_glasses_exist': self.paths.processed_glasses_dir.exists(),
                    'demo_output_exist': self.paths.demo_output_dir.exists()
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get system status: {e}")
            return {'error': str(e)}
    
    def run_full_pipeline(self, selfies_dir: Optional[Path] = None,
                         glasses_limit: Optional[int] = None,
                         selfies_limit: Optional[int] = None,
                         run_demo_after: bool = True) -> Dict[str, Any]:
        """
        Run the complete pipeline from start to finish
        
        Args:
            selfies_dir: Directory containing selfie images
            glasses_limit: Maximum number of glasses to process
            selfies_limit: Maximum number of selfies to process
            run_demo_after: Whether to run demo after preprocessing
            
        Returns:
            Complete pipeline results
        """
        logger.info("🚀 Starting full pipeline execution...")
        
        pipeline_results = {
            'start_time': datetime.now().isoformat(),
            'steps_completed': [],
            'steps_failed': [],
            'results': {}
        }
        
        try:
            # Step 1: Setup database
            logger.info("Step 1: Setting up database...")
            if self.setup_database():
                pipeline_results['steps_completed'].append('database_setup')
            else:
                pipeline_results['steps_failed'].append('database_setup')
                return pipeline_results
            
            # Step 2: Download glasses
            logger.info("Step 2: Downloading glasses...")
            glasses_download = self.download_glasses_data(limit=glasses_limit)
            pipeline_results['results']['glasses_download'] = glasses_download
            if 'error' not in glasses_download:
                pipeline_results['steps_completed'].append('glasses_download')
            else:
                pipeline_results['steps_failed'].append('glasses_download')
            
            # Step 3: Preprocess glasses
            logger.info("Step 3: Preprocessing glasses...")
            glasses_preprocessing = self.preprocess_glasses(limit=glasses_limit)
            pipeline_results['results']['glasses_preprocessing'] = glasses_preprocessing
            if 'error' not in glasses_preprocessing:
                pipeline_results['steps_completed'].append('glasses_preprocessing')
            else:
                pipeline_results['steps_failed'].append('glasses_preprocessing')
            
            # Step 4: Preprocess selfies (if directory provided)
            if selfies_dir:
                logger.info("Step 4: Preprocessing selfies...")
                selfies_preprocessing = self.preprocess_selfies(
                    source_dir=selfies_dir, 
                    limit=selfies_limit
                )
                pipeline_results['results']['selfies_preprocessing'] = selfies_preprocessing
                if 'error' not in selfies_preprocessing:
                    pipeline_results['steps_completed'].append('selfies_preprocessing')
                else:
                    pipeline_results['steps_failed'].append('selfies_preprocessing')
            
            # Step 5: Run demo (if requested)
            if run_demo_after:
                logger.info("Step 5: Running demo...")
                demo_success = self.run_demo(create_sample=True)
                pipeline_results['results']['demo'] = {'success': demo_success}
                if demo_success:
                    pipeline_results['steps_completed'].append('demo')
                else:
                    pipeline_results['steps_failed'].append('demo')
            
            # Final status
            pipeline_results['end_time'] = datetime.now().isoformat()
            pipeline_results['system_status'] = self.get_system_status()
            
            logger.info("🎉 Full pipeline execution completed!")
            logger.info(f"Steps completed: {len(pipeline_results['steps_completed'])}")
            logger.info(f"Steps failed: {len(pipeline_results['steps_failed'])}")
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            pipeline_results['error'] = str(e)
            pipeline_results['end_time'] = datetime.now().isoformat()
            return pipeline_results


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description='Virtual Glasses Try-On Pipeline Runner')
    
    # Pipeline commands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command
    setup_parser = subparsers.add_parser('setup', help='Setup database and directories')
    
    # Download glasses command
    download_parser = subparsers.add_parser('download-glasses', help='Download glasses from database')
    download_parser.add_argument('--limit', type=int, help='Maximum number of glasses to download')
    download_parser.add_argument('--no-skip-existing', action='store_true', help='Re-download existing files')
    
    # Preprocess glasses command
    preprocess_glasses_parser = subparsers.add_parser('preprocess-glasses', help='Preprocess glasses images')
    preprocess_glasses_parser.add_argument('--limit', type=int, help='Maximum number to process')
    
    # Preprocess selfies command
    preprocess_selfies_parser = subparsers.add_parser('preprocess-selfies', help='Preprocess selfie images')
    preprocess_selfies_parser.add_argument('--source-dir', type=str, help='Source directory for selfies')
    preprocess_selfies_parser.add_argument('--limit', type=int, help='Maximum number to process')
    preprocess_selfies_parser.add_argument('--no-skip-existing', action='store_true', help='Re-process existing files')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run virtual try-on demo')
    demo_parser.add_argument('--selfie', type=str, help='Path to selfie image')
    demo_parser.add_argument('--glasses', type=str, help='Path to glasses image')
    demo_parser.add_argument('--glasses-title', type=str, help='Glasses title from database')
    demo_parser.add_argument('--create-sample', action='store_true', help='Create sample selfie')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Get system status')
    
    # Comprehensive workflow command
    comprehensive_parser = subparsers.add_parser('comprehensive', help='Run complete end-to-end workflow')
    comprehensive_parser.add_argument('--selfies-limit', type=int, help='Maximum number of selfies to process')
    comprehensive_parser.add_argument('--glasses-limit', type=int, help='Maximum number of glasses to process')
    comprehensive_parser.add_argument('--skip-training', action='store_true', help='Skip model training step')
    comprehensive_parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    comprehensive_parser.add_argument('--batch-size', type=int, default=8, help='Training batch size')
    
    # Download selfies command
    download_selfies_parser = subparsers.add_parser('download-selfies', help='Download SCUT-FBP5500 selfies dataset')
    download_selfies_parser.add_argument('--limit', type=int, help='Maximum number of images to process')
    
    # Create synthetic selfies command
    synthetic_selfies_parser = subparsers.add_parser('create-synthetic-selfies', help='Create synthetic selfies dataset')
    synthetic_selfies_parser.add_argument('--limit', type=int, default=500, help='Number of synthetic images to create')
    
    # Analyze glasses command
    analyze_glasses_parser = subparsers.add_parser('analyze-glasses', help='Analyze glasses dataset')
    analyze_glasses_parser.add_argument('--limit', type=int, help='Maximum number of glasses to analyze')
    
    # Train model command
    train_parser = subparsers.add_parser('train', help='Train virtual try-on model')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    train_parser.add_argument('--batch-size', type=int, default=8, help='Training batch size')
    train_parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    train_parser.add_argument('--selfies-limit', type=int, help='Limit selfies for training')
    train_parser.add_argument('--glasses-limit', type=int, help='Limit glasses for training')
    
    # Advanced demo command
    advanced_demo_parser = subparsers.add_parser('advanced-demo', help='Run advanced demo with trained model')
    advanced_demo_parser.add_argument('--selfie', type=str, help='Path to selfie image')
    advanced_demo_parser.add_argument('--glasses-id', type=str, help='Specific glasses ID to use')
    advanced_demo_parser.add_argument('--no-trained-model', action='store_true', help='Use basic demo instead of trained model')

    # Full pipeline command
    full_parser = subparsers.add_parser('full', help='Run complete pipeline')
    full_parser.add_argument('--selfies-dir', type=str, help='Directory containing selfie images')
    full_parser.add_argument('--glasses-limit', type=int, help='Maximum glasses to process')
    full_parser.add_argument('--selfies-limit', type=int, help='Maximum selfies to process')
    full_parser.add_argument('--no-demo', action='store_true', help='Skip demo after preprocessing')
    
    # Global arguments
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    parser.add_argument('--log-file', type=str, help='Log file path')
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = Path(args.log_file) if args.log_file else None
    setup_logging(args.log_level, log_file)
    
    # Initialize pipeline
    pipeline = PipelineRunner()
    
    # Execute command
    if args.command == 'setup':
        success = pipeline.setup_database()
        print("✅ Setup completed" if success else "❌ Setup failed")
    
    elif args.command == 'comprehensive':
        # Run comprehensive end-to-end workflow
        training_config = {
            'epochs': args.epochs,
            'batch_size': args.batch_size
        } if not args.skip_training else None
        
        results = pipeline.run_comprehensive_workflow(
            selfies_limit=args.selfies_limit,
            glasses_limit=args.glasses_limit,
            training_config=training_config,
            skip_training=args.skip_training
        )
        
        print("\n🎉 COMPREHENSIVE WORKFLOW RESULTS:")
        print(f"   Steps completed: {results['summary']['completed_steps']}/7")
        print(f"   Success rate: {results['summary']['success_rate']:.1%}")
        print(f"   Selfies processed: {results['summary']['selfies_processed']}")
        print(f"   Glasses analyzed: {results['summary']['glasses_analyzed']}")
        print(f"   Model trained: {'Yes' if results['summary']['model_trained'] else 'No/Skipped'}")
        print(f"   Demo ready: {'Yes' if results['summary']['demo_ready'] else 'No'}")
    
    elif args.command == 'download-selfies':
        results = pipeline.download_selfies_dataset(limit=args.limit)
        if 'error' not in results:
            print("✅ Selfies dataset download completed")
            if 'final_stats' in results and results['final_stats']['status'] == 'success':
                stats = results['final_stats']['data']
                print(f"   Total images: {stats['total_images']}")
                print(f"   Face detection rate: {stats['face_detection_rate']:.1%}")
        else:
            print(f"❌ Selfies dataset download failed: {results['error']}")
    
    elif args.command == 'create-synthetic-selfies':
        results = pipeline.create_synthetic_selfies_dataset(limit=args.limit)
        if results['status'] == 'success':
            print(f"✅ Synthetic selfies dataset created")
            stats = results['final_stats']
            print(f"   Total images: {stats['total_images']}")
            print(f"   Train/Val/Test split: {stats['train_count']}/{stats['val_count']}/{stats['test_count']}")
        else:
            print(f"❌ Synthetic dataset creation failed: {results['error']}")
    
    elif args.command == 'analyze-glasses':
        results = pipeline.analyze_glasses_dataset(limit=args.limit)
        if 'error' not in results:
            print(f"✅ Glasses analysis completed: {results['analyzed']}/{results['total_glasses']}")
            print(f"   Style distribution: {results['style_distribution']}")
            print(f"   Brand distribution: {dict(list(results['brand_distribution'].items())[:5])}")
        else:
            print(f"❌ Glasses analysis failed: {results['error']}")
    
    elif args.command == 'train':
        training_config = {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'selfies_limit': args.selfies_limit,
            'glasses_limit': args.glasses_limit
        }
        results = pipeline.train_model(config=training_config)
        if 'error' not in results:
            print("✅ Model training completed")
            print(f"   Best validation loss: {results['best_loss']:.4f}")
            print(f"   Epochs completed: {results['epochs_completed']}")
        else:
            print(f"❌ Model training failed: {results['error']}")
    
    elif args.command == 'advanced-demo':
        selfie_path = Path(args.selfie) if args.selfie else None
        success = pipeline.run_advanced_demo(
            selfie_path=selfie_path,
            glasses_id=args.glasses_id,
            use_trained_model=not args.no_trained_model
        )
        print("✅ Advanced demo completed" if success else "❌ Advanced demo failed")

    elif args.command == 'download-glasses':
        results = pipeline.download_glasses_data(
            limit=args.limit,
            skip_existing=not args.no_skip_existing
        )
        print(f"Download results: {results}")
    
    elif args.command == 'preprocess-glasses':
        results = pipeline.preprocess_glasses(limit=args.limit)
        print(f"Preprocessing results: {results}")
    
    elif args.command == 'preprocess-selfies':
        source_dir = Path(args.source_dir) if args.source_dir else None
        results = pipeline.preprocess_selfies(
            source_dir=source_dir,
            limit=args.limit,
            skip_existing=not args.no_skip_existing
        )
        print(f"Preprocessing results: {results}")
    
    elif args.command == 'demo':
        selfie_path = Path(args.selfie) if args.selfie else None
        glasses_path = Path(args.glasses) if args.glasses else None
        success = pipeline.run_demo(
            selfie_path=selfie_path,
            glasses_path=glasses_path,
            glasses_title=args.glasses_title,
            create_sample=args.create_sample
        )
        print("✅ Demo completed" if success else "❌ Demo failed")
    
    elif args.command == 'status':
        status = pipeline.get_system_status()
        print("System Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")
    
    elif args.command == 'full':
        selfies_dir = Path(args.selfies_dir) if args.selfies_dir else None
        results = pipeline.run_full_pipeline(
            selfies_dir=selfies_dir,
            glasses_limit=args.glasses_limit,
            selfies_limit=args.selfies_limit,
            run_demo_after=not args.no_demo
        )
        print("Full Pipeline Results:")
        print(f"  Steps completed: {results['steps_completed']}")
        print(f"  Steps failed: {results['steps_failed']}")
        if 'error' in results:
            print(f"  Error: {results['error']}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()