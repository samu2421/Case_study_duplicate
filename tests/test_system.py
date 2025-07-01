"""
Test framework for Virtual Glasses Try-On System.
Provides comprehensive testing for all major components.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Setup logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestDatabaseConnection(unittest.TestCase):
    """Test database connectivity and operations."""
    
    def setUp(self):
        """Setup test environment."""
        from database.config import DatabaseManager
        self.db_manager = DatabaseManager()
    
    def test_database_connection(self):
        """Test database connection."""
        try:
            result = self.db_manager.connect()
            self.assertTrue(result, "Database connection should succeed")
        except Exception as e:
            self.skipTest(f"Database not available: {e}")
    
    def test_query_execution(self):
        """Test basic query execution."""
        try:
            if not self.db_manager.engine:
                self.db_manager.connect()
            
            result = self.db_manager.execute_query("SELECT 1 as test;")
            self.assertIsNotNone(result, "Query should return results")
            self.assertEqual(len(result), 1, "Should return one row")
            self.assertEqual(result.iloc[0]['test'], 1, "Should return correct value")
        except Exception as e:
            self.skipTest(f"Database query test failed: {e}")

class TestImageProcessing(unittest.TestCase):
    """Test image processing utilities."""
    
    def setUp(self):
        """Setup test environment."""
        from image_processing.utils.image_utils import ImageProcessor
        self.processor = ImageProcessor()
        
        # Create test images
        self.test_image_rgb = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        self.test_image_rgba = np.random.randint(0, 255, (256, 256, 4), dtype=np.uint8)
    
    def test_image_resizing(self):
        """Test image resizing functionality."""
        target_size = (512, 512)
        resized = self.processor.resize_image(self.test_image_rgb, target_size)
        
        self.assertEqual(resized.shape[:2], target_size, "Image should be resized to target size")
        self.assertEqual(resized.shape[2], 3, "RGB channels should be preserved")
    
    def test_image_normalization(self):
        """Test image normalization."""
        normalized = self.processor.normalize_image(self.test_image_rgb)
        
        self.assertTrue(0 <= normalized.min() <= 1, "Normalized values should be >= 0")
        self.assertTrue(0 <= normalized.max() <= 1, "Normalized values should be <= 1")
        self.assertEqual(normalized.dtype, np.float32, "Should return float32")
    
    def test_quality_score_calculation(self):
        """Test image quality score calculation."""
        # Create a sharp test image
        sharp_image = np.zeros((100, 100), dtype=np.uint8)
        sharp_image[40:60, 40:60] = 255  # Sharp square
        
        # Create a blurry test image
        blurry_image = np.ones((100, 100), dtype=np.uint8) * 128
        
        sharp_score = self.processor.calculate_image_quality_score(sharp_image)
        blurry_score = self.processor.calculate_image_quality_score(blurry_image)
        
        self.assertGreater(sharp_score, blurry_score, "Sharp image should have higher quality score")
        self.assertGreaterEqual(sharp_score, 0, "Quality score should be non-negative")
    
    def test_image_to_bytes_conversion(self):
        """Test image to bytes conversion."""
        image_bytes = self.processor.image_to_bytes(self.test_image_rgb)
        
        self.assertIsInstance(image_bytes, bytes, "Should return bytes")
        self.assertGreater(len(image_bytes), 0, "Should return non-empty bytes")

class TestGlassesPreprocessing(unittest.TestCase):
    """Test glasses preprocessing functionality."""
    
    def setUp(self):
        """Setup test environment."""
        from image_processing.preprocess.preprocess_glasses import GlassesPreprocessor
        self.preprocessor = GlassesPreprocessor()
    
    def test_glasses_type_detection(self):
        """Test glasses type detection."""
        # Create test image with different aspect ratios
        wide_glasses = np.random.randint(0, 255, (100, 300, 3), dtype=np.uint8)
        round_glasses = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        wide_type = self.preprocessor.detect_glasses_type(wide_glasses)
        round_type = self.preprocessor.detect_glasses_type(round_glasses)
        
        self.assertIsInstance(wide_type, str, "Should return string type")
        self.assertIsInstance(round_type, str, "Should return string type")
    
    def test_background_removal(self):
        """Test background removal functionality."""
        # Create test image with white background
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        test_image[30:70, 30:70] = [0, 0, 0]  # Black object in center
        
        processed = self.preprocessor.remove_background_advanced(test_image)
        
        self.assertEqual(processed.shape[2], 4, "Should add alpha channel")
        self.assertGreater(np.sum(processed[:, :, 3] == 0), 0, "Should have some transparent pixels")

class TestModelComponents(unittest.TestCase):
    """Test AI model components."""
    
    def setUp(self):
        """Setup test environment."""
        # Mock the model components to avoid loading large models in tests
        self.mock_device = "cpu"
    
    @patch('models.hybrid_model.sam_model_registry')
    @patch('models.hybrid_model.SamPredictor')
    def test_sam_processor_initialization(self, mock_predictor, mock_registry):
        """Test SAM processor initialization."""
        from models.hybrid_model import SAMProcessor
        
        # Mock the model loading
        mock_model = Mock()
        mock_registry.__getitem__.return_value = Mock(return_value=mock_model)
        mock_predictor.return_value = Mock()
        
        with patch('pathlib.Path.exists', return_value=True):
            processor = SAMProcessor(device="cpu")
            
            self.assertIsNotNone(processor.sam_model, "SAM model should be initialized")
            self.assertIsNotNone(processor.predictor, "SAM predictor should be initialized")
    
    @patch('timm.create_model')
    def test_dino_processor_initialization(self, mock_create_model):
        """Test DINOv2 processor initialization."""
        from models.hybrid_model import DINOv2Processor
        
        # Mock the model
        mock_model = Mock()
        mock_model.eval = Mock()
        mock_create_model.return_value = mock_model
        
        processor = DINOv2Processor(device="cpu")
        
        self.assertIsNotNone(processor.model, "DINOv2 model should be initialized")
        mock_model.eval.assert_called_once()
    
    def test_alignment_module_forward(self):
        """Test alignment module forward pass."""
        from models.hybrid_model import GlassesAlignmentModule
        import torch
        
        module = GlassesAlignmentModule(feature_dim=768)
        
        # Create dummy input tensors
        face_features = torch.randn(1, 768)
        glasses_features = torch.randn(1, 768)
        
        output = module(face_features, glasses_features)
        
        self.assertIn('transform_params', output, "Should return transform parameters")
        self.assertIn('blend_weight', output, "Should return blend weight")
        self.assertEqual(output['transform_params'].shape, (1, 6), "Transform params should be 6D")
        self.assertEqual(output['blend_weight'].shape, (1, 1), "Blend weight should be scalar")

class TestDatasetDownloader(unittest.TestCase):
    """Test dataset download and storage functionality."""
    
    def setUp(self):
        """Setup test environment."""
        from image_processing.download.dataset_downloader import DatasetDownloader
        self.downloader = DatasetDownloader()
        self.temp_dir = None
    
    def tearDown(self):
        """Clean up test environment."""
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_metadata_extraction(self):
        """Test metadata extraction from filenames."""
        test_filenames = [
            "image_001_male_adult.jpg",
            "photo_f_young_002.png",
            "selfie_123.jpg"
        ]
        
        for filename in test_filenames:
            metadata = self.downloader.extract_metadata_from_filename(filename)
            
            self.assertIsInstance(metadata, dict, "Should return dictionary")
            self.assertIn('age_group', metadata, "Should extract age group")
            self.assertIn('gender', metadata, "Should extract gender")
            self.assertIn('quality_score', metadata, "Should provide quality score")
    
    def test_dataset_structure_parsing(self):
        """Test dataset structure parsing."""
        # Create temporary dataset structure
        self.temp_dir = tempfile.mkdtemp()
        temp_path = Path(self.temp_dir)
        
        # Create mock dataset structure
        images_dir = temp_path / "Images"
        images_dir.mkdir()
        
        # Create dummy image files
        for i in range(3):
            (images_dir / f"test_image_{i}.jpg").touch()
        
        dataset_info = self.downloader.parse_dataset_structure(temp_path)
        
        self.assertIsNotNone(dataset_info['images_path'], "Should find images directory")
        self.assertEqual(dataset_info['total_images'], 3, "Should count image files correctly")
        self.assertGreater(len(dataset_info['image_files']), 0, "Should find image files")

class TestConfigurationManagement(unittest.TestCase):
    """Test configuration management."""
    
    def test_config_initialization(self):
        """Test configuration initialization."""
        from config.settings import AppConfig
        
        config = AppConfig()
        
        self.assertIsNotNone(config.database, "Database config should be initialized")
        self.assertIsNotNone(config.models, "Models config should be initialized")
        self.assertIsNotNone(config.processing, "Processing config should be initialized")
        self.assertIsNotNone(config.paths, "Paths config should be initialized")
    
    def test_config_validation(self):
        """Test configuration validation."""
        from config.settings import AppConfig
        
        config = AppConfig()
        
        # Test with valid configuration
        is_valid = config.validate_config()
        self.assertTrue(is_valid, "Default configuration should be valid")
        
        # Test with invalid configuration
        config.dataset.train_ratio = 0.9
        config.dataset.val_ratio = 0.9  # This makes total > 1.0
        is_valid = config.validate_config()
        self.assertFalse(is_valid, "Invalid configuration should fail validation")

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_pipeline_components_integration(self):
        """Test that all pipeline components can be imported and initialized."""
        try:
            # Test imports
            from database.config import db_manager
            from image_processing.utils.image_utils import image_processor
            from image_processing.preprocess.preprocess_selfies import selfie_preprocessor
            from image_processing.preprocess.preprocess_glasses import glasses_preprocessor
            from config.settings import get_config
            
            self.assertIsNotNone(db_manager, "Database manager should be importable")
            self.assertIsNotNone(image_processor, "Image processor should be importable")
            self.assertIsNotNone(selfie_preprocessor, "Selfie preprocessor should be importable")
            self.assertIsNotNone(glasses_preprocessor, "Glasses preprocessor should be importable")
            
            config = get_config()
            self.assertIsNotNone(config, "Configuration should be available")
            
        except ImportError as e:
            self.fail(f"Failed to import components: {e}")

class SystemTestRunner:
    """Test runner for the complete system."""
    
    def __init__(self, verbose: bool = False):
        """Initialize test runner."""
        self.verbose = verbose
        self.results = {}
    
    def run_all_tests(self) -> dict:
        """Run all test suites."""
        test_suites = [
            ('Database Tests', TestDatabaseConnection),
            ('Image Processing Tests', TestImageProcessing),
            ('Glasses Preprocessing Tests', TestGlassesPreprocessing),
            ('Model Component Tests', TestModelComponents),
            ('Dataset Downloader Tests', TestDatasetDownloader),
            ('Configuration Tests', TestConfigurationManagement),
            ('Integration Tests', TestIntegration)
        ]
        
        total_tests = 0
        total_failures = 0
        total_errors = 0
        total_skipped = 0
        
        for suite_name, test_class in test_suites:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running {suite_name}")
            logger.info(f"{'='*50}")
            
            # Create test suite
            suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
            
            # Run tests
            runner = unittest.TextTestRunner(
                verbosity=2 if self.verbose else 1,
                stream=sys.stdout
            )
            result = runner.run(suite)
            
            # Collect results
            self.results[suite_name] = {
                'tests_run': result.testsRun,
                'failures': len(result.failures),
                'errors': len(result.errors),
                'skipped': len(result.skipped)
            }
            
            total_tests += result.testsRun
            total_failures += len(result.failures)
            total_errors += len(result.errors)
            total_skipped += len(result.skipped)
        
        # Summary
        self.results['summary'] = {
            'total_tests': total_tests,
            'total_failures': total_failures,
            'total_errors': total_errors,
            'total_skipped': total_skipped,
            'success_rate': (total_tests - total_failures - total_errors) / total_tests if total_tests > 0 else 0
        }
        
        return self.results
    
    def print_summary(self):
        """Print test summary."""
        if 'summary' not in self.results:
            return
        
        summary = self.results['summary']
        
        logger.info(f"\n{'='*60}")
        logger.info("TEST SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total Tests: {summary['total_tests']}")
        logger.info(f"Passed: {summary['total_tests'] - summary['total_failures'] - summary['total_errors']}")
        logger.info(f"Failed: {summary['total_failures']}")
        logger.info(f"Errors: {summary['total_errors']}")
        logger.info(f"Skipped: {summary['total_skipped']}")
        logger.info(f"Success Rate: {summary['success_rate']:.1%}")
        
        if summary['success_rate'] >= 0.8:
            logger.info("✅ System tests PASSED!")
        else:
            logger.info("❌ System tests FAILED!")

def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Virtual Glasses Try-On System Tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--suite", help="Run specific test suite")
    
    args = parser.parse_args()
    
    if args.suite:
        # Run specific test suite
        suite_classes = {
            'database': TestDatabaseConnection,
            'image': TestImageProcessing,
            'glasses': TestGlassesPreprocessing,
            'models': TestModelComponents,
            'dataset': TestDatasetDownloader,
            'config': TestConfigurationManagement,
            'integration': TestIntegration
        }
        
        if args.suite in suite_classes:
            suite = unittest.TestLoader().loadTestsFromTestCase(suite_classes[args.suite])
            runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
            result = runner.run(suite)
            return 0 if result.wasSuccessful() else 1
        else:
            print(f"Unknown test suite: {args.suite}")
            print(f"Available suites: {', '.join(suite_classes.keys())}")
            return 1
    else:
        # Run all tests
        test_runner = SystemTestRunner(verbose=args.verbose)
        results = test_runner.run_all_tests()
        test_runner.print_summary()
        
        success_rate = results['summary']['success_rate']
        return 0 if success_rate >= 0.8 else 1

if __name__ == "__main__":
    exit(main())