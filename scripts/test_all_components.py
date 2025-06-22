"""
Comprehensive testing script for the Virtual Glasses Try-On system
Tests all components systematically to ensure everything works
"""
import sys
import traceback
from pathlib import Path
import subprocess
import importlib

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def print_header(text):
    """Print a formatted header"""
    print("\n" + "="*60)
    print(f"🔍 {text}")
    print("="*60)

def print_test(test_name, status, details=""):
    """Print test result"""
    icon = "✅" if status else "❌"
    print(f"{icon} {test_name}")
    if details:
        print(f"   {details}")

def test_environment():
    """Test Python environment and dependencies"""
    print_header("PHASE 1: Environment Testing")
    
    results = {}
    
    # Test Python version
    try:
        version = sys.version
        major, minor = sys.version_info[:2]
        if major == 3 and minor >= 8:
            print_test("Python Version", True, f"Python {major}.{minor}")
            results['python'] = True
        else:
            print_test("Python Version", False, f"Python {major}.{minor} (need 3.8+)")
            results['python'] = False
    except Exception as e:
        print_test("Python Version", False, str(e))
        results['python'] = False
    
    # Test core dependencies
    dependencies = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'), 
        ('cv2', 'OpenCV'),
        ('PIL', 'Pillow'),
        ('torch', 'PyTorch'),
        ('requests', 'Requests'),
        ('sqlalchemy', 'SQLAlchemy'),
        ('psycopg2', 'PostgreSQL Driver')
    ]
    
    for module, name in dependencies:
        try:
            imported = importlib.import_module(module)
            version = getattr(imported, '__version__', 'Unknown')
            print_test(f"{name} Import", True, f"v{version}")
            results[module] = True
        except ImportError as e:
            print_test(f"{name} Import", False, str(e))
            results[module] = False
    
    return results

def test_project_structure():
    """Test project structure and file existence"""
    print_header("PHASE 2: Project Structure Testing")
    
    results = {}
    
    # Check project root
    project_root = Path(__file__).parent.parent
    print_test("Project Root", project_root.exists(), str(project_root))
    results['structure'] = project_root.exists()
    
    # Check key directories
    directories = [
        'config', 'image_processing', 'models', 'scripts', 
        'demo', 'data', 'notebooks', 'logs'
    ]
    
    for dir_name in directories:
        dir_path = project_root / dir_name
        exists = dir_path.exists()
        print_test(f"Directory: {dir_name}", exists)
        results[f'dir_{dir_name}'] = exists
    
    # Check key Python files
    files = [
        'config/database_config.py',
        'image_processing/utils/path_utils.py',
        'image_processing/utils/image_utils.py',
        'models/hybrid_model.py',
        'scripts/run_pipeline.py',
        'demo/demo_tryon.py'
    ]
    
    for file_path in files:
        full_path = project_root / file_path
        exists = full_path.exists()
        print_test(f"File: {file_path}", exists)
        results[f'file_{file_path.replace("/", "_")}'] = exists
    
    return results

def test_core_components():
    """Test core system components"""
    print_header("PHASE 3: Core Components Testing")
    
    results = {}
    
    # Test path utilities
    try:
        from image_processing.utils.path_utils import ProjectPaths
        paths = ProjectPaths()
        paths.ensure_directories_exist()
        print_test("Path Utilities", True, f"Root: {paths.project_root.name}")
        results['paths'] = True
    except Exception as e:
        print_test("Path Utilities", False, str(e))
        results['paths'] = False
    
    # Test image processing
    try:
        from image_processing.utils.image_utils import ImageProcessor
        import numpy as np
        
        processor = ImageProcessor()
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        resized = processor.resize_image(test_image, (50, 50))
        
        if resized.shape == (50, 50, 3):
            print_test("Image Processing", True, "Resize and basic ops working")
            results['image_processing'] = True
        else:
            print_test("Image Processing", False, f"Unexpected shape: {resized.shape}")
            results['image_processing'] = False
            
    except Exception as e:
        print_test("Image Processing", False, str(e))
        results['image_processing'] = False
    
    # Test database connection
    try:
        from config.database_config import DatabaseManager
        
        db = DatabaseManager()
        connected = db.test_connection()
        
        if connected:
            tables_info = db.get_tables_info()
            table_count = len(tables_info.get('tables', []))
            print_test("Database Connection", True, f"{table_count} tables found")
            results['database'] = True
        else:
            print_test("Database Connection", False, "Cannot connect")
            results['database'] = False
            
    except Exception as e:
        print_test("Database Connection", False, str(e))
        results['database'] = False
    
    return results

def test_preprocessing():
    """Test preprocessing components"""
    print_header("PHASE 4: Preprocessing Testing")
    
    results = {}
    
    # Test selfie preprocessor
    try:
        from image_processing.preprocess.preprocess_selfies import SelfiePreprocessor
        
        preprocessor = SelfiePreprocessor()
        stats = preprocessor.get_preprocessing_stats()
        
        print_test("Selfie Preprocessor", True, f"Stats: {type(stats)}")
        results['selfie_preprocessing'] = True
        
    except Exception as e:
        print_test("Selfie Preprocessor", False, str(e))
        results['selfie_preprocessing'] = False
    
    # Test glasses preprocessor
    try:
        from image_processing.preprocess.preprocess_glasses import GlassesPreprocessor
        
        preprocessor = GlassesPreprocessor()
        stats = preprocessor.get_glasses_stats()
        
        print_test("Glasses Preprocessor", True, f"Stats retrieved")
        results['glasses_preprocessing'] = True
        
    except Exception as e:
        print_test("Glasses Preprocessor", False, str(e))
        results['glasses_preprocessing'] = False
    
    return results

def test_model_architecture():
    """Test model architecture"""
    print_header("PHASE 5: Model Architecture Testing")
    
    results = {}
    
    # Test hybrid model
    try:
        from models.hybrid_model import HybridVirtualTryOnModel
        import numpy as np
        
        model = HybridVirtualTryOnModel()
        
        # Test with dummy data
        test_selfie = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        test_glasses = np.random.randint(0, 255, (100, 150, 4), dtype=np.uint8)
        
        forward_results = model.forward(test_selfie, test_glasses)
        
        if 'result' in forward_results:
            print_test("Hybrid Model", True, f"Forward pass successful")
            results['hybrid_model'] = True
        else:
            print_test("Hybrid Model", False, "No result in forward pass")
            results['hybrid_model'] = False
            
    except Exception as e:
        print_test("Hybrid Model", False, str(e))
        results['hybrid_model'] = False
    
    return results

def test_demo_system():
    """Test demo system"""
    print_header("PHASE 6: Demo System Testing")
    
    results = {}
    
    # Test demo initialization
    try:
        from demo.demo_tryon import VirtualTryOnDemo
        
        demo = VirtualTryOnDemo()
        print_test("Demo Initialization", True, "Demo system ready")
        results['demo_init'] = True
        
        # Test sample selfie creation
        try:
            sample_path = demo.create_sample_selfie()
            if sample_path.exists():
                size_kb = sample_path.stat().st_size // 1024
                print_test("Sample Selfie Creation", True, f"{size_kb} KB created")
                results['sample_selfie'] = True
            else:
                print_test("Sample Selfie Creation", False, "File not created")
                results['sample_selfie'] = False
                
        except Exception as e:
            print_test("Sample Selfie Creation", False, str(e))
            results['sample_selfie'] = False
        
        # Test face detection
        try:
            if results.get('sample_selfie', False):
                face_detected = demo.debug_face_detection(sample_path)
                print_test("Face Detection Test", face_detected, "Debug image created")
                results['face_detection'] = face_detected
            else:
                print_test("Face Detection Test", False, "No sample selfie to test")
                results['face_detection'] = False
                
        except Exception as e:
            print_test("Face Detection Test", False, str(e))
            results['face_detection'] = False
            
    except Exception as e:
        print_test("Demo Initialization", False, str(e))
        results['demo_init'] = False
    
    return results

def test_pipeline_runner():
    """Test pipeline runner"""
    print_header("PHASE 7: Pipeline Runner Testing")
    
    results = {}
    
    try:
        # Test pipeline runner import
        from scripts.run_pipeline import PipelineRunner
        
        runner = PipelineRunner()
        print_test("Pipeline Runner Init", True, "Runner initialized")
        results['pipeline_init'] = True
        
        # Test system status
        try:
            status = runner.get_system_status()
            if isinstance(status, dict) and 'timestamp' in status:
                print_test("System Status", True, f"Status retrieved")
                results['system_status'] = True
            else:
                print_test("System Status", False, "Invalid status format")
                results['system_status'] = False
                
        except Exception as e:
            print_test("System Status", False, str(e))
            results['system_status'] = False
            
    except Exception as e:
        print_test("Pipeline Runner Init", False, str(e))
        results['pipeline_init'] = False
    
    return results

def run_comprehensive_test():
    """Run all tests and provide summary"""
    print_header("🚀 COMPREHENSIVE SYSTEM TEST STARTING")
    
    all_results = {}
    
    # Run all test phases
    test_phases = [
        ("Environment", test_environment),
        ("Project Structure", test_project_structure),
        ("Core Components", test_core_components),
        ("Preprocessing", test_preprocessing),
        ("Model Architecture", test_model_architecture),
        ("Demo System", test_demo_system),
        ("Pipeline Runner", test_pipeline_runner)
    ]
    
    for phase_name, test_func in test_phases:
        try:
            results = test_func()
            all_results[phase_name] = results
        except Exception as e:
            print(f"❌ PHASE FAILED: {phase_name}")
            print(f"   Error: {e}")
            traceback.print_exc()
            all_results[phase_name] = {'error': str(e)}
    
    # Print comprehensive summary
    print_header("📊 COMPREHENSIVE TEST SUMMARY")
    
    total_tests = 0
    passed_tests = 0
    
    for phase_name, results in all_results.items():
        if 'error' in results:
            print(f"❌ {phase_name}: PHASE FAILED")
            continue
            
        phase_passed = sum(1 for v in results.values() if v is True)
        phase_total = len(results)
        total_tests += phase_total
        passed_tests += phase_passed
        
        percentage = (phase_passed / phase_total * 100) if phase_total > 0 else 0
        icon = "✅" if percentage > 80 else "⚠️" if percentage > 50 else "❌"
        
        print(f"{icon} {phase_name}: {phase_passed}/{phase_total} ({percentage:.1f}%)")
    
    # Overall summary
    overall_percentage = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    overall_icon = "🎉" if overall_percentage > 90 else "✅" if overall_percentage > 70 else "⚠️"
    
    print("\n" + "="*60)
    print(f"{overall_icon} OVERALL SYSTEM STATUS: {passed_tests}/{total_tests} ({overall_percentage:.1f}%)")
    print("="*60)
    
    # Recommendations
    if overall_percentage > 90:
        print("🎉 EXCELLENT! Your system is fully ready for development and experimentation.")
    elif overall_percentage > 70:
        print("✅ GOOD! Most components working. Check failed tests and proceed with caution.")
    else:
        print("⚠️ ISSUES DETECTED! Please fix failed components before proceeding.")
    
    # Next steps
    print("\n📋 NEXT STEPS:")
    if overall_percentage > 80:
        print("   1. Run: python scripts/run_pipeline.py download-glasses --limit 3")
        print("   2. Run: python scripts/run_pipeline.py preprocess-glasses --limit 3")
        print("   3. Run: python scripts/run_pipeline.py demo")
        print("   4. Open: jupyter notebook notebooks/experiments.ipynb")
    else:
        print("   1. Fix failed components shown above")
        print("   2. Check .env configuration")
        print("   3. Verify all Python files are created correctly")
        print("   4. Re-run this test script")
    
    return all_results

if __name__ == "__main__":
    try:
        run_comprehensive_test()
    except KeyboardInterrupt:
        print("\n\n⚠️ Testing interrupted by user")
    except Exception as e:
        print(f"\n\n❌ CRITICAL ERROR: {e}")
        traceback.print_exc()