# scripts/verify_setup.py
"""
Comprehensive setup verification script for the Virtual Glasses Try-On project
Verifies all components are working before starting the main workflow
"""
import sys
import traceback
from pathlib import Path
import importlib
import subprocess
import json
from typing import Dict, List, Tuple

def print_header(text: str, level: int = 1):
    """Print formatted header"""
    if level == 1:
        print("\n" + "="*70)
        print(f"🔍 {text}")
        print("="*70)
    else:
        print(f"\n{'='*50}")
        print(f"📋 {text}")
        print("="*50)

def print_result(test_name: str, passed: bool, details: str = ""):
    """Print test result with consistent formatting"""
    icon = "✅" if passed else "❌"
    print(f"{icon} {test_name}")
    if details:
        for line in details.split('\n'):
            if line.strip():
                print(f"   {line}")

def test_python_environment() -> Tuple[bool, str]:
    """Test Python environment and version"""
    try:
        major, minor = sys.version_info[:2]
        if major == 3 and minor >= 8:
            return True, f"Python {major}.{minor}.{sys.version_info.micro}"
        else:
            return False, f"Python {major}.{minor} (requires Python 3.8+)"
    except Exception as e:
        return False, str(e)

def test_dependencies() -> Dict[str, Tuple[bool, str]]:
    """Test all required dependencies"""
    dependencies = [
        ('numpy', 'NumPy - Numerical computing'),
        ('pandas', 'Pandas - Data manipulation'),
        ('cv2', 'OpenCV - Computer vision'),
        ('PIL', 'Pillow - Image processing'),
        ('torch', 'PyTorch - Deep learning'),
        ('torchvision', 'TorchVision - Computer vision'),
        ('requests', 'Requests - HTTP library'),
        ('sqlalchemy', 'SQLAlchemy - Database ORM'),
        ('psycopg2', 'psycopg2 - PostgreSQL adapter'),
        ('matplotlib', 'Matplotlib - Plotting'),
        ('tqdm', 'tqdm - Progress bars'),
        ('dotenv', 'python-dotenv - Environment variables')
    ]
    
    results = {}
    for module, description in dependencies:
        try:
            imported = importlib.import_module(module)
            version = getattr(imported, '__version__', 'Unknown')
            results[description] = (True, f"v{version}")
        except ImportError as e:
            results[description] = (False, f"Not installed: {e}")
    
    return results

def test_project_structure() -> Dict[str, bool]:
    """Test project directory structure"""
    project_root = Path(__file__).parent.parent
    
    required_files = [
        'config/database_config.py',
        'image_processing/utils/path_utils.py',
        'image_processing/utils/image_utils.py',
        'image_processing/utils/glasses_dataset_analyzer.py',
        'image_processing/download/google_drive_downloader.py',
        'image_processing/preprocess/preprocess_selfies.py',
        'image_processing/preprocess/preprocess_glasses.py',
        'models/hybrid_model.py',
        'models/data_loaders.py',
        'scripts/run_pipeline.py',
        'scripts/train_model.py',
        'demo/demo_tryon.py',
        'requirements.txt',
        '.env.template'
    ]
    
    results = {}
    for file_path in required_files:
        full_path = project_root / file_path
        results[file_path] = full_path.exists()
    
    return results

def test_database_connection() -> Tuple[bool, str]:
    """Test database connection"""
    try:
        # Add project root to path
        sys.path.append(str(Path(__file__).parent.parent))
        
        from config.database_config import DatabaseManager
        
        db_manager = DatabaseManager()
        connected = db_manager.test_connection()
        
        if connected:
            tables_info = db_manager.get_tables_info()
            tables = tables_info.get('tables', [])
            return True, f"Connected successfully. Tables: {tables}"
        else:
            return False, "Connection failed - check .env credentials"
            
    except Exception as e:
        return False, f"Error: {str(e)}"

def test_core_imports() -> Dict[str, Tuple[bool, str]]:
    """Test core project imports"""
    # Add project root to path
    sys.path.append(str(Path(__file__).parent.parent))
    
    imports_to_test = [
        ('config.database_config', 'DatabaseManager'),
        ('image_processing.utils.path_utils', 'ProjectPaths'),
        ('image_processing.utils.image_utils', 'ImageProcessor'),
        ('image_processing.utils.glasses_dataset_analyzer', 'GlassesDatasetAnalyzer'),
        ('image_processing.download.google_drive_downloader', 'GoogleDriveDatasetDownloader'),
        ('models.hybrid_model', 'HybridVirtualTryOnModel'),
        ('models.data_loaders', 'SelfiesDataset'),
        ('demo.demo_tryon', 'VirtualTryOnDemo')
    ]
    
    results = {}
    for module_path, class_name in imports_to_test:
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            results[f"{module_path}.{class_name}"] = (True, "Import successful")
        except Exception as e:
            results[f"{module_path}.{class_name}"] = (False, str(e))
    
    return results

def test_pytorch_functionality() -> Tuple[bool, str]:
    """Test PyTorch functionality"""
    try:
        import torch
        
        # Test basic tensor operations
        x = torch.randn(2, 3)
        y = torch.randn(3, 2)
        z = torch.mm(x, y)
        
        # Test device availability
        device_info = []
        device_info.append(f"CPU available: True")
        
        if torch.cuda.is_available():
            device_info.append(f"CUDA available: True")
            device_info.append(f"CUDA devices: {torch.cuda.device_count()}")
            device_info.append(f"Current device: {torch.cuda.get_device_name()}")
        else:
            device_info.append(f"CUDA available: False")
        
        # Test model creation
        from models.hybrid_model import HybridVirtualTryOnModel
        model = HybridVirtualTryOnModel()
        
        return True, "\n".join(device_info + ["Model creation: Success"])
        
    except Exception as e:
        return False, str(e)

def test_data_processing() -> Tuple[bool, str]:
    """Test data processing capabilities"""
    try:
        import numpy as np
        import cv2
        
        # Test image processing
        from image_processing.utils.image_utils import ImageProcessor
        
        processor = ImageProcessor()
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        
        # Test basic operations
        resized = processor.resize_image(dummy_image, (256, 256))
        enhanced = processor.enhance_image(resized)
        faces = processor.detect_faces(dummy_image)
        
        return True, f"Image processing: OK, Face detection: {len(faces)} faces"
        
    except Exception as e:
        return False, str(e)

def check_env_file() -> Tuple[bool, str]:
    """Check .env file configuration"""
    try:
        project_root = Path(__file__).parent.parent
        env_file = project_root / '.env'
        env_template = project_root / '.env.template'
        
        if not env_template.exists():
            return False, ".env.template not found"
        
        if not env_file.exists():
            return False, ".env file not found - copy from .env.template"
        
        # Check required variables
        from dotenv import load_dotenv
        import os
        
        load_dotenv(env_file)
        
        required_vars = ['POSTGRES_HOST', 'POSTGRES_USER', 'POSTGRES_PASSWORD', 'POSTGRES_DB']
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            return False, f"Missing variables: {missing_vars}"
        
        return True, "All required environment variables found"
        
    except Exception as e:
        return False, str(e)

def check_disk_space() -> Tuple[bool, str]:
    """Check available disk space"""
    try:
        import shutil
        
        project_root = Path(__file__).parent.parent
        total, used, free = shutil.disk_usage(project_root)
        
        free_gb = free // (1024**3)
        
        if free_gb < 5:
            return False, f"Only {free_gb}GB free (need at least 5GB)"
        else:
            return True, f"{free_gb}GB available"
            
    except Exception as e:
        return False, str(e)

def generate_setup_report() -> Dict:
    """Generate comprehensive setup report"""
    print_header("VIRTUAL GLASSES TRY-ON SETUP VERIFICATION")
    
    report = {
        'timestamp': str(Path(__file__).parent.parent),
        'tests': {},
        'summary': {}
    }
    
    # Test 1: Python Environment
    print_header("Python Environment", 2)
    passed, details = test_python_environment()
    print_result("Python Version", passed, details)
    report['tests']['python_version'] = {'passed': passed, 'details': details}
    
    # Test 2: Dependencies
    print_header("Python Dependencies", 2)
    deps_results = test_dependencies()
    deps_passed = 0
    for desc, (passed, details) in deps_results.items():
        print_result(desc, passed, details)
        if passed:
            deps_passed += 1
        report['tests'][f'dependency_{desc}'] = {'passed': passed, 'details': details}
    
    # Test 3: Project Structure
    print_header("Project Structure", 2)
    structure_results = test_project_structure()
    structure_passed = sum(structure_results.values())
    structure_total = len(structure_results)
    
    for file_path, exists in structure_results.items():
        print_result(f"File: {file_path}", exists)
        report['tests'][f'file_{file_path}'] = {'passed': exists, 'details': 'File exists' if exists else 'File missing'}
    
    print_result(f"Project Structure", structure_passed == structure_total, 
                f"{structure_passed}/{structure_total} files found")
    
    # Test 4: Environment Configuration
    print_header("Environment Configuration", 2)
    env_passed, env_details = check_env_file()
    print_result("Environment Variables", env_passed, env_details)
    report['tests']['environment_config'] = {'passed': env_passed, 'details': env_details}
    
    # Test 5: Database Connection
    print_header("Database Connection", 2)
    db_passed, db_details = test_database_connection()
    print_result("PostgreSQL Connection", db_passed, db_details)
    report['tests']['database_connection'] = {'passed': db_passed, 'details': db_details}
    
    # Test 6: Core Imports
    print_header("Core Project Imports", 2)
    import_results = test_core_imports()
    imports_passed = 0
    for import_name, (passed, details) in import_results.items():
        print_result(import_name, passed, details)
        if passed:
            imports_passed += 1
        report['tests'][f'import_{import_name}'] = {'passed': passed, 'details': details}
    
    # Test 7: PyTorch Functionality
    print_header("PyTorch and Model Testing", 2)
    torch_passed, torch_details = test_pytorch_functionality()
    print_result("PyTorch Functionality", torch_passed, torch_details)
    report['tests']['pytorch_functionality'] = {'passed': torch_passed, 'details': torch_details}
    
    # Test 8: Data Processing
    print_header("Data Processing", 2)
    data_passed, data_details = test_data_processing()
    print_result("Image Processing", data_passed, data_details)
    report['tests']['data_processing'] = {'passed': data_passed, 'details': data_details}
    
    # Test 9: Disk Space
    print_header("System Resources", 2)
    disk_passed, disk_details = check_disk_space()
    print_result("Disk Space", disk_passed, disk_details)
    report['tests']['disk_space'] = {'passed': disk_passed, 'details': disk_details}
    
    # Generate summary
    total_tests = len(report['tests'])
    passed_tests = sum(1 for test in report['tests'].values() if test['passed'])
    success_rate = passed_tests / total_tests if total_tests > 0 else 0
    
    report['summary'] = {
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'failed_tests': total_tests - passed_tests,
        'success_rate': success_rate,
        'ready_for_workflow': success_rate >= 0.8
    }
    
    # Print final summary
    print_header("SETUP VERIFICATION SUMMARY")
    print(f"📊 Tests Passed: {passed_tests}/{total_tests} ({success_rate:.1%})")
    print(f"🔗 Database Connected: {'Yes' if db_passed else 'No'}")
    print(f"🐍 Python Dependencies: {deps_passed}/{len(deps_results)} installed")
    print(f"📁 Project Files: {structure_passed}/{structure_total} found")
    print(f"🧠 PyTorch Ready: {'Yes' if torch_passed else 'No'}")
    
    if success_rate >= 0.9:
        print("\n🎉 EXCELLENT! Your setup is ready for the complete workflow.")
        print("   Run: python scripts/run_pipeline.py comprehensive")
    elif success_rate >= 0.8:
        print("\n✅ GOOD! Your setup is mostly ready. Fix any failed tests and proceed.")
        print("   Run: python scripts/run_pipeline.py comprehensive")
    elif success_rate >= 0.6:
        print("\n⚠️ PARTIAL! Some components need attention. Fix critical issues first.")
        print("   Focus on: Database connection, Dependencies, Project files")
    else:
        print("\n❌ SETUP INCOMPLETE! Please fix the major issues before proceeding.")
        print("   Check: Dependencies installation, Project structure, Database config")
    
    print("\n" + "="*70)
    
    return report

def save_report(report: Dict):
    """Save verification report to file"""
    try:
        project_root = Path(__file__).parent.parent
        logs_dir = project_root / 'logs'
        logs_dir.mkdir(exist_ok=True)
        
        report_file = logs_dir / 'setup_verification_report.json'
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Verification report saved: {report_file}")
        
    except Exception as e:
        print(f"⚠️ Could not save report: {e}")

def main():
    """Main verification function"""
    try:
        # Generate and display report
        report = generate_setup_report()
        
        # Save report
        save_report(report)
        
        # Provide next steps
        if report['summary']['ready_for_workflow']:
            print("\n🚀 NEXT STEPS:")
            print("1. Run comprehensive workflow:")
            print("   python scripts/run_pipeline.py comprehensive --selfies-limit 500 --glasses-limit 50")
            print()
            print("2. Or run step-by-step:")
            print("   python scripts/run_pipeline.py setup")
            print("   python scripts/run_pipeline.py download-selfies --limit 1000")
            print("   python scripts/run_pipeline.py train --epochs 50")
        else:
            print("\n🔧 REQUIRED FIXES:")
            failed_tests = [name for name, result in report['tests'].items() if not result['passed']]
            for test_name in failed_tests[:5]:  # Show first 5 failed tests
                print(f"   - Fix: {test_name}")
            
            print("\n   Then re-run: python scripts/verify_setup.py")
        
        return report['summary']['success_rate']
        
    except Exception as e:
        print(f"❌ Verification failed with error: {e}")
        traceback.print_exc()
        return 0.0

if __name__ == "__main__":
    success_rate = main()