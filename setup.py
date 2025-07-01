"""
Setup script for Virtual Glasses Try-On System.
Handles initial project setup, dependency verification, and environment configuration.
"""

import sys
import subprocess
import argparse
from pathlib import Path
import logging
import platform
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProjectSetup:
    """Handles project setup and environment configuration."""
    
    def __init__(self):
        """Initialize setup manager."""
        self.project_root = Path(__file__).parent
        self.python_version = sys.version_info
        self.platform = platform.system()
        
    def check_python_version(self) -> bool:
        """Check if Python version meets requirements."""
        required_version = (3, 10)
        if self.python_version < required_version:
            logger.error(f"Python {required_version[0]}.{required_version[1]}+ required. Current: {self.python_version[0]}.{self.python_version[1]}")
            return False
        
        logger.info(f"✅ Python version check passed: {self.python_version[0]}.{self.python_version[1]}.{self.python_version[2]}")
        return True
    
    def create_directories(self) -> bool:
        """Create necessary project directories."""
        try:
            directories = [
                "data/raw",
                "data/processed",
                "demo/selfies",
                "demo/output",
                "models/sam_checkpoints",
                "models/dino_checkpoints",
                "logs",
                "notebooks/outputs"
            ]
            
            for directory in directories:
                dir_path = self.project_root / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {dir_path}")
            
            logger.info("✅ Project directories created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create directories: {e}")
            return False
    
    def check_dependencies(self) -> bool:
        """Check if required system dependencies are available."""
        try:
            # Check for essential system packages
            required_commands = {
                'git': 'Git is required for downloading models',
                'wget': 'wget is required for downloading datasets (or curl)',
                'unzip': 'unzip is required for extracting datasets'
            }
            
            missing_commands = []
            for cmd, description in required_commands.items():
                if not shutil.which(cmd):
                    # For wget, check if curl is available as alternative
                    if cmd == 'wget' and shutil.which('curl'):
                        continue
                    missing_commands.append(f"{cmd}: {description}")
            
            if missing_commands:
                logger.warning("⚠️ Missing system dependencies:")
                for cmd in missing_commands:
                    logger.warning(f"  - {cmd}")
                logger.warning("Please install missing dependencies using your system package manager")
                return False
            
            logger.info("✅ System dependencies check passed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Dependency check failed: {e}")
            return False
    
    def install_python_dependencies(self, upgrade: bool = False) -> bool:
        """Install Python dependencies."""
        try:
            requirements_file = self.project_root / "requirements.txt"
            if not requirements_file.exists():
                logger.error("❌ requirements.txt not found")
                return False
            
            cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
            if upgrade:
                cmd.append("--upgrade")
            
            logger.info("Installing Python dependencies...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Python dependencies installed successfully")
                return True
            else:
                logger.error(f"❌ Failed to install dependencies: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Python dependency installation failed: {e}")
            return False
    
    def verify_pytorch_installation(self) -> bool:
        """Verify PyTorch installation and CUDA availability."""
        try:
            import torch
            logger.info(f"✅ PyTorch version: {torch.__version__}")
            
            if torch.cuda.is_available():
                logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
                logger.info(f"   CUDA version: {torch.version.cuda}")
                logger.info(f"   Available GPUs: {torch.cuda.device_count()}")
            else:
                logger.warning("⚠️ CUDA not available - using CPU mode")
                logger.info("   For better performance, consider installing CUDA-enabled PyTorch")
            
            return True
            
        except ImportError:
            logger.error("❌ PyTorch not installed properly")
            return False
        except Exception as e:
            logger.error(f"❌ PyTorch verification failed: {e}")
            return False
    
    def test_database_connection(self) -> bool:
        """Test database connection."""
        try:
            from database.config import db_manager
            
            if db_manager.connect():
                logger.info("✅ Database connection successful")
                
                # Test basic query
                result = db_manager.execute_query("SELECT 1 as test;")
                if len(result) > 0:
                    logger.info("✅ Database query test passed")
                    return True
                else:
                    logger.error("❌ Database query test failed")
                    return False
            else:
                logger.error("❌ Database connection failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Database connection test failed: {e}")
            return False
    
    def create_env_template(self) -> bool:
        """Create environment configuration template."""
        try:
            env_template = self.project_root / ".env.template"
            env_content = """# Virtual Glasses Try-On Environment Configuration

# Database Configuration
DB_HOST=152.53.12.68
DB_PORT=4000
DB_USER=student_diff
DB_PASSWORD=diff_pass
DB_DATABASE=postgres
DB_SCHEMA=diffusion

# Model Configuration
TORCH_HOME=./models/.cache/torch
HF_HOME=./models/.cache/huggingface

# Processing Configuration
DEFAULT_BATCH_SIZE=16
DEFAULT_IMAGE_SIZE=512
QUALITY_THRESHOLD=0.4

# Docker Configuration
DOCKER_BUILDKIT=1
COMPOSE_DOCKER_CLI_BUILD=1

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/application.log
"""
            
            with open(env_template, 'w') as f:
                f.write(env_content)
            
            logger.info("✅ Environment template created (.env.template)")
            logger.info("   Copy to .env and customize as needed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create environment template: {e}")
            return False
    
    def run_setup(self, install_deps: bool = True, test_db: bool = True) -> bool:
        """Run complete setup process."""
        logger.info("🚀 Starting Virtual Glasses Try-On project setup...")
        
        steps = [
            ("Python version check", self.check_python_version),
            ("Directory creation", self.create_directories),
            ("System dependencies check", self.check_dependencies),
            ("Environment template creation", self.create_env_template)
        ]
        
        if install_deps:
            steps.append(("Python dependencies installation", self.install_python_dependencies))
            steps.append(("PyTorch verification", self.verify_pytorch_installation))
        
        if test_db:
            steps.append(("Database connection test", self.test_database_connection))
        
        failed_steps = []
        for step_name, step_func in steps:
            logger.info(f"\n--- {step_name} ---")
            if not step_func():
                failed_steps.append(step_name)
        
        if failed_steps:
            logger.error(f"\n❌ Setup completed with {len(failed_steps)} failed steps:")
            for step in failed_steps:
                logger.error(f"   - {step}")
            return False
        else:
            logger.info("\n🎉 Setup completed successfully!")
            logger.info("\nNext steps:")
            logger.info("1. Run: python scripts/run_pipeline.py --help")
            logger.info("2. Or start with: python demo/demo_tryon.py")
            logger.info("3. For experiments: jupyter lab notebooks/")
            return True

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Virtual Glasses Try-On Project Setup")
    
    parser.add_argument("--skip-deps", action="store_true",
                       help="Skip Python dependency installation")
    parser.add_argument("--skip-db-test", action="store_true",
                       help="Skip database connection test")
    parser.add_argument("--upgrade-deps", action="store_true",
                       help="Upgrade existing dependencies")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    return parser.parse_args()

def main():
    """Main setup function."""
    try:
        args = parse_arguments()
        
        # Set logging level
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Create setup manager
        setup = ProjectSetup()
        
        # Override dependency installation if requested
        if args.upgrade_deps and not args.skip_deps:
            setup.install_python_dependencies = lambda: setup.install_python_dependencies(upgrade=True)
        
        # Run setup
        success = setup.run_setup(
            install_deps=not args.skip_deps,
            test_db=not args.skip_db_test
        )
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n❌ Setup interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())