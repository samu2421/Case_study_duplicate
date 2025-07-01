#!/usr/bin/env python3
"""
Command Line Interface for Virtual Glasses Try-On System.
Provides a simple, user-friendly interface for all system operations.
"""

import sys
import argparse
from pathlib import Path
import subprocess
import logging
from typing import List, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from config.settings import setup_environment, get_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VirtualTryOnCLI:
    """Command Line Interface for Virtual Glasses Try-On System."""
    
    def __init__(self):
        """Initialize CLI."""
        self.project_root = Path(__file__).parent
        try:
            self.config = setup_environment()
        except Exception as e:
            logger.warning(f"Failed to setup environment: {e}")
            self.config = None
    
    def run_command(self, command: List[str], description: str = None) -> bool:
        """Run a command and return success status."""
        try:
            if description:
                logger.info(f"🚀 {description}")
            
            logger.debug(f"Running command: {' '.join(command)}")
            result = subprocess.run(command, cwd=self.project_root, check=True)
            
            if description:
                logger.info(f"✅ {description} completed successfully")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Command failed with exit code {e.returncode}")
            return False
        except Exception as e:
            logger.error(f"❌ Command execution failed: {e}")
            return False
    
    def cmd_setup(self, args) -> int:
        """Setup the project environment."""
        cmd = [sys.executable, "setup.py"]
        
        if args.skip_deps:
            cmd.append("--skip-deps")
        if args.skip_db_test:
            cmd.append("--skip-db-test")
        if args.upgrade_deps:
            cmd.append("--upgrade-deps")
        if args.verbose:
            cmd.append("--verbose")
        
        success = self.run_command(cmd, "Project setup")
        return 0 if success else 1
    
    def cmd_pipeline(self, args) -> int:
        """Run the complete pipeline."""
        cmd = [sys.executable, "scripts/run_pipeline.py"]
        
        if args.skip_download:
            cmd.append("--skip-download")
        if args.force_download:
            cmd.append("--force-download")
        if args.skip_preprocessing:
            cmd.append("--skip-preprocessing")
        if args.force_preprocessing:
            cmd.append("--force-preprocessing")
        if args.skip_demo:
            cmd.append("--skip-demo")
        if args.glasses_limit:
            cmd.extend(["--glasses-limit", str(args.glasses_limit)])
        if args.demo_selfies:
            cmd.extend(["--demo-selfies", str(args.demo_selfies)])
        if args.demo_glasses:
            cmd.extend(["--demo-glasses", str(args.demo_glasses)])
        if args.verbose:
            cmd.append("--verbose")
        
        success = self.run_command(cmd, "Pipeline execution")
        return 0 if success else 1
    
    def cmd_train(self, args) -> int:
        """Train the alignment module."""
        cmd = [sys.executable, "scripts/train_model.py"]
        
        if args.epochs:
            cmd.extend(["--epochs", str(args.epochs)])
        if args.learning_rate:
            cmd.extend(["--learning-rate", str(args.learning_rate)])
        if args.batch_size:
            cmd.extend(["--batch-size", str(args.batch_size)])
        if args.save_path:
            cmd.extend(["--save-path", args.save_path])
        if args.device:
            cmd.extend(["--device", args.device])
        
        success = self.run_command(cmd, "Model training")
        return 0 if success else 1
    
    def cmd_demo(self, args) -> int:
        """Run demo."""
        cmd = [sys.executable, "demo/demo_tryon.py"]
        
        if args.interactive:
            cmd.append("--interactive")
        
        success = self.run_command(cmd, "Demo execution")
        return 0 if success else 1
    
    def cmd_notebook(self, args) -> int:
        """Start Jupyter notebook."""
        try:
            import jupyter
            cmd = [sys.executable, "-m", "jupyter", "lab", "notebooks/"]
            
            logger.info("🚀 Starting Jupyter Lab...")
            logger.info("   Access the notebook at: http://localhost:8888")
            logger.info("   Press Ctrl+C to stop the server")
            
            subprocess.run(cmd, cwd=self.project_root)
            return 0
            
        except ImportError:
            logger.error("❌ Jupyter not installed. Install with: pip install jupyter")
            return 1
        except KeyboardInterrupt:
            logger.info("✅ Jupyter server stopped")
            return 0
        except Exception as e:
            logger.error(f"❌ Failed to start Jupyter: {e}")
            return 1
    
    def cmd_docker(self, args) -> int:
        """Docker operations."""
        if args.action == "build":
            cmd = ["docker-compose", "build"]
            if args.no_cache:
                cmd.append("--no-cache")
            description = "Docker build"
            
        elif args.action == "up":
            cmd = ["docker-compose", "up"]
            if args.service:
                cmd.append(args.service)
            if args.detach:
                cmd.append("-d")
            description = "Docker compose up"
            
        elif args.action == "down":
            cmd = ["docker-compose", "down"]
            if args.volumes:
                cmd.append("-v")
            description = "Docker compose down"
            
        elif args.action == "logs":
            cmd = ["docker-compose", "logs"]
            if args.service:
                cmd.append(args.service)
            if args.follow:
                cmd.append("-f")
            description = "Docker logs"
            
        else:
            logger.error(f"❌ Unknown docker action: {args.action}")
            return 1
        
        success = self.run_command(cmd, description)
        return 0 if success else 1
    
    def cmd_migrate(self, args) -> int:
        """Run database migration."""
        cmd = [sys.executable, "database/migrate_schema.py"]
        
        success = self.run_command(cmd, "Database migration")
        return 0 if success else 1
    
    def cmd_status(self, args) -> int:
        """Show system status."""
        try:
            logger.info("📊 Virtual Glasses Try-On System Status")
            logger.info("=" * 50)
            
            # Python environment
            logger.info(f"Python: {sys.version.split()[0]}")
            logger.info(f"Platform: {sys.platform}")
            
            # Project structure
            logger.info(f"Project root: {self.project_root}")
            
            # Check key directories
            key_dirs = ["data", "models", "demo/output", "logs"]
            for dir_name in key_dirs:
                dir_path = self.project_root / dir_name
                status = "✅" if dir_path.exists() else "❌"
                logger.info(f"{status} {dir_name}: {dir_path}")
            
            # Check dependencies
            try:
                import torch
                cuda_status = "✅ Available" if torch.cuda.is_available() else "❌ Not available"
                logger.info(f"PyTorch: {torch.__version__}")
                logger.info(f"CUDA: {cuda_status}")
            except ImportError:
                logger.info("PyTorch: ❌ Not installed")
            
            # Database connection
            try:
                if self.config:
                    from database.config import db_manager
                    if db_manager.connect():
                        logger.info("Database: ✅ Connected")
                    else:
                        logger.info("Database: ❌ Connection failed")
                else:
                    logger.info("Database: ❌ Config not available")
            except Exception as e:
                logger.info(f"Database: ❌ Error - {e}")
            
            # Docker status
            try:
                result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
                if result.returncode == 0:
                    logger.info(f"Docker: ✅ {result.stdout.strip()}")
                else:
                    logger.info("Docker: ❌ Not available")
            except FileNotFoundError:
                logger.info("Docker: ❌ Not installed")
            
            return 0
            
        except Exception as e:
            logger.error(f"❌ Status check failed: {e}")
            return 1
    
    def cmd_clean(self, args) -> int:
        """Clean up generated files and caches."""
        try:
            logger.info("🧹 Cleaning up system...")
            
            # Clean up patterns
            cleanup_patterns = [
                "**/__pycache__",
                "**/*.pyc",
                "**/*.pyo",
                ".pytest_cache",
                "*.log"
            ]
            
            if args.data:
                cleanup_patterns.extend([
                    "data/processed/**/*",
                    "demo/output/**/*"
                ])
            
            if args.models:
                cleanup_patterns.extend([
                    "models/.cache/**/*",
                    "models/**/model_*.pth"
                ])
            
            if args.logs:
                cleanup_patterns.append("logs/**/*.log")
            
            cleaned_count = 0
            for pattern in cleanup_patterns:
                for file_path in self.project_root.glob(pattern):
                    if file_path.is_file():
                        file_path.unlink()
                        cleaned_count += 1
                        logger.debug(f"Removed: {file_path}")
                    elif file_path.is_dir() and not any(file_path.iterdir()):
                        file_path.rmdir()
                        cleaned_count += 1
                        logger.debug(f"Removed empty directory: {file_path}")
            
            logger.info(f"✅ Cleaned up {cleaned_count} files/directories")
            return 0
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
            return 1

def create_parser():
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="Virtual Glasses Try-On System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s setup                     # Setup project environment
  %(prog)s pipeline                  # Run complete pipeline
  %(prog)s pipeline --skip-download  # Run pipeline without downloading dataset
  %(prog)s train --epochs 30         # Train for 30 epochs
  %(prog)s demo --interactive        # Run interactive demo
  %(prog)s docker build             # Build Docker images
  %(prog)s status                    # Show system status
        """
    )
    
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Setup project environment")
    setup_parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    setup_parser.add_argument("--skip-db-test", action="store_true", help="Skip database test")
    setup_parser.add_argument("--upgrade-deps", action="store_true", help="Upgrade dependencies")
    setup_parser.set_defaults(func=lambda cli, args: cli.cmd_setup(args))
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Run complete pipeline")
    pipeline_parser.add_argument("--skip-download", action="store_true", help="Skip dataset download")
    pipeline_parser.add_argument("--force-download", action="store_true", help="Force dataset download")
    pipeline_parser.add_argument("--skip-preprocessing", action="store_true", help="Skip preprocessing")
    pipeline_parser.add_argument("--force-preprocessing", action="store_true", help="Force preprocessing")
    pipeline_parser.add_argument("--skip-demo", action="store_true", help="Skip demo")
    pipeline_parser.add_argument("--glasses-limit", type=int, help="Limit glasses to process")
    pipeline_parser.add_argument("--demo-selfies", type=int, default=3, help="Number of demo selfies")
    pipeline_parser.add_argument("--demo-glasses", type=int, default=3, help="Number of demo glasses")
    pipeline_parser.set_defaults(func=lambda cli, args: cli.cmd_pipeline(args))
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train alignment module")
    train_parser.add_argument("--epochs", type=int, help="Number of epochs")
    train_parser.add_argument("--learning-rate", type=float, help="Learning rate")
    train_parser.add_argument("--batch-size", type=int, help="Batch size")
    train_parser.add_argument("--save-path", help="Model save path")
    train_parser.add_argument("--device", help="Device to use (auto, cpu, cuda)")
    train_parser.set_defaults(func=lambda cli, args: cli.cmd_train(args))
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run demo")
    demo_parser.add_argument("--interactive", action="store_true", help="Run interactive demo")
    demo_parser.set_defaults(func=lambda cli, args: cli.cmd_demo(args))
    
    # Notebook command
    notebook_parser = subparsers.add_parser("notebook", help="Start Jupyter notebook")
    notebook_parser.set_defaults(func=lambda cli, args: cli.cmd_notebook(args))
    
    # Docker command
    docker_parser = subparsers.add_parser("docker", help="Docker operations")
    docker_parser.add_argument("action", choices=["build", "up", "down", "logs"], help="Docker action")
    docker_parser.add_argument("--service", help="Specific service")
    docker_parser.add_argument("--no-cache", action="store_true", help="Build without cache")
    docker_parser.add_argument("--detach", "-d", action="store_true", help="Run in background")
    docker_parser.add_argument("--volumes", "-v", action="store_true", help="Remove volumes")
    docker_parser.add_argument("--follow", "-f", action="store_true", help="Follow logs")
    docker_parser.set_defaults(func=lambda cli, args: cli.cmd_docker(args))
    
    # Migrate command
    migrate_parser = subparsers.add_parser("migrate", help="Run database migration")
    migrate_parser.set_defaults(func=lambda cli, args: cli.cmd_migrate(args))
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show system status")
    status_parser.set_defaults(func=lambda cli, args: cli.cmd_status(args))
    
    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean up files")
    clean_parser.add_argument("--data", action="store_true", help="Clean processed data")
    clean_parser.add_argument("--models", action="store_true", help="Clean model caches")
    clean_parser.add_argument("--logs", action="store_true", help="Clean log files")
    clean_parser.set_defaults(func=lambda cli, args: cli.cmd_clean(args))
    
    return parser

def main():
    """Main CLI function."""
    try:
        parser = create_parser()
        args = parser.parse_args()
        
        # Set verbose logging
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Show help if no command provided
        if not args.command:
            parser.print_help()
            return 0
        
        # Create CLI instance and run command
        cli = VirtualTryOnCLI()
        return args.func(cli, args)
        
    except KeyboardInterrupt:
        print("\n❌ Operation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ CLI error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())