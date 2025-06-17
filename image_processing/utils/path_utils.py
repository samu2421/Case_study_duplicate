"""
Path utilities for the Virtual Glasses Try-On project using pathlib
"""
from pathlib import Path
from typing import Union, List, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProjectPaths:
    """Centralized path management for the project"""
    
    def __init__(self, project_root: Optional[Union[str, Path]] = None):
        """
        Initialize project paths
        
        Args:
            project_root: Root directory of the project. If None, will try to detect automatically.
        """
        if project_root is None:
            # Try to find project root by looking for key files/directories
            current_path = Path.cwd()
            for parent in [current_path] + list(current_path.parents):
                if (parent / "requirements.txt").exists() or (parent / "models").exists():
                    self.project_root = parent
                    break
            else:
                # Default to current directory
                self.project_root = current_path
        else:
            self.project_root = Path(project_root)
        
        logger.info(f"Project root set to: {self.project_root.absolute()}")
        
        # Define all major paths
        self._setup_paths()
    
    def _setup_paths(self):
        """Setup all project paths"""
        # Data directories
        self.data_dir = self.project_root / "data"
        self.data_raw_dir = self.data_dir / "raw"
        self.data_processed_dir = self.data_dir / "processed"
        
        # Raw data subdirectories
        self.raw_selfies_dir = self.data_raw_dir / "selfies"
        self.raw_glasses_dir = self.data_raw_dir / "glasses"
        
        # Processed data subdirectories
        self.processed_selfies_dir = self.data_processed_dir / "selfies"
        self.processed_glasses_dir = self.data_processed_dir / "glasses"
        
        # Demo directories
        self.demo_dir = self.project_root / "demo"
        self.demo_selfies_dir = self.demo_dir / "selfies"
        self.demo_output_dir = self.demo_dir / "output"
        
        # Model directories
        self.models_dir = self.project_root / "models"
        self.sam_models_dir = self.models_dir / "sam"
        self.dino_models_dir = self.models_dir / "dino"
        
        # Other directories
        self.notebooks_dir = self.project_root / "notebooks"
        self.scripts_dir = self.project_root / "scripts"
        self.logs_dir = self.project_root / "logs"
        self.config_dir = self.project_root / "config"
        
        # Image processing directories
        self.image_processing_dir = self.project_root / "image_processing"
        self.download_dir = self.image_processing_dir / "download"
        self.preprocess_dir = self.image_processing_dir / "preprocess"
        self.utils_dir = self.image_processing_dir / "utils"
    
    def ensure_directories_exist(self) -> None:
        """Create all necessary directories if they don't exist"""
        directories = [
            self.data_dir, self.data_raw_dir, self.data_processed_dir,
            self.raw_selfies_dir, self.raw_glasses_dir,
            self.processed_selfies_dir, self.processed_glasses_dir,
            self.demo_dir, self.demo_selfies_dir, self.demo_output_dir,
            self.models_dir, self.sam_models_dir, self.dino_models_dir,
            self.notebooks_dir, self.scripts_dir, self.logs_dir, self.config_dir,
            self.image_processing_dir, self.download_dir, self.preprocess_dir, self.utils_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")
    
    def get_selfie_path(self, filename: str, processed: bool = False) -> Path:
        """
        Get path for a selfie image
        
        Args:
            filename: Name of the selfie file
            processed: If True, return processed path, else raw path
        
        Returns:
            Path to the selfie file
        """
        if processed:
            return self.processed_selfies_dir / filename
        else:
            return self.raw_selfies_dir / filename
    
    def get_glasses_path(self, filename: str, processed: bool = False) -> Path:
        """
        Get path for a glasses image
        
        Args:
            filename: Name of the glasses file
            processed: If True, return processed path, else raw path
        
        Returns:
            Path to the glasses file
        """
        if processed:
            return self.processed_glasses_dir / filename
        else:
            return self.raw_glasses_dir / filename
    
    def get_demo_selfie_path(self, filename: str) -> Path:
        """Get path for a demo selfie"""
        return self.demo_selfies_dir / filename
    
    def get_demo_output_path(self, filename: str) -> Path:
        """Get path for demo output"""
        return self.demo_output_dir / filename
    
    def get_model_checkpoint_path(self, model_name: str, checkpoint_name: str) -> Path:
        """
        Get path for model checkpoint
        
        Args:
            model_name: Name of the model (e.g., 'sam', 'dino')
            checkpoint_name: Name of the checkpoint file
        
        Returns:
            Path to the checkpoint file
        """
        if model_name.lower() == 'sam':
            return self.sam_models_dir / checkpoint_name
        elif model_name.lower() == 'dino':
            return self.dino_models_dir / checkpoint_name
        else:
            return self.models_dir / model_name / checkpoint_name
    
    def get_log_path(self, log_name: str) -> Path:
        """Get path for log file"""
        return self.logs_dir / log_name
    
    def list_selfies(self, processed: bool = False, pattern: str = "*") -> List[Path]:
        """
        List all selfie files
        
        Args:
            processed: If True, list processed selfies, else raw selfies
            pattern: File pattern to match (e.g., "*.jpg", "*.png")
        
        Returns:
            List of paths to selfie files
        """
        directory = self.processed_selfies_dir if processed else self.raw_selfies_dir
        if directory.exists():
            return list(directory.glob(pattern))
        return []
    
    def list_glasses(self, processed: bool = False, pattern: str = "*") -> List[Path]:
        """
        List all glasses files
        
        Args:
            processed: If True, list processed glasses, else raw glasses
            pattern: File pattern to match (e.g., "*.png")
        
        Returns:
            List of paths to glasses files
        """
        directory = self.processed_glasses_dir if processed else self.raw_glasses_dir
        if directory.exists():
            return list(directory.glob(pattern))
        return []
    
    def get_relative_path(self, absolute_path: Union[str, Path]) -> str:
        """
        Get relative path from project root
        
        Args:
            absolute_path: Absolute path to convert
        
        Returns:
            Relative path as string
        """
        try:
            abs_path = Path(absolute_path)
            return str(abs_path.relative_to(self.project_root))
        except ValueError:
            # Path is not relative to project root
            return str(absolute_path)
    
    def __str__(self) -> str:
        """String representation of project paths"""
        return f"ProjectPaths(root={self.project_root})"
    
    def __repr__(self) -> str:
        """Detailed representation of project paths"""
        paths_info = {
            'project_root': self.project_root,
            'raw_selfies': self.raw_selfies_dir,
            'processed_selfies': self.processed_selfies_dir,
            'raw_glasses': self.raw_glasses_dir,
            'processed_glasses': self.processed_glasses_dir,
            'models': self.models_dir,
            'demo': self.demo_dir
        }
        return f"ProjectPaths({paths_info})"


# Global instance for easy access
paths = ProjectPaths()

# Convenience functions
def get_project_root() -> Path:
    """Get the project root directory"""
    return paths.project_root

def get_demo_selfies_path() -> Path:
    """Get demo selfies directory path"""
    return paths.demo_selfies_dir

def get_demo_glasses_path() -> Path:
    """Get demo glasses directory path - create glasses dir in demo if needed"""
    demo_glasses_dir = paths.demo_dir / "glasses"
    demo_glasses_dir.mkdir(exist_ok=True)
    return demo_glasses_dir

def get_demo_output_path() -> Path:
    """Get demo output directory path"""
    return paths.demo_output_dir

def ensure_all_directories() -> None:
    """Ensure all project directories exist"""
    paths.ensure_directories_exist()

# Example usage and testing
if __name__ == "__main__":
    # Test the path utilities
    print("Testing ProjectPaths...")
    
    # Create instance
    project_paths = ProjectPaths()
    
    # Print some key paths
    print(f"Project root: {project_paths.project_root}")
    print(f"Raw selfies: {project_paths.raw_selfies_dir}")
    print(f"Processed selfies: {project_paths.processed_selfies_dir}")
    print(f"Models directory: {project_paths.models_dir}")
    print(f"Demo output: {project_paths.demo_output_dir}")
    
    # Ensure directories exist
    project_paths.ensure_directories_exist()
    print(" All directories created/verified")
    
    # Test convenience functions
    print(f"Demo selfies path: {get_demo_selfies_path()}")
    print(f"Demo glasses path: {get_demo_glasses_path()}")
    
    # Test file path generation
    test_selfie_path = project_paths.get_selfie_path("test_selfie.jpg")
    test_glasses_path = project_paths.get_glasses_path("test_glasses.png", processed=True)
    print(f"Test selfie path: {test_selfie_path}")
    print(f"Test processed glasses path: {test_glasses_path}")
    
    print(" Path utilities test completed")