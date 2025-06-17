"""
Script to create the complete project directory structure
"""
from pathlib import Path

def create_project_structure():
    """Create the complete directory structure for the project"""
    
    # Define the main project directory
    project_root = Path("Generative-AI-Based-Virtual-Glasses-Try-On")
    
    # Define all directories to create
    directories = [
        project_root / "data" / "raw" / "selfies",
        project_root / "data" / "raw" / "glasses", 
        project_root / "data" / "processed" / "selfies",
        project_root / "data" / "processed" / "glasses",
        project_root / "demo" / "selfies",
        project_root / "demo" / "output",
        project_root / "image_processing" / "download",
        project_root / "image_processing" / "preprocess", 
        project_root / "image_processing" / "utils",
        project_root / "models" / "sam",
        project_root / "models" / "dino",
        project_root / "notebooks",
        project_root / "scripts",
        project_root / "config",
        project_root / "logs",
    ]
    
    # Create all directories
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Create __init__.py files to make packages
    init_files = [
        project_root / "image_processing" / "__init__.py",
        project_root / "image_processing" / "download" / "__init__.py",
        project_root / "image_processing" / "preprocess" / "__init__.py",
        project_root / "image_processing" / "utils" / "__init__.py",
        project_root / "models" / "__init__.py",
    ]
    
    for init_file in init_files:
        init_file.touch()
        print(f"Created __init__.py: {init_file}")
    
    print(f"\n Project structure created successfully in: {project_root.absolute()}")
    return project_root

if __name__ == "__main__":
    create_project_structure()
