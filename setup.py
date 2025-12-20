# ============================================================================
# FILE: setup.py
# PURPOSE: Complete project setup script
# ============================================================================

import os
import sys
import subprocess
import urllib.request
import zipfile
import shutil

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def create_directory_structure():
    """Create all required directories"""
    print_header("Creating Directory Structure")
    
    dirs = [
        'dataset/train/real',
        'dataset/train/fake',
        'dataset/val/real',
        'dataset/val/fake',
        'dataset/test/real',
        'dataset/test/fake',
        'face_recognition',
        'liveness_model',
        'inference',
        'static/css',
        'static/js',
        'static/uploads/faces',
        'templates',
        'logs'
    ]
    
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Created: {dir_path}")
    
    print("\n✓ Directory structure created successfully!")

def create_init_files():
    """Create __init__.py files for Python packages"""
    print_header("Creating Package Files")
    
    packages = [
        'face_recognition',
        'liveness_model',
        'inference'
    ]
    
    for package in packages:
        init_file = os.path.join(package, '__init__.py')
        with open(init_file, 'w') as f:
            f.write(f'"""{package} package"""\n')
        print(f"✓ Created: {init_file}")
    
    print("\n✓ Package files created successfully!")

def create_gitignore():
    """Create .gitignore file"""
    print_header("Creating .gitignore")
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Flask
instance/
.webassets-cache

# Model files
*.pth
*.h5
*.pkl

# Data
dataset/
*.csv
*.xlsx

# Uploads
static/uploads/*
!static/uploads/.gitkeep

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Logs
logs/
*.log
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    
    print("✓ .gitignore created")

def create_requirements_file():
    """Create requirements.txt with all dependencies"""
    print_header("Creating requirements.txt")
    
    requirements = """# Deep Learning
torch==2.0.1
torchvision==0.15.2

# Computer Vision
opencv-python==4.8.0.74
facenet-pytorch==2.5.3

# Data Processing
numpy==1.24.3
Pillow==10.0.0
imgaug==0.4.0

# Machine Learning
scikit-learn==1.3.0

# Web Framework
Flask==2.3.2
Flask-CORS==4.0.0

# Utilities
tqdm==4.65.0
matplotlib==3.7.2

# Deployment
gunicorn==21.2.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    
    print("✓ requirements.txt created")

def install_dependencies():
    """Install Python dependencies"""
    print_header("Installing Dependencies")
    print("This may take several minutes...")
    
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("\n✓ Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error installing dependencies: {e}")
        print("Please install manually: pip install -r requirements.txt")

def download_sample_dataset():
    """Download sample dataset for testing"""
    print_header("Download Sample Dataset")
    
    response = input("Do you want to download a sample dataset? (y/n): ")
    if response.lower() != 'y':
        print("Skipping dataset download.")
        return
    
    print("\nNote: Please manually download datasets from:")
    print("1. CASIA-FASD: http://www.cbsr.ia.ac.cn/")
    print("2. Replay-Attack: https://www.idiap.ch/en/dataset/replayattack")
    print("\nThen place them in the dataset/ folder following the structure:")
    print("  dataset/train/real/ and dataset/train/fake/")

def create_readme():
    """Create a simple README"""
    print_header("Creating README.md")
    
    readme = """# Face Recognition + Spoof Detection System

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare dataset:
   - Place images in `dataset/train/real/` and `dataset/train/fake/`
   - Split 70% train, 15% val, 15% test

3. Train liveness model:
```bash
python liveness_model/train.py
```

4. Run application:
```bash
python app.py
```

5. Open browser: http://localhost:5000

## Project Structure

See complete documentation in the main README file.

## Support

For issues, please open a GitHub issue or contact the maintainer.
"""
    
    with open('README.md', 'w') as f:
        f.write(readme)
    
    print("✓ README.md created")

def main():
    """Main setup function"""
    print("\n" + "="*60)
    print("  🔐 Face Recognition + Spoof Detection System")
    print("  Project Setup Wizard")
    print("="*60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("\n✗ Error: Python 3.8 or higher is required")
        print(f"  Current version: {sys.version}")
        sys.exit(1)
    
    print(f"\n✓ Python version: {sys.version.split()[0]}")
    
    # Setup steps
    create_directory_structure()
    create_init_files()
    create_gitignore()
    create_requirements_file()
    create_readme()
    
    # Install dependencies
    install_choice = input("\nInstall dependencies now? (y/n): ")
    if install_choice.lower() == 'y':
        install_dependencies()
    
    # Download dataset
    download_sample_dataset()
    
    # Final message
    print_header("Setup Complete!")
    print("\n✅ Project setup completed successfully!")
    print("\nNext steps:")
    print("1. Prepare your dataset in dataset/ folder")
    print("2. Run: python liveness_model/train.py")
    print("3. Run: python app.py")
    print("4. Open: http://localhost:5000")
    print("\n" + "="*60)

if __name__ == '__main__':
    main()


