# setup.py
import subprocess
import sys

def install_packages():
    packages = [
        'tensorflow>=2.13.0',
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'matplotlib>=3.7.0',
        'Pillow>=10.0.0',
        'scikit-learn>=1.3.0',
        'opencv-python>=4.8.0',
        'flask>=3.0.0'
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("All packages installed successfully!")

if __name__ == "__main__":
    install_packages()