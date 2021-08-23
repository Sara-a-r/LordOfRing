import subprocess
import sys

def install_requirements():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt" ])
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "gpu_requirements.txt" ])
    except Exception as e:
        print('Error during the install of pycuda, Nvidia GPU needed. ', e)

if __name__ == '__main__':
    install_requirements()
