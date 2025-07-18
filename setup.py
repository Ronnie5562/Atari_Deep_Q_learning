#!/usr/bin/env python3
"""
Setup script for DQN Breakout project
Run this script to install all dependencies and set up the environment
"""

import subprocess
import sys
import os

def run_command(command):
    """Run a command and handle errors"""
    try:
        result = subprocess.run(command, shell=True, check=True,
                              capture_output=True, text=True)
        print(f"{command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{command}")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Main setup function"""

    print("DQN Breakout Project Setup")
    print("=" * 40)

    # Check Python version
    if sys.version_info < (3, 8):
        print("Python 3.8+ required. Current version:", sys.version)
        return False

    print("Python version check passed")

    # Install requirements
    print("\n Installing dependencies...")
    if not run_command("pip install -r requirements.txt"):
        print("Failed to install dependencies")
        return False

    # Accept ROM license
    print("\n🎮 Setting up Atari ROMs...")
    if not run_command("AutoROM --accept-license"):
        print("Failed to install Atari ROMs")
        print("Try running manually: AutoROM --accept-license")
        return False

    # Create logs directory
    print("\n Creating directories...")
    os.makedirs("logs", exist_ok=True)
    print("Created logs directory")

    # Test imports
    print("\n🧪 Testing imports...")
    try:
        import gymnasium
        import stable_baselines3
        import torch
        print("All imports successful")
    except ImportError as e:
        print(f"Import error: {e}")
        return False

    print("\n Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run 'python train.py' to train the agent")
    print("2. Run 'python play.py' to play with the trained agent")
    print("3. Use 'tensorboard --logdir logs/' to monitor training")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
