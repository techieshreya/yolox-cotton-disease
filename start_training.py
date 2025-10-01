#!/usr/bin/env python3
"""
Interactive Training Mode Selection Script
Helps users choose between manual and auto-optimized training modes.
"""

import os
import sys
import subprocess
import torch
from typing import Optional, Tuple


def get_terminal_width():
    """Get terminal width for formatting."""
    try:
        return os.get_terminal_size().columns
    except:
        return 80


def print_header(text: str, char: str = "="):
    """Print a formatted header."""
    width = min(get_terminal_width(), 100)
    print(f"\n{char * width}")
    print(f"{text:^{width}}")
    print(f"{char * width}\n")


def print_section(title: str):
    """Print a section title."""
    print(f"\n{'─' * 60}")
    print(f"📌 {title}")
    print(f"{'─' * 60}\n")


def detect_hardware() -> Tuple[bool, Optional[str], float, float]:
    """
    Detect available hardware.

    Returns:
        (has_cuda, gpu_name, total_vram_gb, free_vram_gb)
    """
    has_cuda = torch.cuda.is_available()

    if has_cuda:
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        total_vram = props.total_memory / (1024**3)
        torch.cuda.empty_cache()
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        free_vram = total_vram - reserved
        return True, gpu_name, total_vram, free_vram
    else:
        return False, None, 0.0, 0.0


def show_hardware_info(has_cuda: bool, gpu_name: Optional[str], total_vram: float, free_vram: float):
    """Display detected hardware information."""
    print_section("HARDWARE DETECTION")

    if has_cuda:
        print(f"✅ GPU Detected: {gpu_name}")
        print(f"   ├─ Total VRAM:     {total_vram:.2f} GB")
        print(f"   └─ Available VRAM: {free_vram:.2f} GB")

        # CPU info
        cpu_count = os.cpu_count() or 4
        print(f"\n✅ CPU Cores: {cpu_count}")
    else:
        print("❌ No GPU detected!")
        print("   Training on CPU is not recommended for YOLOX.")
        print("   It will be extremely slow.")


def explain_training_modes():
    """Explain the differences between training modes."""
    print_section("TRAINING MODES EXPLAINED")

    print("🔧 MODE 1: train.py (Manual Configuration)")
    print("   ├─ You manually set all parameters")
    print("   ├─ Good for: Experienced users, specific requirements")
    print("   ├─ Requires: Knowledge of optimal batch sizes, model sizes, etc.")
    print("   └─ Risk: Sub-optimal settings or OOM errors")

    print("\n🚀 MODE 2: train-max.py (Auto-Optimized)")
    print("   ├─ Automatically detects your GPU and optimizes everything")
    print("   ├─ Good for: Everyone, especially beginners")
    print("   ├─ Requires: Nothing! Just run it")
    print("   └─ Benefits: Maximum GPU utilization, no OOM errors")


def recommend_mode(has_cuda: bool, free_vram: float) -> int:
    """
    Recommend a training mode based on hardware.

    Returns:
        1 for train.py, 2 for train-max.py
    """
    if not has_cuda:
        return 1  # Manual mode for CPU

    # Always recommend auto mode for GPU users
    return 2


def show_recommendation(mode: int, free_vram: float):
    """Show recommendation based on hardware."""
    print_section("RECOMMENDATION")

    if mode == 2:
        print("🎯 RECOMMENDED: train-max.py (Auto-Optimized)")
        print("\n   Why?")
        print("   • Automatically optimizes for your GPU")
        print("   • Maximizes training speed and efficiency")
        print("   • Prevents out-of-memory errors")
        print("   • No parameter tuning needed")

        if free_vram >= 16:
            print("\n   With your GPU, you can train:")
            print("   • Large models (YOLOX-L or YOLOX-X)")
            print("   • Large batch sizes (12-16)")
            print("   • Expected training: ~6-8 hours for 150 epochs")
        elif free_vram >= 8:
            print("\n   With your GPU, you can train:")
            print("   • Medium models (YOLOX-M)")
            print("   • Good batch sizes (6-8)")
            print("   • Expected training: ~8-10 hours for 150 epochs")
        elif free_vram >= 4:
            print("\n   With your GPU, you can train:")
            print("   • Small models (YOLOX-S)")
            print("   • Small batch sizes (2-4)")
            print("   • Expected training: ~10-12 hours for 150 epochs")
    else:
        print("⚠️  Manual mode (train.py)")
        print("\n   Only use this if you:")
        print("   • Have specific parameter requirements")
        print("   • Are an experienced user")
        print("   • Want full control over training")


def get_user_choice(recommended: int) -> int:
    """Get user's training mode choice."""
    print_section("SELECT TRAINING MODE")

    print("Choose your training mode:\n")
    print("  [1] train.py         - Manual configuration")
    print("  [2] train-max.py     - Auto-optimized (RECOMMENDED ⭐)")
    print("  [0] Exit\n")

    while True:
        try:
            choice = input(f"Enter your choice (0-2) [default: {recommended}]: ").strip()

            if choice == "":
                return recommended

            choice_int = int(choice)
            if choice_int in [0, 1, 2]:
                return choice_int
            else:
                print("❌ Invalid choice. Please enter 0, 1, or 2.")
        except ValueError:
            print("❌ Invalid input. Please enter a number.")


def get_training_parameters(mode: int) -> dict:
    """Get training parameters from user."""
    print_section("TRAINING PARAMETERS")

    params = {}

    # Common parameters
    print("Leave blank to use defaults\n")

    try:
        epochs = input("Number of epochs [150]: ").strip()
        params['epochs'] = int(epochs) if epochs else 150

        num_classes = input("Number of classes [5]: ").strip()
        params['num_classes'] = int(num_classes) if num_classes else 5

        save_dir = input("Save directory [./checkpoints]: ").strip()
        params['save_dir'] = save_dir if save_dir else "./checkpoints"

        if mode == 1:  # Manual mode
            print("\n📝 Manual mode - Additional parameters:\n")

            phi = input("Model size (s/m/l/x) [s]: ").strip().lower()
            params['phi'] = phi if phi in ['s', 'm', 'l', 'x'] else 's'

            batch_size = input("Batch size [2]: ").strip()
            params['batch_size'] = int(batch_size) if batch_size else 2

            input_h = input("Input height [640]: ").strip()
            input_w = input("Input width [640]: ").strip()
            params['input_size'] = [
                int(input_h) if input_h else 640,
                int(input_w) if input_w else 640
            ]

            lr = input("Learning rate [1e-3]: ").strip()
            params['lr'] = float(lr) if lr else 1e-3

            num_workers = input("Number of workers [0]: ").strip()
            params['num_workers'] = int(num_workers) if num_workers else 0

    except ValueError:
        print("\n❌ Invalid input. Using defaults.")
        params = {'epochs': 150, 'num_classes': 5, 'save_dir': './checkpoints'}

    return params


def build_command(mode: int, params: dict) -> list:
    """Build the training command."""
    if mode == 1:
        cmd = ["python", "train.py"]

        cmd.extend(["--epochs", str(params.get('epochs', 150))])
        cmd.extend(["--num_classes", str(params.get('num_classes', 5))])
        cmd.extend(["--save_dir", params.get('save_dir', './checkpoints')])

        if 'phi' in params:
            cmd.extend(["--phi", params['phi']])
        if 'batch_size' in params:
            cmd.extend(["--batch_size", str(params['batch_size'])])
        if 'input_size' in params:
            cmd.extend(["--input_size", str(params['input_size'][0]), str(params['input_size'][1])])
        if 'lr' in params:
            cmd.extend(["--lr", str(params['lr'])])
        if 'num_workers' in params:
            cmd.extend(["--num_workers", str(params['num_workers'])])

    else:  # mode == 2
        cmd = ["python", "train-max.py"]

        cmd.extend(["--epochs", str(params.get('epochs', 150))])
        cmd.extend(["--num_classes", str(params.get('num_classes', 5))])
        cmd.extend(["--save_dir", params.get('save_dir', './checkpoints')])

    return cmd


def confirm_and_run(cmd: list):
    """Show command and ask for confirmation."""
    print_section("READY TO START TRAINING")

    print("Command to execute:")
    print(f"  $ {' '.join(cmd)}\n")

    confirm = input("Start training now? (y/n) [y]: ").strip().lower()

    if confirm in ['', 'y', 'yes']:
        print("\n🚀 Starting training...\n")
        print("=" * 80)

        try:
            subprocess.run(cmd, check=True)
        except KeyboardInterrupt:
            print("\n\n⚠️  Training interrupted by user.")
        except subprocess.CalledProcessError as e:
            print(f"\n\n❌ Training failed with error code {e.returncode}")
        except FileNotFoundError:
            print("\n\n❌ Error: Training script not found!")
            print("   Make sure you're in the project root directory.")
    else:
        print("\n📋 To run training later, use:")
        print(f"  $ {' '.join(cmd)}")


def show_tips():
    """Show helpful tips."""
    print_section("💡 HELPFUL TIPS")

    print("Before training:")
    print("  • Ensure your dataset is in data/train and data/valid")
    print("  • Check that XML annotations are correct")
    print("  • Verify class names match your dataset")

    print("\nDuring training:")
    print("  • Monitor the first epoch for any errors")
    print("  • Check GPU usage with: watch -n 1 nvidia-smi")
    print("  • Training can be stopped with Ctrl+C")

    print("\nAfter training:")
    print("  • Best model saved as: checkpoints/best_model.pth")
    print("  • Visualizations saved every 5 epochs")
    print("  • Check validation mAP to assess performance")


def main():
    """Main interactive script."""
    print_header("🌾 YOLOX COTTON DISEASE DETECTION - TRAINING LAUNCHER 🌾")

    # Detect hardware
    has_cuda, gpu_name, total_vram, free_vram = detect_hardware()
    show_hardware_info(has_cuda, gpu_name, total_vram, free_vram)

    if not has_cuda:
        print("\n⚠️  WARNING: No GPU detected!")
        print("   Training on CPU is extremely slow and not recommended.")
        cont = input("\nContinue anyway? (y/n) [n]: ").strip().lower()
        if cont not in ['y', 'yes']:
            print("\n👋 Exiting. Please ensure CUDA is properly installed.")
            sys.exit(0)

    # Explain modes
    explain_training_modes()

    # Show recommendation
    recommended = recommend_mode(has_cuda, free_vram)
    show_recommendation(recommended, free_vram)

    # Get user choice
    choice = get_user_choice(recommended)

    if choice == 0:
        print("\n👋 Exiting. Happy training!")
        sys.exit(0)

    # Get parameters
    params = get_training_parameters(choice)

    # Build command
    cmd = build_command(choice, params)

    # Show tips
    show_tips()

    # Confirm and run
    confirm_and_run(cmd)

    print_header("✅ TRAINING SESSION ENDED")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        print("   Please report this issue if it persists.")
        sys.exit(1)
