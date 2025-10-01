"""
VRAM Monitoring Utilities for YOLOX Training
Provides real-time GPU memory tracking and optimization recommendations.
"""

import torch
import time
import os
import psutil
from collections import deque
from typing import Dict, List, Optional, Tuple
import json


class VRAMMonitor:
    """
    Monitor GPU VRAM usage during training and provide optimization insights.
    """

    def __init__(self, device_id: int = 0, history_size: int = 100):
        """
        Initialize VRAM monitor.

        Args:
            device_id: CUDA device ID to monitor
            history_size: Number of measurements to keep in history
        """
        self.device_id = device_id
        self.history_size = history_size

        # Check if CUDA is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. VRAM monitoring requires a GPU.")

        # Memory history
        self.allocated_history = deque(maxlen=history_size)
        self.reserved_history = deque(maxlen=history_size)
        self.timestamps = deque(maxlen=history_size)

        # Peak memory tracking
        self.peak_allocated = 0.0
        self.peak_reserved = 0.0

        # Get GPU properties
        self.gpu_properties = torch.cuda.get_device_properties(device_id)
        self.total_memory = self.gpu_properties.total_memory / (1024**3)  # GB

        # Monitoring state
        self.is_monitoring = False
        self.start_time = None

    def start_monitoring(self):
        """Start monitoring VRAM usage."""
        self.is_monitoring = True
        self.start_time = time.time()
        torch.cuda.reset_peak_memory_stats(self.device_id)
        torch.cuda.empty_cache()
        print(f"✅ VRAM monitoring started for GPU {self.device_id}")

    def stop_monitoring(self):
        """Stop monitoring VRAM usage."""
        self.is_monitoring = False
        print(f"⏹️  VRAM monitoring stopped")

    def update(self):
        """Update memory statistics."""
        if not self.is_monitoring:
            return

        allocated = torch.cuda.memory_allocated(self.device_id) / (1024**3)
        reserved = torch.cuda.memory_reserved(self.device_id) / (1024**3)

        self.allocated_history.append(allocated)
        self.reserved_history.append(reserved)
        self.timestamps.append(time.time())

        # Update peaks
        self.peak_allocated = max(self.peak_allocated, allocated)
        self.peak_reserved = max(self.peak_reserved, reserved)

    def get_current_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        allocated = torch.cuda.memory_allocated(self.device_id) / (1024**3)
        reserved = torch.cuda.memory_reserved(self.device_id) / (1024**3)
        free = self.total_memory - reserved

        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'free_gb': free,
            'total_gb': self.total_memory,
            'allocated_percent': (allocated / self.total_memory) * 100,
            'reserved_percent': (reserved / self.total_memory) * 100,
        }

    def get_peak_stats(self) -> Dict[str, float]:
        """Get peak memory statistics."""
        peak_allocated = torch.cuda.max_memory_allocated(self.device_id) / (1024**3)
        peak_reserved = torch.cuda.max_memory_reserved(self.device_id) / (1024**3)

        return {
            'peak_allocated_gb': peak_allocated,
            'peak_reserved_gb': peak_reserved,
            'peak_allocated_percent': (peak_allocated / self.total_memory) * 100,
            'peak_reserved_percent': (peak_reserved / self.total_memory) * 100,
        }

    def get_average_stats(self) -> Dict[str, float]:
        """Get average memory usage from history."""
        if not self.allocated_history:
            return {'avg_allocated_gb': 0.0, 'avg_reserved_gb': 0.0}

        avg_allocated = sum(self.allocated_history) / len(self.allocated_history)
        avg_reserved = sum(self.reserved_history) / len(self.reserved_history)

        return {
            'avg_allocated_gb': avg_allocated,
            'avg_reserved_gb': avg_reserved,
            'avg_allocated_percent': (avg_allocated / self.total_memory) * 100,
            'avg_reserved_percent': (avg_reserved / self.total_memory) * 100,
        }

    def print_summary(self):
        """Print a comprehensive summary of memory usage."""
        current = self.get_current_stats()
        peak = self.get_peak_stats()
        avg = self.get_average_stats()

        print(f"\n{'='*70}")
        print(f"🔍 VRAM USAGE SUMMARY - GPU {self.device_id}: {self.gpu_properties.name}")
        print(f"{'='*70}")

        print(f"\n📊 CURRENT USAGE:")
        print(f"  ├─ Allocated:  {current['allocated_gb']:.2f} GB ({current['allocated_percent']:.1f}%)")
        print(f"  ├─ Reserved:   {current['reserved_gb']:.2f} GB ({current['reserved_percent']:.1f}%)")
        print(f"  ├─ Free:       {current['free_gb']:.2f} GB")
        print(f"  └─ Total:      {current['total_gb']:.2f} GB")

        print(f"\n🔝 PEAK USAGE:")
        print(f"  ├─ Peak Allocated: {peak['peak_allocated_gb']:.2f} GB ({peak['peak_allocated_percent']:.1f}%)")
        print(f"  └─ Peak Reserved:  {peak['peak_reserved_gb']:.2f} GB ({peak['peak_reserved_percent']:.1f}%)")

        print(f"\n📈 AVERAGE USAGE:")
        print(f"  ├─ Avg Allocated: {avg['avg_allocated_gb']:.2f} GB ({avg['avg_allocated_percent']:.1f}%)")
        print(f"  └─ Avg Reserved:  {avg['avg_reserved_gb']:.2f} GB ({avg['avg_reserved_percent']:.1f}%)")

        # Recommendations
        self._print_recommendations(current, peak)

        print(f"{'='*70}\n")

    def _print_recommendations(self, current: Dict, peak: Dict):
        """Print optimization recommendations based on usage."""
        print(f"\n💡 RECOMMENDATIONS:")

        utilization = peak['peak_reserved_percent']

        if utilization < 50:
            print(f"  ⚠️  Low GPU utilization ({utilization:.1f}%)")
            print(f"     → Consider increasing batch size")
            print(f"     → Consider using a larger model (m, l, or x)")
            print(f"     → Consider increasing input resolution")
        elif utilization < 70:
            print(f"  ✅ Good GPU utilization ({utilization:.1f}%)")
            print(f"     → You might be able to increase batch size slightly")
        elif utilization < 85:
            print(f"  ✅ Optimal GPU utilization ({utilization:.1f}%)")
            print(f"     → Current settings are well-optimized")
        elif utilization < 95:
            print(f"  ⚠️  High GPU utilization ({utilization:.1f}%)")
            print(f"     → Risk of OOM errors")
            print(f"     → Consider slightly reducing batch size")
        else:
            print(f"  🚨 Critical GPU utilization ({utilization:.1f}%)")
            print(f"     → High risk of OOM errors!")
            print(f"     → Reduce batch size or use gradient accumulation")
            print(f"     → Consider using mixed precision training (AMP)")

    def save_stats(self, filepath: str):
        """Save memory statistics to JSON file."""
        stats = {
            'gpu_name': self.gpu_properties.name,
            'total_memory_gb': self.total_memory,
            'current': self.get_current_stats(),
            'peak': self.get_peak_stats(),
            'average': self.get_average_stats(),
            'monitoring_duration_seconds': time.time() - self.start_time if self.start_time else 0,
        }

        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"📝 Memory stats saved to {filepath}")


def get_model_memory_footprint(model: torch.nn.Module, input_size: Tuple[int, int, int, int]) -> Dict[str, float]:
    """
    Estimate the memory footprint of a model.

    Args:
        model: PyTorch model
        input_size: Input tensor size (batch, channels, height, width)

    Returns:
        Dictionary with memory breakdown
    """
    # Count parameters
    param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**3)
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers()) / (1024**3)

    # Estimate gradients (same size as parameters)
    grad_size = param_size

    # Estimate optimizer state (AdamW has 2 states per parameter)
    optimizer_size = param_size * 2

    # Estimate activations (rough estimate)
    batch_size = input_size[0]
    input_memory = (input_size[1] * input_size[2] * input_size[3] * 4 * batch_size) / (1024**3)
    # Activations are roughly 3-4x input size for deep networks
    activation_size = input_memory * 3.5

    total = param_size + buffer_size + grad_size + optimizer_size + activation_size

    return {
        'parameters_gb': param_size,
        'buffers_gb': buffer_size,
        'gradients_gb': grad_size,
        'optimizer_state_gb': optimizer_size,
        'activations_estimate_gb': activation_size,
        'total_estimate_gb': total,
    }


def print_memory_breakdown(model: torch.nn.Module, input_size: Tuple[int, int, int, int]):
    """Print detailed memory breakdown for a model."""
    breakdown = get_model_memory_footprint(model, input_size)

    print(f"\n{'='*60}")
    print("💾 MODEL MEMORY BREAKDOWN")
    print(f"{'='*60}")
    print(f"Parameters:       {breakdown['parameters_gb']:.3f} GB")
    print(f"Buffers:          {breakdown['buffers_gb']:.3f} GB")
    print(f"Gradients:        {breakdown['gradients_gb']:.3f} GB")
    print(f"Optimizer State:  {breakdown['optimizer_state_gb']:.3f} GB")
    print(f"Activations:      {breakdown['activations_estimate_gb']:.3f} GB (estimated)")
    print(f"{'-'*60}")
    print(f"TOTAL ESTIMATE:   {breakdown['total_estimate_gb']:.3f} GB")
    print(f"{'='*60}\n")


def recommend_batch_size(model: torch.nn.Module,
                         input_size: Tuple[int, int, int],
                         available_memory_gb: float,
                         safety_margin: float = 0.2) -> int:
    """
    Recommend optimal batch size based on available memory.

    Args:
        model: PyTorch model
        input_size: Input size (channels, height, width)
        available_memory_gb: Available GPU memory in GB
        safety_margin: Safety margin (0.2 = 20% buffer)

    Returns:
        Recommended batch size
    """
    # Get memory for batch size 1
    memory_bs1 = get_model_memory_footprint(model, (1, *input_size))
    total_bs1 = memory_bs1['total_estimate_gb']

    # Memory without batch-dependent components
    fixed_memory = (memory_bs1['parameters_gb'] +
                   memory_bs1['buffers_gb'] +
                   memory_bs1['gradients_gb'] +
                   memory_bs1['optimizer_state_gb'])

    # Memory per batch (activations + input)
    per_batch_memory = total_bs1 - fixed_memory

    # Calculate max batch size with safety margin
    usable_memory = available_memory_gb * (1 - safety_margin)
    max_batch_size = int((usable_memory - fixed_memory) / per_batch_memory)

    return max(1, max_batch_size)


def optimize_for_gpu():
    """Apply general GPU optimizations."""
    if torch.cuda.is_available():
        # Enable TF32 on Ampere GPUs for faster training
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Enable cuDNN benchmarking for faster training (if input sizes are fixed)
        torch.backends.cudnn.benchmark = True

        # Set memory allocator settings for better memory management
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

        print("⚡ GPU optimizations enabled:")
        print("  ├─ TF32: Enabled")
        print("  ├─ cuDNN Benchmark: Enabled")
        print("  └─ Memory Allocator: Optimized")
    else:
        print("⚠️  No GPU available for optimization")


if __name__ == '__main__':
    # Example usage
    if torch.cuda.is_available():
        print("Testing VRAM Monitor...\n")

        monitor = VRAMMonitor()
        monitor.start_monitoring()

        # Simulate some operations
        for i in range(5):
            # Allocate some tensors
            x = torch.randn(10, 3, 640, 640, device='cuda')
            y = torch.randn(10, 3, 640, 640, device='cuda')
            z = x @ y.transpose(-2, -1)

            monitor.update()
            time.sleep(0.5)

            del x, y, z

        monitor.stop_monitoring()
        monitor.print_summary()
    else:
        print("❌ CUDA not available. Cannot test VRAM monitor.")
