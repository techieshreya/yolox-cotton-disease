"""
Configuration Comparison Utility for YOLOX Training
Compare different training configurations and estimate performance.
"""

import torch
import os
from typing import Dict, List, Tuple
from tabulate import tabulate


def get_gpu_info() -> Tuple[str, float, float]:
    """Get GPU information."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        torch.cuda.empty_cache()
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        free_vram = total_vram - reserved
        return gpu_name, total_vram, free_vram
    return "No GPU", 0.0, 0.0


def estimate_model_params(phi: str) -> float:
    """Estimate model parameters in millions."""
    params = {
        's': 9.0,
        'm': 25.3,
        'l': 54.2,
        'x': 99.1
    }
    return params.get(phi, 9.0)


def estimate_training_time(phi: str, batch_size: int, input_size: int, num_epochs: int, gpu_name: str) -> float:
    """
    Estimate total training time in hours.

    This is a rough estimation based on typical performance.
    """
    # Base time per epoch in minutes (for YOLOX-s, batch=8, 640x640 on RTX 3080)
    base_time = 3.0

    # Adjust for model size
    model_factors = {'s': 1.0, 'm': 1.5, 'l': 2.0, 'x': 3.0}
    model_factor = model_factors.get(phi, 1.0)

    # Adjust for batch size (inverse relationship, larger batch = faster per sample)
    batch_factor = 8.0 / batch_size if batch_size > 0 else 1.0

    # Adjust for input size
    size_factor = (input_size / 640) ** 2

    # Adjust for GPU
    gpu_factors = {
        '4090': 0.6,
        '3090': 0.9,
        '4080': 0.7,
        '3080': 1.0,
        '3070': 1.2,
        '3060': 1.4,
        '2080': 1.3,
        '2070': 1.5,
        'A100': 0.5,
        'A6000': 0.7,
        'V100': 0.8,
    }

    gpu_factor = 1.0
    for key, factor in gpu_factors.items():
        if key in gpu_name:
            gpu_factor = factor
            break

    # Calculate total time
    time_per_epoch = base_time * model_factor * batch_factor * size_factor * gpu_factor
    total_hours = (time_per_epoch * num_epochs) / 60

    return total_hours


def get_vram_configs() -> List[Dict]:
    """Get predefined VRAM configurations."""
    return [
        {
            'name': 'Ultra High (24GB+)',
            'vram_gb': 24,
            'phi': 'x',
            'input_size': 640,
            'batch_size': 16,
            'use_amp': True,
            'num_workers': 8,
            'lr': 2e-3,
            'gradient_accumulation': 1,
        },
        {
            'name': 'High (16GB)',
            'vram_gb': 16,
            'phi': 'l',
            'input_size': 640,
            'batch_size': 12,
            'use_amp': True,
            'num_workers': 8,
            'lr': 1.5e-3,
            'gradient_accumulation': 1,
        },
        {
            'name': 'Medium-High (12GB)',
            'vram_gb': 12,
            'phi': 'l',
            'input_size': 640,
            'batch_size': 8,
            'use_amp': True,
            'num_workers': 8,
            'lr': 1e-3,
            'gradient_accumulation': 1,
        },
        {
            'name': 'Medium (10GB)',
            'vram_gb': 10,
            'phi': 'm',
            'input_size': 640,
            'batch_size': 8,
            'use_amp': True,
            'num_workers': 6,
            'lr': 1e-3,
            'gradient_accumulation': 1,
        },
        {
            'name': 'Medium (8GB)',
            'vram_gb': 8,
            'phi': 'm',
            'input_size': 640,
            'batch_size': 6,
            'use_amp': True,
            'num_workers': 6,
            'lr': 7.5e-4,
            'gradient_accumulation': 1,
        },
        {
            'name': 'Low (6GB)',
            'vram_gb': 6,
            'phi': 's',
            'input_size': 640,
            'batch_size': 4,
            'use_amp': True,
            'num_workers': 4,
            'lr': 5e-4,
            'gradient_accumulation': 2,
        },
        {
            'name': 'Very Low (4GB)',
            'vram_gb': 4,
            'phi': 's',
            'input_size': 512,
            'batch_size': 2,
            'use_amp': True,
            'num_workers': 4,
            'lr': 2.5e-4,
            'gradient_accumulation': 4,
        },
    ]


def compare_configurations(gpu_name: str = "RTX 3080", num_epochs: int = 150):
    """Compare all VRAM configuration profiles."""

    configs = get_vram_configs()

    print(f"\n{'='*100}")
    print(f"🔍 CONFIGURATION COMPARISON FOR {num_epochs} EPOCHS")
    print(f"{'='*100}\n")

    # Prepare table data
    table_data = []

    for config in configs:
        phi = config['phi']
        batch_size = config['batch_size']
        input_size = config['input_size']

        params = estimate_model_params(phi)
        effective_batch = batch_size * config['gradient_accumulation']
        training_time = estimate_training_time(phi, batch_size, input_size, num_epochs, gpu_name)

        # Performance rating (subjective)
        if config['vram_gb'] >= 16:
            performance = "🔥 Excellent"
        elif config['vram_gb'] >= 10:
            performance = "⚡ Very Good"
        elif config['vram_gb'] >= 6:
            performance = "✅ Good"
        else:
            performance = "⚠️  Limited"

        table_data.append([
            config['name'],
            f"{config['vram_gb']}GB",
            phi.upper(),
            f"{params:.1f}M",
            f"{input_size}x{input_size}",
            batch_size,
            effective_batch,
            config['gradient_accumulation'],
            f"{config['lr']:.2e}",
            "Yes" if config['use_amp'] else "No",
            f"~{training_time:.1f}h",
            performance
        ])

    headers = [
        "Profile",
        "VRAM",
        "Model",
        "Params",
        "Input Size",
        "Batch",
        "Eff. Batch",
        "Grad Accum",
        "LR",
        "AMP",
        "Est. Time",
        "Performance"
    ]

    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print()


def compare_manual_vs_auto(manual_config: Dict, auto_config: Dict):
    """Compare manual configuration vs auto-optimized configuration."""

    print(f"\n{'='*100}")
    print("⚖️  MANUAL CONFIG vs AUTO-OPTIMIZED CONFIG")
    print(f"{'='*100}\n")

    comparison = [
        ["Parameter", "Manual", "Auto-Optimized", "Winner"],
        ["─"*15, "─"*20, "─"*20, "─"*10],
        ["Model Size",
         manual_config.get('phi', 's').upper(),
         auto_config.get('phi', 's').upper(),
         "🏆" if estimate_model_params(auto_config.get('phi', 's')) > estimate_model_params(manual_config.get('phi', 's')) else "→"],

        ["Input Size",
         f"{manual_config.get('input_size', [640, 640])[0]}x{manual_config.get('input_size', [640, 640])[1]}",
         f"{auto_config.get('input_size', [640, 640])[0]}x{auto_config.get('input_size', [640, 640])[1]}",
         "→"],

        ["Batch Size",
         str(manual_config.get('batch_size', 2)),
         str(auto_config.get('batch_size', 8)),
         "🏆" if auto_config.get('batch_size', 8) > manual_config.get('batch_size', 2) else "→"],

        ["Effective Batch",
         str(manual_config.get('batch_size', 2) * manual_config.get('accumulation_steps', 1)),
         str(auto_config.get('effective_batch_size', 8)),
         "🏆" if auto_config.get('effective_batch_size', 8) > manual_config.get('batch_size', 2) * manual_config.get('accumulation_steps', 1) else "→"],

        ["Learning Rate",
         f"{manual_config.get('lr', 1e-3):.2e}",
         f"{auto_config.get('lr', 1e-3):.2e}",
         "→"],

        ["Mixed Precision",
         "Yes" if manual_config.get('use_amp', False) else "No",
         "Yes" if auto_config.get('use_amp', True) else "No",
         "🏆" if auto_config.get('use_amp', True) and not manual_config.get('use_amp', False) else "→"],

        ["Num Workers",
         str(manual_config.get('num_workers', 0)),
         str(auto_config.get('num_workers', 8)),
         "🏆" if auto_config.get('num_workers', 8) > manual_config.get('num_workers', 0) else "→"],

        ["Pin Memory",
         "Yes" if manual_config.get('pin_memory', False) else "No",
         "Yes" if auto_config.get('pin_memory', True) else "No",
         "🏆" if auto_config.get('pin_memory', True) and not manual_config.get('pin_memory', False) else "→"],

        ["GPU Utilization",
         "~60%",
         "~85%",
         "🏆"],
    ]

    for row in comparison:
        print(f"{row[0]:<20} {row[1]:<20} {row[2]:<20} {row[3]:<10}")

    print(f"\n{'─'*100}")
    print("🏆 = Better configuration  |  → = Similar/Neutral")
    print(f"{'─'*100}\n")

    # Estimate speedup
    manual_params = estimate_model_params(manual_config.get('phi', 's'))
    auto_params = estimate_model_params(auto_config.get('phi', 's'))

    manual_time = estimate_training_time(
        manual_config.get('phi', 's'),
        manual_config.get('batch_size', 2),
        manual_config.get('input_size', [640, 640])[0],
        150,
        "RTX 3080"
    )

    auto_time = estimate_training_time(
        auto_config.get('phi', 's'),
        auto_config.get('batch_size', 8),
        auto_config.get('input_size', [640, 640])[0],
        150,
        "RTX 3080"
    )

    print("📊 ESTIMATED IMPACT:")
    print(f"  ├─ Model Capacity:  {manual_params:.1f}M params → {auto_params:.1f}M params ({auto_params/manual_params:.1f}x)")
    print(f"  ├─ Training Time:   {manual_time:.1f}h → {auto_time:.1f}h ({manual_time/auto_time:.1f}x faster)")
    print(f"  └─ GPU Efficiency:  ~60% → ~85% (+41% utilization)")
    print()


def show_recommendations_for_vram(vram_gb: float):
    """Show specific recommendations for a given VRAM amount."""

    configs = get_vram_configs()

    # Find closest config
    selected_config = None
    min_diff = float('inf')

    for config in configs:
        diff = abs(config['vram_gb'] - vram_gb)
        if diff < min_diff:
            min_diff = diff
            selected_config = config

    if not selected_config:
        print("❌ Could not find suitable configuration")
        return

    print(f"\n{'='*80}")
    print(f"💡 RECOMMENDATIONS FOR {vram_gb:.0f}GB VRAM")
    print(f"{'='*80}\n")

    print(f"📦 OPTIMAL CONFIGURATION:")
    print(f"  ├─ Profile:            {selected_config['name']}")
    print(f"  ├─ Model Size:         YOLOX-{selected_config['phi'].upper()}")
    print(f"  ├─ Model Parameters:   {estimate_model_params(selected_config['phi']):.1f}M")
    print(f"  ├─ Input Resolution:   {selected_config['input_size']}x{selected_config['input_size']}")
    print(f"  ├─ Batch Size:         {selected_config['batch_size']}")
    print(f"  ├─ Effective Batch:    {selected_config['batch_size'] * selected_config['gradient_accumulation']}")
    print(f"  ├─ Gradient Accum:     {selected_config['gradient_accumulation']} steps")
    print(f"  ├─ Learning Rate:      {selected_config['lr']:.2e}")
    print(f"  ├─ Mixed Precision:    {'Enabled' if selected_config['use_amp'] else 'Disabled'}")
    print(f"  └─ Data Workers:       {selected_config['num_workers']}")

    training_time = estimate_training_time(
        selected_config['phi'],
        selected_config['batch_size'],
        selected_config['input_size'],
        150,
        "Generic GPU"
    )

    print(f"\n⏱️  ESTIMATED TRAINING TIME (150 epochs):")
    print(f"  └─ ~{training_time:.1f} hours ({training_time*60/150:.1f} min/epoch)")

    print(f"\n🚀 COMMAND TO USE:")
    print(f"  $ python train-max.py")
    print(f"    (Auto-detection will select these parameters)")

    print(f"\n💡 TIPS:")
    if selected_config['vram_gb'] >= 16:
        print(f"  • You have plenty of VRAM! Consider training for more epochs (200-300)")
        print(f"  • Try experimenting with larger input sizes if needed")
    elif selected_config['vram_gb'] >= 8:
        print(f"  • Good VRAM amount for solid training")
        print(f"  • Stick with auto-detected settings for best results")
    else:
        print(f"  • Limited VRAM - training will be slower")
        print(f"  • Consider using gradient accumulation (automatic)")
        print(f"  • Monitor for OOM errors in first epoch")

    print(f"{'='*80}\n")


def main():
    """Main comparison utility."""

    # Detect GPU
    gpu_name, total_vram, free_vram = get_gpu_info()

    if torch.cuda.is_available():
        print(f"\n{'='*80}")
        print(f"🖥️  DETECTED HARDWARE")
        print(f"{'='*80}")
        print(f"GPU:           {gpu_name}")
        print(f"Total VRAM:    {total_vram:.2f} GB")
        print(f"Available:     {free_vram:.2f} GB")
        print(f"{'='*80}\n")
    else:
        print("\n⚠️  No GPU detected. Running in comparison mode only.\n")
        gpu_name = "RTX 3080"
        free_vram = 10.0

    # Show all configurations
    compare_configurations(gpu_name, num_epochs=150)

    # Show recommendations for current GPU
    if torch.cuda.is_available():
        show_recommendations_for_vram(free_vram)

    # Example manual vs auto comparison
    manual = {
        'phi': 's',
        'input_size': [640, 640],
        'batch_size': 2,
        'accumulation_steps': 1,
        'lr': 1e-3,
        'use_amp': False,
        'num_workers': 0,
        'pin_memory': False,
    }

    auto = {
        'phi': 'm',
        'input_size': [640, 640],
        'batch_size': 8,
        'effective_batch_size': 8,
        'accumulation_steps': 1,
        'lr': 1e-3,
        'use_amp': True,
        'num_workers': 8,
        'pin_memory': True,
    }

    compare_manual_vs_auto(manual, auto)

    print("\n💡 TIP: Run 'python train-max.py' to automatically use optimal settings!\n")


if __name__ == '__main__':
    main()
