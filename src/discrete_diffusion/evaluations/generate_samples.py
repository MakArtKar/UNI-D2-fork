"""Script to generate samples from a trained checkpoint.

This script loads a trained model checkpoint and generates samples using the
configured sampler. The output is saved as a PyTorch tensor (.pt) which can be
used for evaluation (e.g. with generative_ppl.py).

Supports multi-GPU parallel generation by dividing samples across available GPUs.
Also supports CPU-only generation with device=cpu.
"""

import math

import hydra
import torch
import torch.multiprocessing as mp
import tqdm
from pathlib import Path
from omegaconf import OmegaConf
from torchvision.utils import make_grid
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from discrete_diffusion.data import get_tokenizer


def get_num_devices(cfg):
    """Determine the number of devices to use for generation."""
    # Check if devices is explicitly set in CLI config
    if cfg.get("devices", None) is not None:
        return cfg.devices
    
    # Default to 1 device
    return 1


def create_mask_visualization(trajectories, mask_id, sampler_name="Sampler", padding=4):
    """Create visualization of masking pattern across diffusion steps.
    
    Args:
        trajectories: Tensor of shape [num_steps, num_samples, seq_len].
        mask_id: Token ID used for masking.
        sampler_name: Name of the sampler for the title.
        padding: Padding between sample images in pixels.
        
    Returns:
        matplotlib Figure with the visualization.
    """
    num_steps, num_samples, seq_len = trajectories.shape
    
    # Create binary mask: 1 = masked (yellow), 0 = unmasked (purple)
    # Shape: [num_samples, num_steps, seq_len]
    is_masked = (trajectories == mask_id).permute(1, 0, 2).float()
    
    # Colors: yellow for masked (1), purple for unmasked (0)
    # Yellow: RGB(255, 255, 0), Purple: RGB(128, 0, 128)
    yellow = torch.tensor([1.0, 1.0, 0.0])  # Masked
    purple = torch.tensor([0.5, 0.0, 0.5])  # Unmasked
    
    # Create RGB images for each sample
    # Shape per sample: [3, num_steps, seq_len] (C, H, W) where H=steps, W=positions
    images = []
    for sample_idx in range(num_samples):
        mask = is_masked[sample_idx]  # [num_steps, seq_len]
        
        # Create RGB image
        img = torch.zeros(3, num_steps, seq_len)
        for c in range(3):
            img[c] = mask * yellow[c] + (1 - mask) * purple[c]
        
        images.append(img)
    
    # Stack into batch: [num_samples, 3, num_steps, seq_len]
    images = torch.stack(images, dim=0)
    
    # Use make_grid to combine with white padding
    # nrow = sqrt(num_samples) for a square-ish grid layout
    nrow = int(math.sqrt(num_samples))
    ncol = math.ceil(num_samples / nrow)
    # pad_value=1.0 creates white padding (both horizontal and vertical)
    grid = make_grid(images, nrow=nrow, padding=padding, pad_value=1.0)
    
    # Convert to numpy for matplotlib
    # grid shape: [3, H, W], values in [0, 1]
    grid_np = grid.permute(1, 2, 0).numpy()
    
    # Calculate cell dimensions (including padding)
    cell_width = seq_len + padding
    cell_height = num_steps + padding
    
    # Create matplotlib figure with axes
    # Adjust figsize for grid layout
    fig_width = max(12, nrow * seq_len / 40)
    fig_height = max(8, ncol * num_steps / 25)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    ax.imshow(grid_np, aspect='auto')
    ax.set_xlabel('Position', fontsize=48, fontweight='bold')
    ax.set_ylabel('Step', fontsize=48, fontweight='bold')
    ax.set_title(f'{sampler_name} - Decoding Order Visualization', fontsize=59, fontweight='bold')
    
    # Create cycled tick labels for X-axis (position)
    # Each cell starts from 0
    x_tick_interval = max(1, seq_len // 8)  # ~8 ticks per cell
    x_ticks = []
    x_labels = []
    for col in range(nrow):
        cell_start = col * cell_width + padding
        for pos in range(0, seq_len, x_tick_interval):
            x_ticks.append(cell_start + pos)
            x_labels.append(str(pos))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=32)
    
    # Create cycled tick labels for Y-axis (step)
    # Each cell starts from 0
    y_tick_interval = max(1, num_steps // 8)  # ~8 ticks per cell
    y_ticks = []
    y_labels = []
    for row in range(ncol):
        cell_start = row * cell_height + padding
        for step in range(0, num_steps, y_tick_interval):
            y_ticks.append(cell_start + step)
            y_labels.append(str(step))
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=32)
    
    # Create legend with bigger font
    yellow_patch = mpatches.Patch(color=(1.0, 1.0, 0.0), label='Masked')
    purple_patch = mpatches.Patch(color=(0.5, 0.0, 0.5), label='Unmasked')
    ax.legend(handles=[yellow_patch, purple_patch], loc='upper right', fontsize=43)
    
    plt.tight_layout()
    
    return fig


def save_trajectories(trajectories, samples_path, tokenizer, num_samples, mask_id=None, sampler_name="Sampler"):
    """Save trajectory data in tensor, text, and visualization formats.
    
    Args:
        trajectories: Tensor of shape [num_steps, num_samples, seq_len] containing
                     token ids at each diffusion step.
        samples_path: Path to the samples file (used to derive trajectory path).
        tokenizer: Tokenizer for decoding tokens to text.
        num_samples: Number of samples to save in text format.
        mask_id: Token ID used for masking (for visualization).
        sampler_name: Name of the sampler for visualization title.
    """
    print("Saving trajectories...")
    
    # Compute trajectory path (similar to metrics path in generative_ppl.py)
    traj_path = Path(samples_path.replace('samples/', 'trajectories/', 1))
    traj_path = Path(hydra.utils.to_absolute_path(str(traj_path)))
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save as .pt file
    traj_pt_path = traj_path.with_suffix('.pt')
    torch.save(trajectories, traj_pt_path)
    print(f"Saved trajectories tensor to {traj_pt_path}")
    print(f"  Shape: {trajectories.shape} (steps, samples, sequence_length)")
    
    # Save as text file - decode each step for each sample
    traj_txt_path = traj_path.with_suffix('.txt')
    num_steps_total, num_samples_traj, seq_len = trajectories.shape
    
    with open(traj_txt_path, 'w', encoding='utf-8') as f:
        for sample_idx in range(min(num_samples_traj, num_samples)):
            f.write(f"{'='*80}\n")
            f.write(f"SAMPLE {sample_idx}\n")
            f.write(f"{'='*80}\n\n")
            
            for step_idx in range(num_steps_total):
                step_tokens = trajectories[step_idx, sample_idx, :].unsqueeze(0)
                step_text = tokenizer.batch_decode(step_tokens, skip_special_tokens=False)[0]
                
                # Show step number (0 = initial, last = final)
                if step_idx == 0:
                    step_label = "Initial (masked)"
                elif step_idx == num_steps_total - 1:
                    step_label = "Final (denoised)"
                else:
                    step_label = f"Step {step_idx}"
                
                f.write(f"--- {step_label} ---\n")
                f.write(f"{step_text}\n\n")
            
            f.write("\n")
    
    print(f"Saved trajectory texts to {traj_txt_path}")
    
    # Save visualization if mask_id is provided
    if mask_id is not None:
        traj_img_path = traj_path.with_suffix('.png')
        fig = create_mask_visualization(trajectories, mask_id, sampler_name=sampler_name)
        fig.savefig(traj_img_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved trajectory visualization to {traj_img_path}")


def worker_generate(rank, num_devices, samples_per_device, remainder, cfg, checkpoint_path, 
                    model_config, tokenizer, return_dict, use_cpu=False, return_trajectory=False):
    """Worker function for generating samples on a specific GPU or CPU.
    
    Args:
        rank: Device rank (0 to num_devices-1)
        num_devices: Total number of devices
        samples_per_device: Base number of samples per device
        remainder: Extra samples to distribute among first devices
        cfg: Generation config
        checkpoint_path: Path to model checkpoint
        model_config: Model configuration
        tokenizer: Tokenizer instance
        return_dict: Shared dict for returning results
        use_cpu: Whether to use CPU instead of GPU
        return_trajectory: Whether to return intermediate states
    """
    try:
        if use_cpu:
            device = torch.device("cpu")
            print(f"[CPU] Loading model and generating samples...")
        else:
            device = torch.device(f"cuda:{rank}")
            torch.cuda.set_device(device)
            # Log which physical GPU we're using
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(rank)
                print(f"[Device {rank}] Using GPU: {gpu_name}")
        
        torch.set_float32_matmul_precision('high')
        torch.set_grad_enabled(False)
        
        # Calculate samples for this worker
        # Distribute remainder among first 'remainder' devices
        num_samples_this_device = samples_per_device + (1 if rank < remainder else 0)
        
        if num_samples_this_device == 0:
            return_dict[rank] = torch.tensor([], dtype=torch.long)
            return
        
        device_label = "CPU" if use_cpu else f"Device {rank}"
        print(f"[{device_label}] Loading model and generating {num_samples_this_device} samples...")
        
        # Load model on this device
        algo_target = model_config.algo._target_
        algo_cls = hydra.utils.get_class(algo_target)
        
        model = algo_cls.load_from_checkpoint(
            checkpoint_path, 
            config=model_config, 
            tokenizer=tokenizer,
            map_location=device,
            strict=False
        )
        
        model.to(device)
        model.eval()
        
        # Activate EMA weights for generation
        if hasattr(model, '_eval_mode'):
            model._eval_mode()
        
        if cfg.torch_compile and not use_cpu:
            model = torch.compile(model)
        
        batch_size = cfg.batch_size
        num_steps = cfg.num_steps
        
        all_samples = []
        all_trajectories = [] if return_trajectory else None
        
        # Generate samples for this device
        samples_generated = 0
        while samples_generated < num_samples_this_device:
            current_batch_size = min(batch_size, num_samples_this_device - samples_generated)
            
            result = model.generate_samples(
                num_samples=current_batch_size,
                num_steps=num_steps,
                return_trajectory=return_trajectory
            )
            
            if return_trajectory:
                samples, trajectory = result
                all_samples.append(samples.detach().cpu())
                # trajectory is a list of tensors, each [batch, length]
                # Convert to [num_steps+2, batch, length] then move to CPU
                traj_tensor = torch.stack(trajectory, dim=0).detach().cpu()
                all_trajectories.append(traj_tensor)
            else:
                all_samples.append(result.detach().cpu())
            
            samples_generated += current_batch_size
        
        result = torch.cat(all_samples, dim=0) if all_samples else torch.tensor([], dtype=torch.long)
        print(f"[{device_label}] Generated {len(result)} samples")
        
        if return_trajectory:
            # Concatenate trajectories along batch dimension
            # Each trajectory is [num_steps+2, batch, length], concat on dim=1
            combined_trajectories = torch.cat(all_trajectories, dim=1) if all_trajectories else None
            return_dict[rank] = {'samples': result, 'trajectories': combined_trajectories}
        else:
            return_dict[rank] = result
        
    except Exception as e:
        device_label = "CPU" if use_cpu else f"Device {rank}"
        print(f"[{device_label}] Error: {e}")
        import traceback
        traceback.print_exc()
        return_dict[rank] = None


@hydra.main(config_path="../../../configs", config_name="generate_samples", version_base="1.3")
def main(cfg):
    # Check if we should use CPU
    use_cpu = cfg.get("device", "cuda") == "cpu"
    
    if use_cpu:
        print("Running in CPU mode...")
    elif not torch.cuda.is_available():
        print("CUDA is not available. Falling back to CPU mode.")
        use_cpu = True
    
    torch.set_float32_matmul_precision('high')

    print(f"Loading checkpoint from {cfg.checkpoint_path}")
    checkpoint_path = hydra.utils.to_absolute_path(cfg.checkpoint_path)
    
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # Load checkpoint to extract config (on CPU first)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Extract config from hyper_parameters
    if 'hyper_parameters' not in ckpt:
        raise ValueError("Checkpoint does not contain 'hyper_parameters'. Cannot load config.")
    
    if 'config' not in ckpt['hyper_parameters']:
         raise ValueError("Checkpoint hyper_parameters does not contain 'config'.")
         
    model_config = ckpt['hyper_parameters']['config']
    # Ensure it's an OmegaConf object
    if not isinstance(model_config, (dict, list, OmegaConf.get_type("DictConfig"), OmegaConf.get_type("ListConfig"))):
         model_config = OmegaConf.create(model_config)
    
    # Override sampling config if provided in CLI arguments
    if cfg.get("sampling", None) is not None:
        OmegaConf.set_struct(model_config.sampling, False)
        model_config.sampling = OmegaConf.merge(model_config.sampling, cfg.sampling)
        OmegaConf.set_struct(model_config.sampling, True)
        print(f"Sampling config overridden: {cfg.sampling.sampler._target_}")
    
    # Get tokenizer - use the one from checkpoint to ensure compatibility
    print("Loading tokenizer...")
    if 'tokenizer' in ckpt['hyper_parameters']:
        print("Using tokenizer from checkpoint")
        tokenizer = ckpt['hyper_parameters']['tokenizer']
    else:
        print("Creating new tokenizer from config")
        tokenizer = get_tokenizer(model_config)
    
    # Determine number of devices
    num_devices = get_num_devices(cfg)
    
    if use_cpu:
        # CPU mode - single device only
        num_devices = 1
    else:
        available_devices = torch.cuda.device_count()
        print(f"Available CUDA devices: {available_devices}")
        
        if num_devices > available_devices:
            print(f"Warning: Requested {num_devices} devices but only {available_devices} available. Using {available_devices}.")
            num_devices = available_devices
        
        # Ensure we have at least 1 device
        num_devices = max(1, num_devices)
    
    num_samples = cfg.num_samples
    
    # Calculate samples per device
    samples_per_device = num_samples // num_devices
    remainder = num_samples % num_devices
    
    print(f"Detected algorithm class: {hydra.utils.get_class(model_config.algo._target_).__name__}")
    
    if use_cpu:
        print(f"Generating {num_samples} samples on CPU")
    else:
        print(f"Generating {num_samples} samples across {num_devices} GPU(s)")
        print(f"  - Base samples per GPU: {samples_per_device}")
        if remainder > 0:
            print(f"  - Extra samples on first {remainder} GPU(s): +1 each")
    
    # Free memory from checkpoint loading
    del ckpt
    
    # Check if trajectory tracking is enabled
    return_trajectory = cfg.get("return_trajectory", False)
    if return_trajectory:
        print("Trajectory tracking enabled - will save intermediate states")
    
    if num_devices == 1:
        # Single device mode - no multiprocessing needed
        if use_cpu:
            print("Running in single-CPU mode...")
        else:
            print("Running in single-GPU mode...")
        manager = mp.Manager()
        return_dict = manager.dict()
        worker_generate(0, 1, samples_per_device, remainder, cfg, checkpoint_path, 
                       model_config, tokenizer, return_dict, use_cpu=use_cpu,
                       return_trajectory=return_trajectory)
        all_results = [return_dict[0]]
    else:
        # Multi-GPU mode using multiprocessing
        print("Spawning worker processes...")
        mp.set_start_method('spawn', force=True)
        
        manager = mp.Manager()
        return_dict = manager.dict()
        
        processes = []
        for rank in range(num_devices):
            p = mp.Process(
                target=worker_generate,
                args=(rank, num_devices, samples_per_device, remainder, cfg, 
                      checkpoint_path, model_config, tokenizer, return_dict, use_cpu,
                      return_trajectory)
            )
            p.start()
            processes.append(p)
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        # Collect results in order
        all_results = []
        for rank in range(num_devices):
            if return_dict.get(rank) is None:
                raise RuntimeError(f"Worker on device {rank} failed to generate samples")
            all_results.append(return_dict[rank])
    
    # Concatenate all samples and trajectories
    if return_trajectory:
        # Extract samples and trajectories from results
        all_samples = []
        all_trajectories = []
        for result in all_results:
            if result is not None and isinstance(result, dict):
                if result['samples'] is not None and len(result['samples']) > 0:
                    all_samples.append(result['samples'])
                if result['trajectories'] is not None:
                    all_trajectories.append(result['trajectories'])
        
        if all_samples:
            all_samples = torch.cat(all_samples, dim=0)
        else:
            all_samples = torch.tensor([], dtype=torch.long)
        
        if all_trajectories:
            # Each trajectory is [num_steps+2, batch, length], concat on dim=1
            all_trajectories = torch.cat(all_trajectories, dim=1)
        else:
            all_trajectories = None
    else:
        # Simple case: results are just tensors
        all_samples = []
        for result in all_results:
            if result is not None and len(result) > 0:
                all_samples.append(result)
        
        if all_samples:
            all_samples = torch.cat(all_samples, dim=0)
        else:
            all_samples = torch.tensor([], dtype=torch.long)
        all_trajectories = None
    
    print(f"Total samples collected: {len(all_samples)}")
    
    # Verify we have exactly the requested number of samples
    if len(all_samples) != num_samples:
        print(f"Warning: Expected {num_samples} samples but got {len(all_samples)}")
    
    # Save samples
    out_path = Path(hydra.utils.to_absolute_path(cfg.samples_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(all_samples, out_path)
    print(f"Saved {len(all_samples)} samples to {out_path}")

    if cfg.get("save_text", False):
        print("Decoding samples to text...")
        texts = tokenizer.batch_decode(all_samples, skip_special_tokens=True)
        text_path = out_path.with_suffix('.txt')
        with open(text_path, 'w', encoding='utf-8') as f:
            for i, text in enumerate(texts):
                f.write(f"Sample {i}:\n{text}\n{'-'*80}\n")
        print(f"Saved text samples to {text_path}")

    # Save trajectories if enabled
    if return_trajectory and all_trajectories is not None:
        # Get mask_id from tokenizer for visualization
        mask_id = getattr(tokenizer, 'mask_token_id', None)
        
        # Extract sampler name from config (just the class name, no params)
        sampler_target = getattr(model_config.sampling.sampler, '_target_', 'Sampler')
        sampler_name = sampler_target.split('.')[-1]  # Get class name only
        
        save_trajectories(all_trajectories, cfg.samples_path, tokenizer, num_samples, 
                         mask_id=mask_id, sampler_name=sampler_name)


if __name__ == "__main__":
    main()
