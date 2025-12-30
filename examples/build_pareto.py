#!/usr/bin/env python3
"""Build Pareto frontier plots for gen_ppl vs diversity metrics.

Each group on the plot contains points that differ only by diffusion_temperature.
Groups are defined by: model type, float64, nucleus, remasker_temperature, 
t_on, t_off, schedule, num_steps.
"""

import json
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class MetricPoint:
    """A single metrics point."""
    ppl: float
    diversity: float
    diffusion_temp: float
    filename: str


def parse_filename(filename: str) -> Dict[str, str]:
    """Parse metric filename to extract parameters.
    
    Examples:
    - mdlm_float64_True_nucleus_0.9_difftemp_1.0_len_512_steps_256.json
    - gstar_float64_True_nucleus_0.9_difftemp_1.1_remaskertemp_0_t_on_0.55_t_off_0.05_schedule_default_len_512_steps_256.json
    - starshape_float64_True_nucleus_0.9_difftemp_1.0_t_on_0.2_t_off_0_schedule_default_len_512_steps_256.json
    """
    params = {}
    
    # Extract model type (first part before _float64)
    model_match = re.match(r'^([a-z]+)_float64', filename)
    if model_match:
        params['model'] = model_match.group(1)
    
    # Extract float64
    float64_match = re.search(r'float64_(\w+)', filename)
    if float64_match:
        params['float64'] = float64_match.group(1)
    
    # Extract nucleus
    nucleus_match = re.search(r'nucleus_([\d.]+)', filename)
    if nucleus_match:
        params['nucleus'] = nucleus_match.group(1)
    
    # Extract diffusion temperature
    difftemp_match = re.search(r'difftemp_([\d.]+)', filename)
    if difftemp_match:
        params['difftemp'] = difftemp_match.group(1)
    
    # Extract remasker temperature (gstar only)
    remaskertemp_match = re.search(r'remaskertemp_(\d+)', filename)
    if remaskertemp_match:
        params['remaskertemp'] = remaskertemp_match.group(1)
    
    # Extract t_on
    t_on_match = re.search(r't_on_([\d.]+)', filename)
    if t_on_match:
        params['t_on'] = t_on_match.group(1)
    
    # Extract t_off
    t_off_match = re.search(r't_off_([\d.]+)', filename)
    if t_off_match:
        params['t_off'] = t_off_match.group(1)
    
    # Extract schedule
    schedule_match = re.search(r'schedule_(\w+)', filename)
    if schedule_match:
        params['schedule'] = schedule_match.group(1)
    
    # Extract num_steps
    steps_match = re.search(r'steps_(\d+)', filename)
    if steps_match:
        params['steps'] = steps_match.group(1)
    
    # Extract length
    len_match = re.search(r'len_(\d+)', filename)
    if len_match:
        params['len'] = len_match.group(1)
    
    return params


def get_group_key(params: Dict[str, str], subdir: str) -> str:
    """Create a group key from parameters (excluding diffusion_temperature).
    
    Points in the same group only differ by diffusion_temperature.
    """
    # Include subdirectory as part of the group (different model variants)
    parts = [subdir] if subdir else []
    
    # Add model if present
    if 'model' in params:
        parts.append(params['model'])
    
    # Add all parameters except difftemp
    for key in ['float64', 'nucleus', 'remaskertemp', 't_on', 't_off', 'schedule', 'steps']:
        if key in params:
            parts.append(f"{key}={params[key]}")
    
    return ' | '.join(parts)


def get_display_label(group_key: str) -> str:
    """Create a shorter display label for the legend."""
    # Replace some verbose parts
    label = group_key
    label = label.replace('float64=True', 'f64')
    label = label.replace('float64=False', 'f32')
    label = label.replace('nucleus=', 'p=')
    label = label.replace('remaskertemp=', 'rT=')
    label = label.replace('schedule=', 'sch=')
    label = label.replace('steps=', 's=')
    label = label.replace('t_on=', 'on=')
    label = label.replace('t_off=', 'off=')
    return label


def load_metrics(metrics_dir: Path) -> Dict[str, List[MetricPoint]]:
    """Load all metrics and group them.
    
    Returns a dict mapping group_key -> list of MetricPoints
    """
    groups = defaultdict(list)
    
    # Walk through metrics directory
    for json_file in metrics_dir.rglob('*.json'):
        # Get subdirectory relative to metrics_dir
        rel_path = json_file.relative_to(metrics_dir)
        subdir = str(rel_path.parent) if rel_path.parent != Path('.') else ''
        
        # Parse filename
        params = parse_filename(json_file.name)
        if not params:
            print(f"Warning: Could not parse {json_file.name}")
            continue
        
        # Load metrics
        try:
            with open(json_file, 'r') as f:
                metrics = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load {json_file}: {e}")
            continue
        
        # Extract PPL and diversity
        ppl = metrics.get('ppl')
        diversity = metrics.get('distinct_n')
        
        if ppl is None or diversity is None:
            print(f"Warning: Missing ppl or diversity in {json_file}")
            continue
        
        # Get diffusion temperature
        difftemp = float(params.get('difftemp', 1.0))
        
        # Create group key
        group_key = get_group_key(params, subdir)
        
        # Add to group
        groups[group_key].append(MetricPoint(
            ppl=ppl,
            diversity=diversity,
            diffusion_temp=difftemp,
            filename=json_file.name
        ))
    
    return dict(groups)


def compute_pareto_frontier(points: List[MetricPoint]) -> List[MetricPoint]:
    """Compute Pareto frontier for PPL vs diversity.
    
    We want to minimize PPL and maximize diversity.
    A point is on the Pareto frontier if no other point has both lower PPL and higher diversity.
    """
    if not points:
        return []
    
    # Sort by PPL (ascending)
    sorted_points = sorted(points, key=lambda p: p.ppl)
    
    frontier = []
    max_diversity = float('-inf')
    
    for point in sorted_points:
        if point.diversity > max_diversity:
            frontier.append(point)
            max_diversity = point.diversity
    
    return frontier


def plot_pareto(groups: Dict[str, List[MetricPoint]], output_path: Path):
    """Plot Pareto frontiers for all groups."""
    
    # Set up the plot with a nice style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Color map for different groups
    colors = plt.cm.tab20(np.linspace(0, 1, len(groups)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*', 'X', 'P']
    
    for idx, (group_key, points) in enumerate(sorted(groups.items())):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        label = get_display_label(group_key)
        
        # Extract x (PPL) and y (diversity) values
        ppls = [p.ppl for p in points]
        diversities = [p.diversity for p in points]
        diff_temps = [p.diffusion_temp for p in points]
        
        # Plot all points
        scatter = ax.scatter(ppls, diversities, c=[color], marker=marker, 
                            s=100, alpha=0.7, label=label, edgecolors='black', linewidths=0.5)
        
        # Annotate points with diffusion temperature
        for ppl, div, temp in zip(ppls, diversities, diff_temps):
            ax.annotate(f'{temp}', (ppl, div), textcoords="offset points", 
                       xytext=(5, 5), fontsize=7, alpha=0.7)
        
        # Compute and plot Pareto frontier
        frontier = compute_pareto_frontier(points)
        if len(frontier) > 1:
            frontier_sorted = sorted(frontier, key=lambda p: p.ppl)
            frontier_ppls = [p.ppl for p in frontier_sorted]
            frontier_divs = [p.diversity for p in frontier_sorted]
            ax.plot(frontier_ppls, frontier_divs, c=color, linestyle='--', alpha=0.5, linewidth=1.5)
    
    ax.set_xlabel('Generative Perplexity (↓ better)', fontsize=12)
    ax.set_ylabel('Diversity (distinct-n) (↑ better)', fontsize=12)
    ax.set_title('Pareto Frontier: PPL vs Diversity\n(points annotated with diffusion_temperature)', fontsize=14)
    
    # Place legend outside the plot
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, 
             title='Groups (vary by diff_temp)', title_fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved Pareto plot to {output_path}")


def plot_pareto_by_model(groups: Dict[str, List[MetricPoint]], output_dir: Path):
    """Create separate plots for each model type."""
    
    # Group by model type
    model_groups = defaultdict(dict)
    for group_key, points in groups.items():
        # Extract model from group key (first part)
        parts = group_key.split(' | ')
        if len(parts) >= 2:
            model = f"{parts[0]}/{parts[1]}" if parts[0] else parts[1]
        else:
            model = parts[0]
        
        model_groups[model][group_key] = points
    
    # Create a plot for each model
    for model, model_data in model_groups.items():
        # Sanitize model name for filename
        safe_model = model.replace('/', '_').replace(' ', '_')
        output_path = output_dir / f"pareto_{safe_model}.png"
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_data)))
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*']
        
        for idx, (group_key, points) in enumerate(sorted(model_data.items())):
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            label = get_display_label(group_key)
            
            ppls = [p.ppl for p in points]
            diversities = [p.diversity for p in points]
            diff_temps = [p.diffusion_temp for p in points]
            
            ax.scatter(ppls, diversities, c=[color], marker=marker, 
                      s=120, alpha=0.8, label=label, edgecolors='black', linewidths=0.5)
            
            for ppl, div, temp in zip(ppls, diversities, diff_temps):
                ax.annotate(f'{temp}', (ppl, div), textcoords="offset points", 
                           xytext=(5, 5), fontsize=8, alpha=0.8)
            
            frontier = compute_pareto_frontier(points)
            if len(frontier) > 1:
                frontier_sorted = sorted(frontier, key=lambda p: p.ppl)
                frontier_ppls = [p.ppl for p in frontier_sorted]
                frontier_divs = [p.diversity for p in frontier_sorted]
                ax.plot(frontier_ppls, frontier_divs, c=color, linestyle='--', alpha=0.6, linewidth=2)
        
        ax.set_xlabel('Generative Perplexity (↓ better)', fontsize=12)
        ax.set_ylabel('Diversity (distinct-n) (↑ better)', fontsize=12)
        ax.set_title(f'Pareto Frontier: {model}\n(points annotated with diffusion_temperature)', fontsize=14)
        
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9,
                 title='Configurations', title_fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {model} Pareto plot to {output_path}")


def main():
    # Find project root (parent of examples/)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    metrics_dir = project_root / 'metrics'
    output_dir = project_root / 'plots'
    output_dir.mkdir(exist_ok=True)
    
    if not metrics_dir.exists():
        print(f"Error: Metrics directory not found at {metrics_dir}")
        return
    
    print(f"Loading metrics from {metrics_dir}")
    groups = load_metrics(metrics_dir)
    
    if not groups:
        print("No metrics found!")
        return
    
    print(f"Found {len(groups)} groups:")
    for group_key, points in sorted(groups.items()):
        print(f"  - {group_key}: {len(points)} points")
    
    # Create combined plot
    combined_output = output_dir / 'pareto_all.png'
    plot_pareto(groups, combined_output)
    
    # Create per-model plots
    plot_pareto_by_model(groups, output_dir)
    
    print(f"\nAll plots saved to {output_dir}/")


if __name__ == '__main__':
    main()

