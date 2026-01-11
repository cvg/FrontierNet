#!/usr/bin/env python3
"""
Statistics script for FrontierNet exploration.

This script reads the JSON file with mapped volumes from replay.py,
loads a reference voxel grid PLY file, and computes the per-step
exploration ratio (explored volume / total volume).
"""
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def load_json_entries(json_path: str) -> List[Dict[str, Any]]:
    """
    Load JSON lines file (output from replay.py).
    
    Args:
        json_path: Path to the JSON lines file
        
    Returns:
        List of entry dictionaries
    """
    entries = []
    with open(json_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def extract_volumes(entries: List[Dict[str, Any]]) -> List[float]:
    """
    Extract mapped_vol values from entries.
    
    Args:
        entries: List of entry dictionaries
        
    Returns:
        List of volumes (None values replaced with 0.0)
    """
    volumes = []
    for entry in entries:
        vol = entry.get("mapped_vol")
        if vol is None:
            vol = 0.0
        volumes.append(float(vol))
    return volumes


def load_voxel_grid(ply_path: str) -> Tuple[o3d.geometry.VoxelGrid, float, int]:
    """
    Load voxel grid from PLY file and extract properties.
    
    Args:
        ply_path: Path to the voxel_grid.ply file
        
    Returns:
        Tuple of (voxel_grid, voxel_size, num_voxels)
    """
    # Try loading as VoxelGrid first
    try:
        voxel_grid = o3d.io.read_voxel_grid(ply_path)
        voxel_size = voxel_grid.voxel_size
        num_voxels = len(voxel_grid.get_voxels())
        logging.info(f"Loaded voxel grid: {num_voxels} voxels, voxel_size={voxel_size:.4f}m")
        return voxel_grid, voxel_size, num_voxels
    except Exception as e:
        logging.warning(f"Could not load as VoxelGrid: {e}")
    
    # Fallback: load as point cloud and estimate voxel size
    try:
        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)
        num_points = len(points)
        
        # Estimate voxel size from point spacing (assume regular grid)
        if num_points > 1:
            # Find minimum non-zero distance between adjacent points
            from scipy.spatial import cKDTree
            tree = cKDTree(points)
            distances, _ = tree.query(points, k=2)
            min_dist = np.min(distances[:, 1][distances[:, 1] > 0])
            voxel_size = min_dist
        else:
            voxel_size = 0.1  # Default
            
        logging.info(f"Loaded point cloud as voxel centers: {num_points} points, estimated voxel_size={voxel_size:.4f}m")
        return None, voxel_size, num_points
    except Exception as e:
        logging.error(f"Could not load PLY file: {e}")
        raise


def compute_total_volume(num_voxels: int, voxel_size: float) -> float:
    """
    Compute total volume from voxel count and size.
    
    Args:
        num_voxels: Number of voxels
        voxel_size: Size of each voxel in meters
        
    Returns:
        Total volume in cubic meters
    """
    return num_voxels * (voxel_size ** 3)


def compute_exploration_ratios(
    volumes: List[float], 
    total_volume: float
) -> List[float]:
    """
    Compute per-step exploration ratios.
    
    Args:
        volumes: List of explored volumes per step
        total_volume: Total explorable volume
        
    Returns:
        List of ratios (explored / total)
    """
    if total_volume <= 0:
        logging.warning("Total volume is zero or negative, returning zeros")
        return [0.0] * len(volumes)
    
    ratios = [vol / total_volume for vol in volumes]
    return ratios


def plot_exploration_curve(
    steps: List[int],
    ratios: List[float],
    output_path: Optional[str] = None,
    title: str = "Exploration Progress",
    show_plot: bool = True
) -> None:
    """
    Plot exploration ratio vs step.
    
    Args:
        steps: List of step numbers
        ratios: List of exploration ratios
        output_path: Path to save the plot (optional)
        title: Plot title
        show_plot: Whether to display the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(steps, ratios, 'b-', linewidth=2, label='Exploration Ratio')
    ax.fill_between(steps, ratios, alpha=0.3)
    
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Exploration Ratio (Explored / Total)', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    ax.set_xlim(0, max(steps) if steps else 1)
    ax.set_ylim(0, min(1.1, max(ratios) * 1.1) if ratios else 1.0)
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    
    # Add final ratio annotation
    if ratios:
        final_ratio = ratios[-1]
        ax.axhline(y=final_ratio, color='r', linestyle='--', alpha=0.5)
        ax.annotate(
            f'Final: {final_ratio:.2%}',
            xy=(steps[-1], final_ratio),
            xytext=(steps[-1] * 0.8, final_ratio + 0.05),
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='red', alpha=0.5)
        )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logging.info(f"Plot saved to {output_path}")
    
    if show_plot:
        plt.show()
    
    plt.close()


def print_statistics(
    volumes: List[float],
    ratios: List[float],
    total_volume: float,
    voxel_size: float,
    num_total_voxels: int
) -> None:
    """
    Print exploration statistics summary.
    """
    print("\n" + "=" * 50)
    print("EXPLORATION STATISTICS")
    print("=" * 50)
    print(f"\nReference Voxel Grid:")
    print(f"  Voxel size:     {voxel_size:.4f} m")
    print(f"  Total voxels:   {num_total_voxels:,}")
    print(f"  Total volume:   {total_volume:.4f} m³")
    
    print(f"\nExploration Progress:")
    print(f"  Total steps:    {len(volumes)}")
    print(f"  Final volume:   {volumes[-1]:.4f} m³" if volumes else "  No data")
    print(f"  Final ratio:    {ratios[-1]:.2%}" if ratios else "  No data")
    
    if ratios:
        # Find when we reached certain milestones
        milestones = [0.25, 0.50, 0.75, 0.90, 0.95]
        print(f"\n  Milestones:")
        for milestone in milestones:
            reached = False
            for i, r in enumerate(ratios):
                if r >= milestone:
                    print(f"    {milestone:.0%} reached at step {i + 1}")
                    reached = True
                    break
            if not reached:
                print(f"    {milestone:.0%} not reached")
    
    print("=" * 50 + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser."""
    p = argparse.ArgumentParser(
        description="Compute and plot exploration statistics from replay output"
    )
    
    p.add_argument(
        "--json_file",
        "-j",
        type=str,
        required=True,
        help="Path to the JSON file with mapped volumes (from replay.py)"
    )
    p.add_argument(
        "--voxel_grid",
        "-v",
        type=str,
        required=True,
        help="Path to the full voxel_grid.ply file (reference for total volume)"
    )
    p.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path for the plot image (default: json_file_exploration.png)"
    )
    p.add_argument(
        "--voxel_size",
        type=float,
        default=None,
        help="Override voxel size (meters). If not provided, extracted from PLY."
    )
    p.add_argument(
        "--title",
        type=str,
        default="Exploration Progress",
        help="Title for the plot"
    )
    p.add_argument(
        "--no_show",
        action="store_true",
        help="Don't display the plot (only save)"
    )
    p.add_argument(
        "--log_level",
        "-ll",
        type=int,
        default=20,
        help="Logging level (10=debug, 20=info, 30=warning)"
    )
    
    return p


def main():
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    
    parser = build_arg_parser()
    args = parser.parse_args()
    
    # Set logging level
    if args.log_level < 20:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.log_level < 30:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Load JSON entries
    logging.info(f"Loading JSON file: {args.json_file}")
    entries = load_json_entries(args.json_file)
    if not entries:
        logging.error("No entries found in JSON file")
        return
    logging.info(f"Loaded {len(entries)} entries")
    
    # Extract volumes
    volumes = extract_volumes(entries)
    logging.info(f"Extracted {len(volumes)} volume values")
    
    # Load voxel grid
    logging.info(f"Loading voxel grid: {args.voxel_grid}")
    voxel_grid, voxel_size, num_total_voxels = load_voxel_grid(args.voxel_grid)
    
    # Override voxel size if provided
    if args.voxel_size is not None:
        voxel_size = args.voxel_size
        logging.info(f"Using override voxel size: {voxel_size}m")
    
    # Compute total volume
    total_volume = compute_total_volume(num_total_voxels, voxel_size)
    logging.info(f"Total volume: {total_volume:.4f} m³")
    
    # Compute exploration ratios
    ratios = compute_exploration_ratios(volumes, total_volume)
    
    # Create step numbers (1-indexed)
    steps = list(range(1, len(volumes) + 1))
    
    # Print statistics
    print_statistics(volumes, ratios, total_volume, voxel_size, num_total_voxels)
    
    # Determine output path
    output_path = args.output
    if output_path is None:
        base = Path(args.json_file).stem
        output_path = str(Path(args.json_file).parent / f"{base}_exploration.png")
    
    # Plot
    plot_exploration_curve(
        steps=steps,
        ratios=ratios,
        output_path=output_path,
        title=args.title,
        show_plot=not args.no_show
    )


if __name__ == "__main__":
    main()
