#!/usr/bin/env python3
"""
Visualize training statistics from CSV files.

This script reads training statistics saved during training/fine-tuning
and generates visualization graphs.

Usage:
    python -m src.scripts.visualize_stats stats.csv
    python -m src.scripts.visualize_stats stats.csv --output training_plot.png
    python -m src.scripts.visualize_stats stats1.csv stats2.csv --labels "Run 1" "Run 2"
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.stats_logger import load_stats_csv


def plot_training_stats(
    csv_paths: list[str],
    output_path: str | None = None,
    labels: list[str] | None = None,
    title: str = "Training Statistics",
    figsize: tuple[int, int] = (14, 10),
    show: bool = True,
) -> None:
    """
    Plot training statistics from one or more CSV files.

    Args:
        csv_paths: List of paths to CSV files
        output_path: Path to save the figure (optional)
        labels: Labels for each CSV file (for legend)
        title: Plot title
        figsize: Figure size (width, height)
        show: Whether to display the plot interactively
    """
    if labels is None:
        labels = [Path(p).stem for p in csv_paths]

    # Load all stats
    all_stats = []
    for csv_path in csv_paths:
        stats = load_stats_csv(csv_path)
        all_stats.append(stats)

    # Determine which metrics are available
    available_metrics = set()
    for stats in all_stats:
        if stats:
            available_metrics.update(stats[0].keys())

    # Remove non-numeric columns
    available_metrics.discard("epoch")

    # Define the metrics we want to plot and their display names
    metric_config = {
        "loss": {"name": "Total Loss", "color_idx": 0},
        "loss_mel": {"name": "Mel Loss", "color_idx": 1},
        "loss_kl": {"name": "KL Loss", "color_idx": 2},
        "loss_adv": {"name": "Adversarial Loss", "color_idx": 3},
        "loss_fm": {"name": "Feature Matching Loss", "color_idx": 4},
        "loss_d": {"name": "Discriminator Loss", "color_idx": 5},
        "learning_rate": {"name": "Learning Rate", "color_idx": 6},
        # Quality metrics
        "mcd": {"name": "MCD (dB, lower=better)", "color_idx": 7},
        "d_acc_real": {"name": "D Accuracy (Real)", "color_idx": 8},
        "d_acc_fake": {"name": "D Accuracy (Fake, lower=G fools D)", "color_idx": 9},
    }

    # Filter to only available metrics
    metrics_to_plot = [m for m in metric_config.keys() if m in available_metrics]

    if not metrics_to_plot:
        print("No plottable metrics found in CSV files.")
        return

    # Create color palette
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Determine subplot layout
    n_metrics = len(metrics_to_plot)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_metrics > 1 else [axes]

    # Plot each metric
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        config = metric_config[metric]

        for stats, label in zip(all_stats, labels):
            epochs = [s["epoch"] for s in stats]
            values = [s.get(metric, 0) for s in stats]

            # Skip if all values are zero
            if all(v == 0 for v in values):
                continue

            ax.plot(epochs, values, label=label, linewidth=1.5, alpha=0.8)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(config["name"])
        ax.set_title(config["name"])
        ax.grid(True, alpha=0.3)

        if len(csv_paths) > 1:
            ax.legend(loc="best", fontsize=8)

        # Use log scale for learning rate if values span multiple orders of magnitude
        if metric == "learning_rate":
            values_flat = []
            for stats in all_stats:
                values_flat.extend([s.get(metric, 0) for s in stats])
            values_flat = [v for v in values_flat if v > 0]
            if values_flat and max(values_flat) / min(values_flat) > 10:
                ax.set_yscale("log")

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)

    # Add overall title
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    # Save if output path provided
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {output_path}")

    # Show if requested
    if show:
        plt.show()

    plt.close()


def plot_loss_comparison(
    csv_paths: list[str],
    output_path: str | None = None,
    labels: list[str] | None = None,
    title: str = "Loss Comparison",
    figsize: tuple[int, int] = (10, 6),
    show: bool = True,
) -> None:
    """
    Plot only the total loss for comparison between runs.

    Args:
        csv_paths: List of paths to CSV files
        output_path: Path to save the figure (optional)
        labels: Labels for each CSV file (for legend)
        title: Plot title
        figsize: Figure size (width, height)
        show: Whether to display the plot interactively
    """
    if labels is None:
        labels = [Path(p).stem for p in csv_paths]

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(csv_paths)))

    for idx, (csv_path, label) in enumerate(zip(csv_paths, labels)):
        stats = load_stats_csv(csv_path)
        epochs = [s["epoch"] for s in stats]
        losses = [s.get("loss", 0) for s in stats]

        ax.plot(epochs, losses, label=label, color=colors[idx], linewidth=2)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Total Loss", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {output_path}")

    if show:
        plt.show()

    plt.close()


def print_summary(csv_path: str) -> None:
    """Print a summary of training statistics."""
    stats = load_stats_csv(csv_path)

    if not stats:
        print("No statistics found.")
        return

    print(f"\n{'='*60}")
    print(f"Training Statistics Summary: {csv_path}")
    print(f"{'='*60}")

    print(f"\nTotal epochs: {len(stats)}")

    # Get available metrics
    metrics = [k for k in stats[0].keys() if k != "epoch"]

    # Print final values
    print("\nFinal values (last epoch):")
    final = stats[-1]
    for metric in metrics:
        value = final.get(metric, 0)
        if value != 0:
            print(f"  {metric}: {value:.6f}")

    # Print best values (minimum loss)
    print("\nBest values (minimum):")
    for metric in ["loss", "loss_mel", "loss_kl"]:
        if metric in stats[0]:
            values = [s.get(metric, float("inf")) for s in stats]
            min_val = min(values)
            min_epoch = values.index(min_val) + 1
            print(f"  {metric}: {min_val:.6f} (epoch {min_epoch})")

    # Print quality metrics summary if available
    if "mcd" in stats[0] and stats[0]["mcd"] != 0:
        print("\nQuality Metrics (best values):")
        # MCD: lower is better
        mcd_values = [s.get("mcd", float("inf")) for s in stats if s.get("mcd", 0) > 0]
        if mcd_values:
            min_mcd = min(mcd_values)
            min_mcd_epoch = [s.get("mcd", float("inf")) for s in stats].index(min_mcd) + 1
            print(f"  MCD (lowest): {min_mcd:.2f} dB (epoch {min_mcd_epoch})")

        # Discriminator accuracy
        if "d_acc_real" in stats[0]:
            d_real_values = [s.get("d_acc_real", 0) for s in stats if s.get("d_acc_real", 0) > 0]
            d_fake_values = [s.get("d_acc_fake", 0) for s in stats if s.get("d_acc_fake", 0) > 0]
            if d_real_values and d_fake_values:
                print(f"  D accuracy (final): Real={d_real_values[-1]:.1%}, Fake={d_fake_values[-1]:.1%}")
                print(f"    (Real should stay high; lower Fake means G is fooling D)")

    # Print statistics
    print("\nLoss statistics:")
    losses = [s.get("loss", 0) for s in stats]
    print(f"  Initial: {losses[0]:.6f}")
    print(f"  Final:   {losses[-1]:.6f}")
    print(f"  Min:     {min(losses):.6f}")
    print(f"  Max:     {max(losses):.6f}")
    print(f"  Mean:    {np.mean(losses):.6f}")

    # Calculate improvement
    if losses[0] > 0:
        improvement = (losses[0] - losses[-1]) / losses[0] * 100
        print(f"  Improvement: {improvement:.1f}%")

    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize training statistics from CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot single training run
  python -m src.scripts.visualize_stats training_stats.csv

  # Save plot to file without displaying
  python -m src.scripts.visualize_stats stats.csv --output plot.png --no-show

  # Compare multiple training runs
  python -m src.scripts.visualize_stats run1.csv run2.csv --labels "Baseline" "With augmentation"

  # Print summary only (no plot)
  python -m src.scripts.visualize_stats stats.csv --summary-only

  # Simple loss comparison plot
  python -m src.scripts.visualize_stats run1.csv run2.csv --compare
        """
    )

    parser.add_argument(
        "csv_files",
        nargs="+",
        help="Path(s) to CSV file(s) with training statistics"
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output path for the plot image"
    )
    parser.add_argument(
        "--labels", "-l",
        nargs="*",
        default=None,
        help="Labels for each CSV file (for legend)"
    )
    parser.add_argument(
        "--title", "-t",
        default="Training Statistics",
        help="Plot title"
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Don't display the plot interactively (useful for scripts)"
    )
    parser.add_argument(
        "--summary-only", "-s",
        action="store_true",
        help="Only print summary, don't generate plot"
    )
    parser.add_argument(
        "--compare", "-c",
        action="store_true",
        help="Simple loss comparison plot (single graph)"
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=int,
        default=[14, 10],
        metavar=("WIDTH", "HEIGHT"),
        help="Figure size in inches (default: 14 10)"
    )

    args = parser.parse_args()

    # Validate CSV files exist
    for csv_path in args.csv_files:
        if not Path(csv_path).exists():
            print(f"Error: File not found: {csv_path}")
            return 1

    # Print summary for each file
    if args.summary_only or len(args.csv_files) == 1:
        for csv_path in args.csv_files:
            print_summary(csv_path)

    if args.summary_only:
        return 0

    # Generate plot
    if args.compare:
        plot_loss_comparison(
            csv_paths=args.csv_files,
            output_path=args.output,
            labels=args.labels,
            title=args.title,
            figsize=tuple(args.figsize[:2]) if len(args.figsize) >= 2 else (10, 6),
            show=not args.no_show,
        )
    else:
        plot_training_stats(
            csv_paths=args.csv_files,
            output_path=args.output,
            labels=args.labels,
            title=args.title,
            figsize=tuple(args.figsize),
            show=not args.no_show,
        )

    return 0


if __name__ == "__main__":
    exit(main())
