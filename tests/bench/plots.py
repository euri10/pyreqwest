import argparse
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from tests.bench.utils import StatsCollection, fmt_size, is_sync


def create_plot(collection: StatsCollection, comparison_lib: str) -> None:
    self_lib = "pyreqwest_sync" if is_sync(comparison_lib) else "pyreqwest"

    body_sizes = sorted({stat.body_size for stat in collection.stats if stat.lib == self_lib})
    concurrency_levels = sorted({stat.concurrency for stat in collection.stats if stat.lib == self_lib})

    fig, axes = plt.subplots(nrows=len(body_sizes), ncols=len(concurrency_levels), figsize=(18, 16))
    fig.suptitle(f"pyreqwest vs {comparison_lib}", fontsize=16, y=0.98)
    legend_colors = {"pyreqwest": "lightblue", comparison_lib: "lightcoral"}

    for i, body_size in enumerate(body_sizes):
        ymax = 0.0

        for j, concurrency in enumerate(concurrency_levels):
            ax: Axes = axes[i][j]

            self_stats = collection.find(self_lib, body_size, concurrency)
            comparison_stats = collection.find(comparison_lib, body_size, concurrency)
            assert self_stats, f"Missing stats for {self_lib}, size={body_size}, concurrency={concurrency}"
            assert comparison_stats, f"Missing stats for {comparison_lib}, size={body_size}, concurrency={concurrency}"

            # Create box plot for this specific body size and concurrency combination
            box_plot = ax.boxplot(
                [self_stats.timings, comparison_stats.timings],
                patch_artist=True,
                showfliers=False,
                tick_labels=["pyreqwest", comparison_lib],
                widths=0.6,
            )
            ymax = max(ymax, ax.get_ylim()[1])

            # Color the boxes
            for patch, color in zip(box_plot["boxes"], legend_colors.values(), strict=False):
                patch.set_facecolor(color)

            # Customize subplot
            ax.set_title(f"{fmt_size(body_size)} @ {concurrency} concurrent", fontweight="bold", pad=10)
            ax.set_ylabel("Response Time (ms)")
            ax.grid(True, alpha=0.3)

            # Calculate and add performance comparison
            pyreqwest_median = statistics.median(self_stats.timings)
            comparison_median = statistics.median(comparison_stats.timings)
            speedup = comparison_median / pyreqwest_median if pyreqwest_median != 0 else 0

            if speedup > 1:
                faster_lib = "pyreqwest"
                speedup_text = f"{((speedup - 1) * 100):.1f}% faster"
            else:
                faster_lib = comparison_lib
                speedup_text = f"{((1 / speedup - 1) * 100):.1f}% faster"

            # Add performance annotation
            ax.text(
                0.5,
                0.95,
                f"{faster_lib}\n{speedup_text}",
                transform=ax.transAxes,
                ha="center",
                va="top",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "wheat", "alpha": 0.8},
                fontsize=9,
                fontweight="bold",
            )

            # Add median time annotations
            ax.text(
                1,
                pyreqwest_median,
                f"{pyreqwest_median:.3f}ms",
                ha="left",
                va="center",
                fontsize=8,
                color="darkblue",
                fontweight="bold",
            )
            ax.text(
                2,
                comparison_median,
                f"{comparison_median:.3f}ms",
                ha="right",
                va="center",
                fontsize=8,
                color="darkred",
                fontweight="bold",
            )

        for j, _ in enumerate(concurrency_levels):
            axes[i][j].set_ylim(ymin=0, ymax=ymax * 1.01)  # Uniform y-axis per row

    # Add overall legend
    legends = [
        Rectangle(xy=(0, 0), width=1, height=1, label=label, facecolor=color) for label, color in legend_colors.items()
    ]
    fig.legend(handles=legends, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2)

    plt.tight_layout()
    plt.subplots_adjust(top=0.94, bottom=0.06)  # Make room for suptitle and legend

    # Save the plot
    img_path = Path(__file__).parent / f"benchmark_{comparison_lib}.png"
    plt.savefig(str(img_path), dpi=300, bbox_inches="tight")
    print(f"Plot saved as '{img_path}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Performance latency")
    parser.add_argument("--lib", type=str)
    args = parser.parse_args()

    create_plot(collection=StatsCollection.load(), comparison_lib=args.lib)


if __name__ == "__main__":
    main()
