"""Plot saved standardized ridge coefficients without fitting or altering them."""
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "output/cp2/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

POSITIVE = "#1675A9"
NEGATIVE = "#C65B24"
ZERO = "#777777"


def display_names(names):
    return names.str.replace(r"^(continuous|categorical)__", "", regex=True)


def draw_coefficients(ax, rows, label_size):
    """Bars retain signed values; row positions keep every input identifiable."""
    values = rows.standardized_coefficient.to_numpy()
    positions = np.arange(len(rows))
    colors = np.where(values > 0, POSITIVE, np.where(values < 0, NEGATIVE, ZERO))
    bars = ax.barh(positions, values, color=colors, height=0.68, zorder=3)
    ax.set_yticks(positions, display_names(rows.predictor), fontsize=label_size)
    ax.invert_yaxis()
    ax.axvline(0, color="#333333", linewidth=0.9, zorder=2)
    # A zero-width bar is invisible: show an explicit neutral dot for zero terms.
    ax.scatter(np.zeros(np.sum(values == 0)), positions[values == 0], color=ZERO, s=12, zorder=4)
    ax.set_xlabel("Signed standardized coefficient (log odds per one SD)", fontsize=10)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", length=0, pad=7)
    ax.grid(axis="x", color="#dddddd", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_ylim(len(rows)-0.25, -0.75)
    for y in range(0, len(rows), 2):
        ax.axhspan(y-0.5, y+0.5, color="#f3f5f7", zorder=0)
    # Verify the plotted artist widths, not just the input data selection.
    np.testing.assert_array_equal([bar.get_width() for bar in bars], values)
    assert [tick.get_text() for tick in ax.get_yticklabels()] == display_names(rows.predictor).tolist()
    return [{"predictor": name, "standardized_coefficient": float(bar.get_width())}
            for name, bar in zip(rows.predictor, bars)]


def plot_ridge_coefficients(coefficients, output_dir=Path("output/cp2")):
    """Return an audit of exactly which CSV values were drawn in each figure.

    The supplied frame and source CSV are never written. Supplementary panels
    retain the original ascending-signed-coefficient partition used previously.
    """
    output_dir = Path(output_dir)
    assert len(coefficients) == 267, "Expected the complete 267-coefficient table"
    assert coefficients.predictor.is_unique
    assert display_names(coefficients.predictor).is_unique
    assert np.isfinite(coefficients.standardized_coefficient).all()
    original = coefficients.copy(deep=True)
    top = (coefficients.assign(magnitude=coefficients.standardized_coefficient.abs())
           .sort_values(["magnitude", "predictor"], ascending=[False, True])
           .head(20).drop(columns="magnitude"))
    legend = [Patch(color=POSITIVE, label="Positive"), Patch(color=NEGATIVE, label="Negative"),
              Patch(color=ZERO, label="Zero (dot)")]
    fig, ax = plt.subplots(figsize=(9, 7.4))
    top_audit = draw_coefficients(ax, top, label_size=11)
    ax.set_title("Figure 1. Top 20 standardized ridge coefficients", loc="left", fontsize=13, pad=31)
    ax.text(0, 1.025, "Ranked by absolute magnitude, largest at top; bars show signed values",
            transform=ax.transAxes, fontsize=9)
    fig.legend(handles=legend[:2], loc="lower center", bbox_to_anchor=(0.70, 0.005),
               ncol=2, fontsize=9, frameon=False)
    fig.subplots_adjust(left=0.43, right=0.97, top=0.88, bottom=0.11)
    fig.savefig(output_dir / "ridge_coefficients_top20.png", dpi=220)
    plt.close(fig)

    ordered = coefficients.sort_values("standardized_coefficient")
    panels = [ordered.iloc[index] for index in np.array_split(np.arange(len(ordered)), 4)]
    panel_audits = []
    for number, panel in enumerate(panels, 1):
        fig, ax = plt.subplots(figsize=(9, 10.6))
        plotted = draw_coefficients(ax, panel, label_size=9.5)
        low = min(0.0, panel.standardized_coefficient.min())
        high = max(0.0, panel.standardized_coefficient.max())
        padding = max((high-low)*0.08, 0.00001)
        ax.set_xlim(low-padding, high+padding)
        ax.set_title(f"Figure A1{chr(96+number)}. Complete ridge coefficients ({number}/4; n={len(panel)})",
                     loc="left", fontsize=11, pad=32)
        ax.text(0, 1.025, "Panel-specific x scale; read tick values when comparing panels",
                transform=ax.transAxes, fontsize=8.5)
        fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.70, 0.005),
                   ncol=3, fontsize=8, frameon=False)
        fig.subplots_adjust(left=0.45, right=0.97, top=0.91, bottom=0.075)
        fig.savefig(output_dir / f"ridge_coefficients_panel_{number}.png", dpi=220)
        panel_audits.append({"panel": number, "x_limits": list(ax.get_xlim()), "rows": plotted})
        plt.close(fig)
    all_rows = [row for panel in panel_audits for row in panel["rows"]]
    names = [row["predictor"] for row in all_rows]
    assert len(names) == len(set(names)) == 267
    assert set(names) == set(coefficients.predictor)
    pd.testing.assert_frame_equal(coefficients, original)
    return {"top20": top_audit, "supplementary_panels": panel_audits,
            "total_supplementary_coefficients": len(names), "unique_supplementary_coefficients": len(set(names)),
            "bar_widths_and_display_labels_checked": True,
            "scale_note": "Supplementary panels use explicitly labeled panel-specific x-axis scales."}
