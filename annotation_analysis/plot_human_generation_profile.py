"""Plot separate generation-model and degradation-profile comparisons."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# Use print sized typography so labels remain readable when figures are placed
# at thesis column/page width.
plt.rcParams.update({
    "font.size": 21,
    "axes.titlesize": 22,
    "axes.labelsize": 22,
    "xtick.labelsize": 24,
    "ytick.labelsize": 20,
    "legend.fontsize": 21,
})


DIMENSION_ORDER = ["factuality", "relevance", "coherence", "fluency"]
MODEL_ORDER = [
    "gpt-oss-120b",
    "mistral-large-3-675b-instruct-2512",
    "qwen3.5-122b-a10b",
]
PROFILE_ORDER = [
    "high_quality",
    "balanced_mid_quality",
    "low_factuality",
    "low_coverage",
    "low_coherence",
    "low_fluency",
]
PROFILE_LABELS = {
    "high_quality": "High quality",
    "balanced_mid_quality": "Balanced mid-quality",
    "low_factuality": "Low factuality",
    "low_coverage": "Low coverage",
    "low_coherence": "Low coherence",
    "low_fluency": "Low fluency",
}
MODEL_LABELS = {
    "gpt-oss-120b": "GPT-OSS",
    "mistral-large-3-675b-instruct-2512": "Mistral",
    "qwen3.5-122b-a10b": "Qwen",
}
MODEL_COLORS = {
    "gpt-oss-120b": "#0072B2",
    "mistral-large-3-675b-instruct-2512": "#D55E00",
    "qwen3.5-122b-a10b": "#009E73",
}
MODEL_OFFSETS = {
    "gpt-oss-120b": -0.16,
    "mistral-large-3-675b-instruct-2512": 0.0,
    "qwen3.5-122b-a10b": 0.16,
}
PROFILE_COLORS = {
    "high_quality": "#0072B2",
    "balanced_mid_quality": "#009E73",
    "low_factuality": "#D55E00",
    "low_coverage": "#CC79A7",
    "low_coherence": "#E69F00",
    "low_fluency": "#56B4E9",
}


def style_axis(axis, dimension):
    axis_label = "Coverage" if dimension == "relevance" else dimension.capitalize()
    axis.set_title(axis_label, loc="left", fontweight="bold", pad=16)
    axis.set_ylim(1, 5)
    axis.set_yticks([1, 2, 3, 4, 5])
    axis.grid(axis="y", color="#D9D9D9", linewidth=1)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#555555")
    axis.spines["bottom"].set_color("#555555")


def add_errorbar(axis, group, x, color, label):
    axis.errorbar(
        x,
        group["mean"],
        yerr=[group["mean"] - group["ci_95_low"], group["ci_95_high"] - group["mean"]],
        marker="o",
        linestyle="none",
        linewidth=1.2,
        markersize=5,
        capsize=2,
        elinewidth=1.1,
        color=color,
        label=label,
    )


def base_figure(title, subtitle):
    figure, axes = plt.subplots(2, 2, figsize=(16, 16), sharey=True)
    axes = axes.ravel()
    figure.suptitle(title, fontsize=27, x=0.05, ha="left")
    figure.text(0.05, 0.925, subtitle, fontsize=18, color="#555555")
    return figure, axes


def finish_figure(figure, axes, handles, labels, output_path, note=None):
    axes[0].set_ylabel("Mean human rating")
    axes[2].set_ylabel("Mean human rating")
    axes[2].set_xlabel("Generation model")
    axes[3].set_xlabel("Generation model")
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.075),
                  ncol=3, frameon=False)
    if note:
        figure.text(0.05, 0.16, note, fontsize=16, color="#555555")
    figure.tight_layout(rect=[0, 0.24, 1, 0.89])
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    figure.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def plot_generation_models(data, output_path):
    data = data.copy()
    data = data[data["dimension"].isin(DIMENSION_ORDER)]
    data = data[data["summary_model"].isin(MODEL_ORDER)]
    data["model_position"] = data["summary_model"].map({model: index for index, model in enumerate(MODEL_ORDER)})

    figure, axes = base_figure(
        "Human ratings by generation model",
        "Points: summary-level means; bars: bootstrap 95% confidence intervals",
    )
    color = "#0072B2"
    for axis, dimension in zip(axes, DIMENSION_ORDER):
        panel = data[data["dimension"] == dimension]
        group = panel.sort_values("model_position")
        if not group.empty:
            add_errorbar(axis, group, group["model_position"], color, "Mean human rating")
        axis.set_xticks(range(len(MODEL_ORDER)))
        axis.set_xticklabels([MODEL_LABELS[model] for model in MODEL_ORDER])
        style_axis(axis, dimension)
    handles, labels = axes[0].get_legend_handles_labels()
    finish_figure(figure, axes, handles, labels, output_path)


def plot_profiles(data, output_path):
    data = data[data["dimension"].isin(DIMENSION_ORDER)]
    data = data[data["summary_model"].isin(MODEL_ORDER)]
    data = data[data["profile_category"].isin(PROFILE_ORDER)]
    data["profile_position"] = data["profile_category"].map(
        {profile: index for index, profile in enumerate(PROFILE_ORDER)}
    )
    figure, axes = base_figure(
        "Human ratings by degradation profile",
        "Points: summary-level means; bars: bootstrap 95% confidence intervals",
    )
    for axis, dimension in zip(axes, DIMENSION_ORDER):
        dimension_low_profile = "low_coverage" if dimension == "relevance" else f"low_{dimension}"
        dimension_profiles = [
            profile for profile in ("high_quality", "balanced_mid_quality", dimension_low_profile)
            if profile in set(data["profile_category"])
        ]
        profile_positions = {profile: position for position, profile in enumerate(dimension_profiles)}
        panel = data[
            data["dimension"].eq(dimension)
            & data["profile_category"].isin(dimension_profiles)
        ]
        for model in MODEL_ORDER:
            group = panel[panel["summary_model"] == model].copy()
            if not group.empty:
                positions = group["profile_category"].map(profile_positions) + MODEL_OFFSETS[model]
                add_errorbar(axis, group, positions, MODEL_COLORS[model], MODEL_LABELS[model])
        axis.set_xticks(range(len(dimension_profiles)))
        axis.set_xticklabels([PROFILE_LABELS[profile] for profile in dimension_profiles], rotation=25, ha="right")
        style_axis(axis, dimension)
    axes[2].set_xlabel("Simplified degradation profile")
    axes[3].set_xlabel("Simplified degradation profile")
    axes[0].set_ylabel("Mean human rating")
    axes[2].set_ylabel("Mean human rating")
    handles, labels = axes[0].get_legend_handles_labels()
    finish_figure(
        figure, axes, handles, labels, output_path,
        "Profile categories overlap when a summary has multiple degraded dimensions.",
    )


def plot_generation_model_profiles(data, output_path):
    """Plot generation-model ratings as lines for each degradation profile."""
    data = data[data["dimension"].isin(DIMENSION_ORDER)].copy()
    data = data[data["summary_model"].isin(MODEL_ORDER)]
    data = data[data["profile_category"].isin(PROFILE_ORDER)]
    model_positions = {model: position for position, model in enumerate(MODEL_ORDER)}

    figure, axes = base_figure(
        "Human ratings by generation model and degradation profile",
        "Points: summary-level means; bars: bootstrap 95% confidence intervals",
    )
    for axis, dimension in zip(axes, DIMENSION_ORDER):
        panel = data[data["dimension"].eq(dimension)]
        for profile in PROFILE_ORDER:
            group = panel[panel["profile_category"].eq(profile)].set_index("summary_model").reindex(MODEL_ORDER)
            available = group["mean"].notna()
            if not available.any():
                continue
            group = group.loc[available]
            x = [model_positions[model] for model in group.index]
            axis.errorbar(
                x,
                group["mean"],
                yerr=[group["mean"] - group["ci_95_low"], group["ci_95_high"] - group["mean"]],
                marker="o",
                linestyle="-",
                linewidth=1.8,
                markersize=5,
                capsize=2,
                color=PROFILE_COLORS[profile],
                label=PROFILE_LABELS[profile],
            )
        axis.set_xticks(range(len(MODEL_ORDER)))
        axis.set_xticklabels([MODEL_LABELS[model] for model in MODEL_ORDER])
        style_axis(axis, dimension)
    handles, labels = axes[0].get_legend_handles_labels()
    finish_figure(figure, axes, handles, labels, output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-input-file",
        type=Path,
        default=Path("annotation_analysis/analysis_output/human_analysis/human_descriptives_by_generation_model.csv"),
    )
    parser.add_argument(
        "--profile-input-file",
        type=Path,
        default=Path("annotation_analysis/analysis_output/human_analysis/human_descriptives_by_generation_model_profile.csv"),
    )
    parser.add_argument(
        "--model-output-file",
        type=Path,
        default=Path("annotation_analysis/analysis_output/human_analysis/human_generation_model_comparison.png"),
    )
    parser.add_argument(
        "--profile-output-file",
        type=Path,
        default=Path("annotation_analysis/analysis_output/human_analysis/human_profile_comparison.png"),
    )
    parser.add_argument(
        "--model-profile-output-file",
        type=Path,
        default=Path("annotation_analysis/analysis_output/human_analysis/human_generation_model_profile_comparison.png"),
    )
    args = parser.parse_args()
    model_data = pd.read_csv(args.model_input_file)
    profile_data = pd.read_csv(args.profile_input_file)
    args.model_output_file.parent.mkdir(parents=True, exist_ok=True)
    plot_generation_models(model_data, args.model_output_file)
    plot_profiles(profile_data, args.profile_output_file)
    plot_generation_model_profiles(profile_data, args.model_profile_output_file)
    for path in (args.model_output_file, args.profile_output_file, args.model_profile_output_file):
        print(f"Wrote {path}")
        print(f"Wrote {path.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
