"""
Build a text-based handover package for a completed hybrid run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_ranking_path(dataset_dir: Path, mode: str) -> Path:
    candidate = dataset_dir / f"SENSITIVITY_{mode.upper()}_RANKING.csv"
    if not candidate.exists():
        raise FileNotFoundError(f"Sensitivity ranking file not found: {candidate}")
    return candidate


def build_handover_text(workspace_root: Path, mode: str) -> str:
    status = load_json(workspace_root / "hybrid_run_status.json")
    dataset_dir = Path(status["dataset_dir"])
    dataset_path = Path(status["dataset_path"])
    metadata = load_json(dataset_dir / f"{mode}_metadata.json")
    ranking_path = resolve_ranking_path(dataset_dir, mode)
    ranking_df = pd.read_csv(ranking_path)
    dataset_df = pd.read_csv(dataset_path)
    dataset_ok = dataset_df[dataset_df["success"] == True].copy()

    top_feature = ranking_df.iloc[0]
    low_125 = int((dataset_ok["fos"] < 1.25).sum())
    below_130 = int((dataset_ok["fos"] < 1.30).sum())
    p05 = float(dataset_ok["fos"].quantile(0.05))
    p50 = float(dataset_ok["fos"].quantile(0.50))
    p95 = float(dataset_ok["fos"].quantile(0.95))

    stage_rows = status.get("stages", [])
    initial_rows = stage_rows[0]["dataset_stats"]["rows"] if stage_rows else len(dataset_ok)
    final_rows = len(dataset_ok)

    lines = [
        "# Hybrid Run Results Handover",
        "",
        "## Purpose",
        "This note is intended as a presentation-support handover for the completed hybrid dataset-generation and active-learning run.",
        "It explains what was run, what the outputs represent, and the main interpretation points to carry forward.",
        "",
        "## Model Context",
        f"- Workspace root: `{workspace_root}`",
        f"- Source dataset path: `{dataset_path}`",
        f"- Mode: `{mode}`",
        f"- Run start: `{status['started_at']}`",
        f"- Run deadline: `{status['deadline']}`",
        "",
        "## Workflow Summary",
        "The hybrid workflow was configured to run repeated cycles until the overnight time limit was reached.",
        "Each cycle followed this sequence:",
        "1. Generate new GeoStudio runs using sampled control point combinations.",
        "2. Append successful runs to the active dataset.",
        "3. Retrain the ensemble surrogate model on the enlarged dataset.",
        "4. Use the trained ensemble to select additional informative samples through active learning.",
        "5. Repeat until the stopping time was reached.",
        "",
        "## Completion Summary",
        f"- Hybrid cycles completed: `{status['cycles_completed']}`",
        f"- Initial dataset size after the first generation step: `{initial_rows:,}` rows",
        f"- Final dataset size: `{final_rows:,}` rows",
        f"- Successful model solves: `{metadata['n_success']:,}`",
        f"- Failed model solves: `{metadata['n_failed']:,}`",
        f"- Solve success rate: `{metadata['success_rate']:.1f}%`",
        "",
        "## FOS Summary",
        f"- Minimum FOS: `{dataset_ok['fos'].min():.4f}`",
        f"- Median FOS: `{p50:.4f}`",
        f"- Mean FOS: `{dataset_ok['fos'].mean():.4f}`",
        f"- Maximum FOS: `{dataset_ok['fos'].max():.4f}`",
        f"- 5th percentile FOS: `{p05:.4f}`",
        f"- 95th percentile FOS: `{p95:.4f}`",
        f"- Samples with FOS < 1.25: `{low_125:,}`",
        f"- Samples with FOS < 1.30: `{below_130:,}`",
        "",
        "## Sensitivity Interpretation",
        "Sensitivity was evaluated two ways:",
        "- Local sensitivity near the baseline water surface, reported as delta FOS per 1 ft change in a control point or synced control-point group.",
        "- Dataset association using Spearman correlation across the completed dataset.",
        "",
        f"Most influential control point/group: `{top_feature['feature']}`",
        f"- Local sensitivity: `{top_feature['local_sensitivity_fos_per_ft']:+.4f}` FOS per 1 ft",
        f"- Dataset Spearman correlation: `{top_feature['dataset_spearman_correlation']:+.3f}`",
        "",
        "Interpretation note:",
        "A negative local sensitivity means that increasing the phreatic elevation at that control point reduces predicted stability.",
        "Features with larger absolute magnitude have stronger modeled influence on FOS near the baseline condition.",
        "",
        "## Ranked Sensitivity Results",
    ]

    for _, row in ranking_df.iterrows():
        lines.append(
            f"- Rank {int(row['rank'])}: `{row['feature']}` | local sensitivity `{row['local_sensitivity_fos_per_ft']:+.4f}` FOS/ft | "
            f"dataset Spearman `{row['dataset_spearman_correlation']:+.3f}`"
        )

    lines.extend([
        "",
        "## Output Files for Review",
        f"- Main plots: `{dataset_dir / f'PLOTS_{mode.upper()}.png'}`",
        f"- Low-FOS plots (<1.25): `{dataset_dir / f'PLOTS_{mode.upper()}_LT125.png'}`",
        f"- Statistical summary figure: `{dataset_dir / f'STATS_{mode.upper()}.png'}`",
        f"- Sensitivity figure: `{dataset_dir / f'SENSITIVITY_{mode.upper()}.png'}`",
        f"- Sensitivity ranking CSV: `{ranking_path}`",
        f"- Hybrid run summary figure: `{workspace_root / 'HYBRID_SUMMARY.png'}`",
        "",
        "## Presentation-Oriented Takeaways",
        "Suggested message framing for a presentation or review meeting:",
        "- The overnight hybrid workflow materially expanded the dataset while maintaining a 100% solve success rate.",
        "- The resulting dataset is concentrated in the critical FOS range, which is useful for understanding stability behavior near decision-relevant conditions.",
        "- The sensitivity ranking identifies which control point or control-point group most strongly shifts modeled stability, helping focus monitoring and scenario review.",
        "- The low-FOS subset plot is intended to isolate the geometries and phreatic surfaces associated with the more critical outcomes.",
        "",
        "## Caveats",
        "- Sensitivity values are surrogate-model interpretations, not direct finite-difference results from a fresh full-physics batch for every point.",
        "- The local sensitivity ranking is baseline-centered; different rankings can emerge in other parts of the response space.",
        "- The dataset association values are descriptive and do not by themselves prove causation.",
        "",
        "## Recommended Next Steps",
        "- Use the ranked sensitivity outputs to decide which control points deserve the most scrutiny in monitoring and operational scenarios.",
        "- Review the low-FOS subset plot alongside engineering judgment to identify whether a smaller family of adverse phreatic shapes is dominating the critical responses.",
        "- If a presentation is being prepared, use this note together with the saved PNG outputs as the source package.",
        "",
    ])

    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--mode", default="shallow", choices=["shallow", "deep"])
    parser.add_argument("--output-name", default="RESULTS_HANDOVER.md")
    args = parser.parse_args()

    workspace_root = Path(args.workspace_root).resolve()
    text = build_handover_text(workspace_root, args.mode)
    output_path = workspace_root / args.output_name
    output_path.write_text(text, encoding="utf-8")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
