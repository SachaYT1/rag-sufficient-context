"""Run a grid of experiment configs and collect a comparison table.

Usage:
    python -m run.pipeline.run_matrix configs/experiments/qwen3b_bm25.yaml \
        configs/experiments/qwen7b_hybrid.yaml configs/experiments/asymmetric.yaml

Each config is executed via ``run_experiment.run``; per-run summaries are
aggregated into ``results/matrix/summary.json`` and a Markdown comparison
table ``results/matrix/leaderboard.md``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from run.pipeline.run_experiment import run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _write_leaderboard(summaries: list[dict[str, Any]], output_path: Path) -> None:
    header = (
        "| Experiment | Correct | Abstain | Halluc. | Ans. Acc. | Baseline AURC | Proposed AURC |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    rows: list[str] = []
    for s in summaries:
        m = s.get("metrics", {})
        rows.append(
            f"| {s['experiment']} | "
            f"{m.get('correct_rate', 0):.3f} | "
            f"{m.get('abstain_rate', 0):.3f} | "
            f"{m.get('hallucinate_rate', 0):.3f} | "
            f"{m.get('answered_accuracy', 0):.3f} | "
            f"{s.get('baseline_aurc', float('nan')):.4f} | "
            f"{s.get('proposed_aurc', float('nan')):.4f} |"
        )
    output_path.write_text(header + "\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a matrix of experiment configs.")
    parser.add_argument("configs", nargs="+", help="Experiment YAML paths.")
    parser.add_argument(
        "--output_dir", default="results/matrix", help="Where to write aggregate outputs."
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for cfg_path in args.configs:
        logger.info("=== Running %s ===", cfg_path)
        try:
            summary = run(cfg_path)
            summaries.append(summary)
        except Exception as exc:  # pragma: no cover - runtime path
            logger.exception("Failed on %s: %s", cfg_path, exc)
            summaries.append({"experiment": Path(cfg_path).stem, "error": str(exc)})

    (out / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    _write_leaderboard(summaries, out / "leaderboard.md")
    logger.info("Wrote matrix outputs to %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
