#!/usr/bin/env python3
"""Offline summary for pipeline evaluation CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Dict, List


def _as_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def summarize(metrics_csv: Path) -> Dict[str, float]:
    rows: List[Dict[str, str]] = []
    with metrics_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return {"plans": 0.0}

    dt_text_to_done = [_as_float(r.get("dt_text_to_done", "")) for r in rows]
    dt_text_to_plan = [_as_float(r.get("dt_text_to_plan", "")) for r in rows]
    dt_plan_to_done = [_as_float(r.get("dt_plan_to_done", "")) for r in rows]
    llm_total = [_as_float(r.get("llm_total", "")) for r in rows]
    num_steps = [_as_float(r.get("num_steps", "")) for r in rows]
    cer_values = [_as_float(r.get("cer", "")) for r in rows if r.get("cer", "") not in ("", None)]
    plan_success = [_as_float(r.get("plan_success", "")) for r in rows if r.get("plan_success", "") not in ("", None)]
    execution_success = [
        _as_float(r.get("execution_success", "")) for r in rows if r.get("execution_success", "") not in ("", None)
    ]

    summary = {
        "plans": float(len(rows)),
        "avg_steps": mean(num_steps),
        "avg_dt_text_to_plan_s": mean(dt_text_to_plan),
        "avg_dt_plan_to_done_s": mean(dt_plan_to_done),
        "avg_dt_text_to_done_s": mean(dt_text_to_done),
        "avg_llm_total_s": mean(llm_total),
        "avg_cer": mean(cer_values) if cer_values else 0.0,
        "plan_success_rate": mean(plan_success) if plan_success else 0.0,
        "execution_success_rate": mean(execution_success) if execution_success else 0.0,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize pipeline evaluation metrics CSV.")
    parser.add_argument(
        "--metrics-csv",
        default="~/ros2_ws/pipeline_eval_metrics.csv",
        help="Path to per-plan metrics CSV.",
    )
    args = parser.parse_args()

    metrics_csv = Path(args.metrics_csv).expanduser()
    if not metrics_csv.exists():
        raise SystemExit(f"metrics CSV not found: {metrics_csv}")

    out = summarize(metrics_csv)
    print("Pipeline evaluation summary")
    print(f"- plans: {int(out['plans'])}")
    if out["plans"] > 0:
        print(f"- avg_steps: {out['avg_steps']:.2f}")
        print(f"- avg_dt_text_to_plan_s: {out['avg_dt_text_to_plan_s']:.3f}")
        print(f"- avg_dt_plan_to_done_s: {out['avg_dt_plan_to_done_s']:.3f}")
        print(f"- avg_dt_text_to_done_s: {out['avg_dt_text_to_done_s']:.3f}")
        print(f"- avg_llm_total_s: {out['avg_llm_total_s']:.3f}")
        print(f"- avg_cer: {out['avg_cer']:.4f}")
        print(f"- plan_success_rate: {out['plan_success_rate']:.3f}")
        print(f"- execution_success_rate: {out['execution_success_rate']:.3f}")


if __name__ == "__main__":
    main()
