# summarize_scores.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

DIMENSIONS = ["consistency", "instruction_following", "physical_plausibility", "image_quality"]
RUNS = ["run1", "run2", "run3", "run4", "run5"]

dimension_weights = {
    "consistency": 0.2,
    "instruction_following": 0.3,
    "physical_plausibility": 0.4,
    "image_quality": 0.1,
}


def mean(xs: List[float]) -> float | None:
    return (sum(xs) / len(xs)) if xs else None


def load_rows(jsonl_path: Path) -> List[Dict[str, Any]]:
    rows = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # only keep successful score rows
            if "score" not in obj:
                continue
            rows.append(obj)
    return rows


def normalized_weights() -> Dict[str, float]:
    s = sum(dimension_weights.get(d, 0.0) for d in DIMENSIONS)
    if s <= 0:
        raise ValueError("dimension_weights sum must be > 0")
    return {d: dimension_weights.get(d, 0.0) / s for d in DIMENSIONS}


def weighted_mean_over_dimensions(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute per-dimension mean first, then weighted sum across dimensions.
    Renormalize weights over dimensions that are present (have >=1 sample).
    """
    w = normalized_weights()
    dim_means = {}
    present = []

    for d in DIMENSIONS:
        xs = [r["score"] for r in rows if r.get("dimension") == d]
        m = mean(xs)
        dim_means[d] = {"count": len(xs), "mean": m}
        if m is not None:
            present.append(d)

    if not present:
        return {"count": 0, "weighted_mean": None, "dim_means": dim_means}

    wsum = sum(w[d] for d in present)
    weighted = sum(dim_means[d]["mean"] * (w[d] / wsum) for d in present)

    return {"count": len(rows), "weighted_mean": weighted, "dim_means": dim_means}


def summarize(jsonl_path: Path) -> Dict[str, Any]:
    rows = load_rows(jsonl_path)

    # by_dimension (plain mean, not weighted)
    by_dimension = {}
    for d in DIMENSIONS:
        xs = [r["score"] for r in rows if r.get("dimension") == d]
        by_dimension[d] = {"count": len(xs), "mean": mean(xs)}

    # overall (dimension-weighted)
    overall_pack = weighted_mean_over_dimensions(rows)
    overall = {
        "count": overall_pack["count"],
        "weighted_mean": overall_pack["weighted_mean"],
    }

    # by_run (dimension-weighted)
    by_run = {}
    for run in RUNS:
        sub = [r for r in rows if r.get("run") == run]
        pack = weighted_mean_over_dimensions(sub)
        by_run[run] = {"count": pack["count"], "weighted_mean": pack["weighted_mean"]}

    # by_primary (dimension-weighted)
    primary_map: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        primary = r.get("primary", "UNKNOWN")
        primary_map.setdefault(primary, []).append(r)

    by_primary = {}
    for p, subrows in primary_map.items():
        pack = weighted_mean_over_dimensions(subrows)
        by_primary[p] = {"count": pack["count"], "weighted_mean": pack["weighted_mean"]}

    # by_primary_sub (dimension-weighted)
    ps_map: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        key = f"{r.get('primary','UNKNOWN')}/{r.get('sub','UNKNOWN')}"
        ps_map.setdefault(key, []).append(r)

    by_primary_sub = {}
    for k, subrows in ps_map.items():
        pack = weighted_mean_over_dimensions(subrows)
        by_primary_sub[k] = {"count": pack["count"], "weighted_mean": pack["weighted_mean"]}

    return {
        "source": str(jsonl_path),
        "weights": normalized_weights(),
        "overall": overall,
        "by_dimension": by_dimension,  # plain mean per dimension
        "by_run": by_run,
        "by_primary": by_primary,
        "by_primary_sub": by_primary_sub,
        # optional: show dim means used for overall weighted mean
        "overall_dim_means": overall_pack["dim_means"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=str, required=True, help="Path to scores.jsonl")
    ap.add_argument("--out", type=str, required=True, help="Output summary json path")
    args = ap.parse_args()

    scores_path = Path(args.scores).resolve()
    out_path = Path(args.out).resolve()

    summary = summarize(scores_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved summary to: {out_path}")


if __name__ == "__main__":
    main()