# gpt_eval.py
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

from utils import (
    DIMENSIONS,
    ResponsesClient,
    score_one_dimension,
)

PRIMARY_CLASSES = [
    "Rigid_Body_&_Interaction",
    "Deformation_&_Fracture",
    "Fluid_Dynamics",
    "State_Change_&_Environment",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_existing_keys(jsonl_path: Path) -> set:
    if not jsonl_path.exists():
        return set()
    keys = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                keys.add(obj.get("_key"))
            except Exception:
                continue
    return keys


def join_explain(explain: Any, final_only_note: bool = False) -> str:
    if isinstance(explain, list):
        s = "\n".join([f"- {x}" for x in explain])
    else:
        s = str(explain)
    if final_only_note:
        s += "\n\nNote: Only evaluate whether the FINAL state is achieved in the prediction."
    return s


def build_run_fields(dp: Dict[str, Any], run_name: str) -> Dict[str, Any]:
    frames = dp["frames"]
    steps = dp["instruction"]["steps"]
    global_inst = dp["instruction"]["global"]
    explain_list = dp.get("explain", [])
    invariants = dp.get("invariants", [])

    if run_name == "run1":
        return {
            "input_frame": frames["input"],
            "ref_frame": frames["intermediate_1"],
            "instruction": steps[0],
            "explain": explain_list[0] if isinstance(explain_list, list) and len(explain_list) > 0 else "",
            "invariants": invariants,
        }
    if run_name == "run2":
        return {
            "input_frame": frames["intermediate_1"],
            "ref_frame": frames["intermediate_2"],
            "instruction": steps[1],
            "explain": explain_list[1] if isinstance(explain_list, list) and len(explain_list) > 1 else "",
            "invariants": invariants,
        }
    if run_name == "run3":
        return {
            "input_frame": frames["intermediate_2"],
            "ref_frame": frames["output"],
            "instruction": steps[2],
            "explain": explain_list[2] if isinstance(explain_list, list) and len(explain_list) > 2 else "",
            "invariants": invariants,
        }
    if run_name == "run4":
        packed_inst = (
            "Apply the following steps sequentially to reach the FINAL output state.\n"
            "Do NOT produce intermediate outputs; only the final state matters.\n\n"
            f"Step 1:\n{steps[0]}\n\nStep 2:\n{steps[1]}\n\nStep 3:\n{steps[2]}"
        )
        packed_explain = join_explain(explain_list, final_only_note=True)
        return {
            "input_frame": frames["input"],
            "ref_frame": frames["output"],
            "instruction": packed_inst,
            "explain": packed_explain,
            "invariants": invariants,
        }
    if run_name == "run5":
        explain_final = join_explain(explain_list, final_only_note=True)
        return {
            "input_frame": frames["input"],
            "ref_frame": frames["output"],
            "instruction": global_inst,
            "explain": explain_final,
            "invariants": invariants,
        }
    raise ValueError(f"Unknown run: {run_name}")


def iter_bench_datapoints(bench_root: Path) -> Iterable[Tuple[str, str, Path, Dict[str, Any]]]:
    """
    Yields: (primary, sub, meta_path, datapoint_dict)
    """
    for primary in PRIMARY_CLASSES:
        pdir = bench_root / primary
        if not pdir.exists():
            continue
        for subdir in sorted([d for d in pdir.iterdir() if d.is_dir()]):
            meta_path = subdir / "meta.json"
            if not meta_path.exists():
                continue
            dps = read_json(meta_path)
            if not isinstance(dps, list):
                continue
            for dp in dps:
                yield (primary, subdir.name, meta_path, dp)


def resolve_pred_path(
    generated_root: Path,
    model_name: str,
    primary: str,
    sub: str,
    run_name: str,
    dp_id: str,
) -> Path:
    if run_name in ("run1", "run2", "run3", "run4"):
        return generated_root / model_name / primary / sub / "step" / run_name / f"{dp_id}.png"
    if run_name == "run5":
        return generated_root / model_name / primary / sub / "global" / "run5" / f"{dp_id}.png"
    raise ValueError(run_name)


def resolve_gt_path(bench_root: Path, primary: str, sub: str, rel_path: str) -> Path:
    return bench_root / primary / sub / rel_path


def summarize_scores(jsonl_path: Path, out_path: Path) -> None:
    """
    Summarize averages with *dimension-weighted* aggregation:
    - by_dimension: plain mean per dimension (unchanged)
    - overall/by_run/by_primary_sub: compute mean per dimension first, then weighted sum
    """
    rows = []
    if not jsonl_path.exists():
        raise FileNotFoundError(jsonl_path)

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "score" not in obj:
                continue
            rows.append(obj)

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    # ---- weights (edit here if needed) ----
    dimension_weights = {
        "consistency": 0.2,
        "instruction_following": 0.3,
        "physical_plausibility": 0.4,
        "image_quality": 0.1,
    }

    # normalize weights (robust)
    s = sum(dimension_weights.get(d, 0.0) for d in DIMENSIONS)
    if s <= 0:
        raise ValueError("dimension_weights sum must be > 0")
    dimension_weights = {d: dimension_weights.get(d, 0.0) / s for d in DIMENSIONS}

    def weighted_over_rows(subrows):
        """
        Return dimension means + weighted mean.
        Weighted mean is computed over dimensions that have at least 1 sample.
        We renormalize weights over present dimensions to avoid penalizing missing dims.
        """
        dim_means = {}
        present_dims = []
        for d in DIMENSIONS:
            xs = [r["score"] for r in subrows if r.get("dimension") == d]
            m = mean(xs)
            dim_means[d] = {"count": len(xs), "mean": m}
            if m is not None:
                present_dims.append(d)

        if not present_dims:
            return {"count": 0, "weighted_mean": None, "dim_means": dim_means}

        wsum = sum(dimension_weights[d] for d in present_dims)
        weighted = sum(dim_means[d]["mean"] * (dimension_weights[d] / wsum) for d in present_dims)

        return {
            "count": len(subrows),
            "weighted_mean": weighted,
            "dim_means": dim_means,
        }

    # ---- by_dimension (plain mean, unchanged) ----
    by_dimension = {}
    for d in DIMENSIONS:
        xs = [r["score"] for r in rows if r.get("dimension") == d]
        by_dimension[d] = {"count": len(xs), "mean": mean(xs)}

    # ---- overall weighted over dimensions ----
    overall_pack = weighted_over_rows(rows)
    overall = {
        "count": overall_pack["count"],
        "weighted_mean_score": overall_pack["weighted_mean"],
    }

    # ---- by_run weighted over dimensions ----
    by_run = {}
    for run in ["run1", "run2", "run3", "run4", "run5"]:
        subrows = [r for r in rows if r.get("run") == run]
        pack = weighted_over_rows(subrows)
        by_run[run] = {"count": pack["count"], "weighted_mean": pack["weighted_mean"]}

    # ---- by_primary_sub weighted over dimensions ----
    by_primary_sub = {}
    group_map = {}
    for r in rows:
        key = f'{r.get("primary")}/{r.get("sub")}'
        group_map.setdefault(key, []).append(r)
    for k, subrows in group_map.items():
        pack = weighted_over_rows(subrows)
        by_primary_sub[k] = {"count": pack["count"], "weighted_mean": pack["weighted_mean"]}

    out = {
        "weights": dimension_weights,
        "overall": overall,
        "by_dimension": by_dimension,  # unchanged definition
        "by_run": by_run,
        "by_primary_sub": by_primary_sub,

        # optional: keep the per-dimension means used to form the overall weighted mean
        "overall_dim_means": overall_pack["dim_means"],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def eval_model(
    bench_root: Path,
    generated_root: Path,
    model_name: str,
    out_jsonl: Path,
    base_url: str,
    api_key: str,
    judge_model: str = "gpt-4o",
    skip_existing: bool = True,
    limit: int = 0,
    save_raw: bool = False,
    raw_dir: Optional[Path] = None,
) -> None:
    client = ResponsesClient(base_url=base_url, api_key=api_key, model=judge_model)

    existing = load_existing_keys(out_jsonl) if skip_existing else set()

    runs = ["run1", "run2", "run3", "run4", "run5"]
    n_done = 0

    iterator = iter_bench_datapoints(bench_root)
    if limit > 0:
        # materialize first N datapoints
        tmp = []
        for i, item in enumerate(iterator):
            tmp.append(item)
            if i + 1 >= limit:
                break
        iterator = tmp

    for primary, sub, meta_path, dp in tqdm(list(iterator), desc=f"Eval {model_name}", total=None):
        dp_id = str(dp.get("id"))
        for run_name in runs:
            pred_path = resolve_pred_path(generated_root, model_name, primary, sub, run_name, dp_id)
            if not pred_path.exists():
                # No generated image; skip
                continue

            run_fields = build_run_fields(dp, run_name)
            input_img = resolve_gt_path(bench_root, primary, sub, run_fields["input_frame"])
            ref_img = resolve_gt_path(bench_root, primary, sub, run_fields["ref_frame"])
            instruction = run_fields["instruction"]
            explain = run_fields["explain"]
            invariants = run_fields["invariants"]

            for dim in DIMENSIONS:
                key = f"{model_name}|{primary}|{sub}|{dp_id}|{run_name}|{dim}"
                if skip_existing and key in existing:
                    continue

                raw_path = None
                if save_raw:
                    rd = raw_dir or (out_jsonl.parent / "_raw")
                    raw_path = rd / model_name / primary / sub / run_name / f"{dp_id}_{dim}.json"

                try:
                    result = score_one_dimension(
                        client=client,
                        dimension=dim,
                        input_img=input_img,
                        ref_img=ref_img,
                        pred_img=pred_path,
                        instruction=instruction,
                        explain=explain,
                        invariants=invariants,
                        save_raw_path=raw_path,
                    )

                    row = {
                        "_key": key,
                        "model_name": model_name,
                        "judge_model": judge_model,
                        "primary": primary,
                        "sub": sub,
                        "id": dp_id,
                        "run": run_name,
                        "dimension": dim,
                        "score": result["score"],
                        "reason": result["reason"],
                        "paths": {
                            "input": str(input_img),
                            "ref": str(ref_img),
                            "pred": str(pred_path),
                        },
                    }
                    write_jsonl(out_jsonl, row)
                    existing.add(key)
                    n_done += 1

                except Exception as e:
                    row = {
                        "_key": key,
                        "model_name": model_name,
                        "judge_model": judge_model,
                        "primary": primary,
                        "sub": sub,
                        "id": dp_id,
                        "run": run_name,
                        "dimension": dim,
                        "error": repr(e),
                        "paths": {
                            "input": str(input_img),
                            "ref": str(ref_img),
                            "pred": str(pred_path),
                        },
                    }
                    write_jsonl(out_jsonl, row)
                    existing.add(key)
                    n_done += 1

    print(f"Finished. wrote {n_done} records to {out_jsonl}")
    summary_path = out_jsonl.with_suffix(".summary.json")
    summarize_scores(out_jsonl, summary_path)
    print(f"Summary saved to {summary_path}")


def test_once(
    base_url: str,
    api_key: str,
    judge_model: str,
    input_img: Path,
    ref_img: Path,
    pred_img: Path,
    instruction: str,
    explain: str,
    invariants: List[str],
    out_json: Optional[Path] = None,
    save_raw_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    client = ResponsesClient(base_url=base_url, api_key=api_key, model=judge_model)
    results: Dict[str, Any] = {}

    for dim in DIMENSIONS:
        raw_path = None
        if save_raw_dir:
            save_raw_dir.mkdir(parents=True, exist_ok=True)
            raw_path = save_raw_dir / f"test_{dim}.json"

        r = score_one_dimension(
            client=client,
            dimension=dim,
            input_img=input_img,
            ref_img=ref_img,
            pred_img=pred_img,
            instruction=instruction,
            explain=explain,
            invariants=invariants,
            save_raw_path=raw_path,
        )
        results[dim] = {"score": r["score"], "reason": r["reason"]}

    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    return results


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_eval = sub.add_parser("eval", help="Evaluate a model's generated images against bench GT")
    ap_eval.add_argument("--bench_root", type=str, default="./bench")
    ap_eval.add_argument("--generated_root", type=str, default="./bench_generated")
    ap_eval.add_argument("--model_name", type=str, required=True)
    ap_eval.add_argument("--out", type=str, default="./bench_scores/scores.jsonl")
    ap_eval.add_argument("--base_url", type=str, default=os.environ.get("NEWAPI_BASE_URL", "https://yinli.one"))
    ap_eval.add_argument("--api_key", type=str, default=os.environ.get("NEWAPI_API_KEY", ""))
    ap_eval.add_argument("--judge_model", type=str, default="gpt-4o")
    ap_eval.add_argument("--skip_existing", action="store_true")
    ap_eval.add_argument("--limit", type=int, default=0)
    ap_eval.add_argument("--save_raw", action="store_true")
    ap_eval.add_argument("--raw_dir", type=str, default="")

    ap_test = sub.add_parser("test", help="Test scoring on one run without bench_generated layout")
    ap_test.add_argument("--base_url", type=str, default=os.environ.get("NEWAPI_BASE_URL", "https://yinli.one"))
    ap_test.add_argument("--api_key", type=str, default=os.environ.get("NEWAPI_API_KEY", ""))
    ap_test.add_argument("--judge_model", type=str, default="gpt-4o")
    ap_test.add_argument("--input_img", type=str, required=True)
    ap_test.add_argument("--ref_img", type=str, required=True)
    ap_test.add_argument("--pred_img", type=str, required=True)
    ap_test.add_argument("--instruction", type=str, required=True)
    ap_test.add_argument("--explain", type=str, default="")
    ap_test.add_argument("--invariants_json", type=str, default="[]")
    ap_test.add_argument("--out_json", type=str, default="./bench_scores/test_result.json")
    ap_test.add_argument("--save_raw_dir", type=str, default="./bench_scores/_raw_test")

    args = ap.parse_args()

    if args.cmd == "eval":
        if not args.api_key:
            raise SystemExit("Missing api_key. Set NEWAPI_API_KEY or pass --api_key.")
        raw_dir = Path(args.raw_dir) if args.raw_dir else None
        eval_model(
            bench_root=Path(args.bench_root).resolve(),
            generated_root=Path(args.generated_root).resolve(),
            model_name=args.model_name,
            out_jsonl=Path(args.out).resolve(),
            base_url=args.base_url,
            api_key=args.api_key,
            judge_model=args.judge_model,
            skip_existing=args.skip_existing,
            limit=args.limit,
            save_raw=args.save_raw,
            raw_dir=raw_dir,
        )

    elif args.cmd == "test":
        if not args.api_key:
            raise SystemExit("Missing api_key. Set NEWAPI_API_KEY or pass --api_key.")
        invariants = json.loads(args.invariants_json)
        res = test_once(
            base_url=args.base_url,
            api_key=args.api_key,
            judge_model=args.judge_model,
            input_img=Path(args.input_img).resolve(),
            ref_img=Path(args.ref_img).resolve(),
            pred_img=Path(args.pred_img).resolve(),
            instruction=args.instruction,
            explain=args.explain,
            invariants=invariants if isinstance(invariants, list) else [],
            out_json=Path(args.out_json).resolve() if args.out_json else None,
            save_raw_dir=Path(args.save_raw_dir).resolve() if args.save_raw_dir else None,
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()