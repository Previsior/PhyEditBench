# gpt_eval.py
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

from utils import (
    DIMENSIONS,
    TYPE_ORDER,
    ResponsesClient,
    load_existing_keys,
    score_one_dimension,
    summarize_normal_scores,
    write_jsonl,
    write_summary,
)

PRIMARY_CLASSES = [
    "Rigid_Body_&_Interaction",
    "Deformation_&_Fracture",
    "Fluid_Dynamics",
    "State_Change_&_Environment",
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "model"


def parse_choice_list(value: str, choices: List[str], label: str) -> List[str]:
    if not value or value.lower() == "all":
        return choices
    selected = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in selected if item not in choices]
    if invalid:
        raise SystemExit(f"Invalid {label}: {invalid}. Valid values: {', '.join(choices)}")
    return selected


def join_explain(explain: Any, final_only_note: bool = False) -> str:
    if isinstance(explain, list):
        text = "\n".join([f"- {x}" for x in explain])
    else:
        text = str(explain)
    if final_only_note:
        text += "\n\nNote: Only evaluate whether the FINAL state is achieved in the prediction."
    return text


def build_type_fields(dp: Dict[str, Any], edit_type: str) -> Dict[str, Any]:
    frames = dp["frames"]
    steps = dp["instruction"]["steps"]
    global_inst = dp["instruction"]["global"]
    explain_list = dp.get("explain", [])
    invariants = dp.get("invariants", [])

    if edit_type == "TypeA":
        return {
            "input_frame": frames["input"],
            "ref_frame": frames["intermediate_1"],
            "instruction": steps[0],
            "explain": explain_list[0] if isinstance(explain_list, list) and len(explain_list) > 0 else "",
            "invariants": invariants,
        }
    if edit_type == "TypeB":
        return {
            "input_frame": frames["intermediate_1"],
            "ref_frame": frames["intermediate_2"],
            "instruction": steps[1],
            "explain": explain_list[1] if isinstance(explain_list, list) and len(explain_list) > 1 else "",
            "invariants": invariants,
        }
    if edit_type == "TypeC":
        return {
            "input_frame": frames["intermediate_2"],
            "ref_frame": frames["output"],
            "instruction": steps[2],
            "explain": explain_list[2] if isinstance(explain_list, list) and len(explain_list) > 2 else "",
            "invariants": invariants,
        }
    if edit_type == "TypeD":
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
    if edit_type == "TypeE":
        explain_final = join_explain(explain_list, final_only_note=True)
        return {
            "input_frame": frames["input"],
            "ref_frame": frames["output"],
            "instruction": global_inst,
            "explain": explain_final,
            "invariants": invariants,
        }
    raise ValueError(f"Unknown edit type: {edit_type}")


def iter_bench_datapoints(bench_root: Path) -> Iterable[Tuple[str, str, Path, Dict[str, Any]]]:
    """
    Yields: (primary, sub, meta_path, datapoint_dict)
    """
    for primary in PRIMARY_CLASSES:
        primary_dir = bench_root / primary
        if not primary_dir.exists():
            continue
        for subdir in sorted([d for d in primary_dir.iterdir() if d.is_dir()]):
            meta_path = subdir / "meta.json"
            if not meta_path.exists():
                continue
            datapoints = read_json(meta_path)
            if not isinstance(datapoints, list):
                continue
            for dp in datapoints:
                yield (primary, subdir.name, meta_path, dp)


def resolve_pred_path(
    generated_root: Path,
    model_name: str,
    primary: str,
    sub: str,
    edit_type: str,
    dp_id: str,
) -> Path:
    return generated_root / model_name / primary / sub / edit_type / f"{dp_id}.png"


def resolve_gt_path(bench_root: Path, primary: str, sub: str, rel_path: str) -> Path:
    return bench_root / primary / sub / rel_path


def default_output_paths(output_dir: Path, model_name: str) -> Tuple[Path, Path]:
    name = safe_filename(model_name)
    return output_dir / f"{name}_normal_scores.jsonl", output_dir / f"{name}_normal_summary.json"


def eval_model(
    bench_root: Path,
    generated_root: Path,
    model_name: str,
    out_jsonl: Path,
    summary_path: Path,
    api_key: Optional[str],
    base_url: Optional[str],
    judge_model: str = "gpt-4o",
    skip_existing: bool = True,
    limit: int = 0,
    edit_types: Optional[List[str]] = None,
    dimensions: Optional[List[str]] = None,
    save_raw: bool = False,
    raw_dir: Optional[Path] = None,
    dry_run: bool = False,
) -> None:
    selected_types = edit_types or TYPE_ORDER
    selected_dimensions = dimensions or DIMENSIONS
    existing = load_existing_keys(out_jsonl) if skip_existing and not dry_run else set()

    datapoints = list(iter_bench_datapoints(bench_root))
    if limit > 0:
        datapoints = datapoints[:limit]

    client = None
    if not dry_run:
        client = ResponsesClient(api_key=api_key, base_url=base_url, model=judge_model)

    n_written = 0
    n_planned = 0
    n_missing = 0

    for primary, sub, _, dp in tqdm(datapoints, desc=f"Eval {model_name}", total=len(datapoints)):
        dp_id = str(dp.get("id"))
        for edit_type in selected_types:
            pred_path = resolve_pred_path(generated_root, model_name, primary, sub, edit_type, dp_id)
            if not pred_path.exists():
                n_missing += 1
                continue

            type_fields = build_type_fields(dp, edit_type)
            input_img = resolve_gt_path(bench_root, primary, sub, type_fields["input_frame"])
            ref_img = resolve_gt_path(bench_root, primary, sub, type_fields["ref_frame"])
            instruction = type_fields["instruction"]
            explain = type_fields["explain"]
            invariants = type_fields["invariants"]

            for dimension in selected_dimensions:
                key = f"{model_name}|{primary}|{sub}|{dp_id}|{edit_type}|{dimension}"
                if skip_existing and key in existing:
                    continue

                n_planned += 1
                if dry_run:
                    continue

                raw_path = None
                if save_raw:
                    raw_base = raw_dir or (out_jsonl.parent / "_raw")
                    raw_path = raw_base / model_name / primary / sub / edit_type / f"{dp_id}_{dimension}.json"

                try:
                    assert client is not None
                    result = score_one_dimension(
                        client=client,
                        dimension=dimension,
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
                        "type": edit_type,
                        "dimension": dimension,
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
                    n_written += 1

                except Exception as exc:
                    row = {
                        "_key": key,
                        "model_name": model_name,
                        "judge_model": judge_model,
                        "primary": primary,
                        "sub": sub,
                        "id": dp_id,
                        "type": edit_type,
                        "dimension": dimension,
                        "error": repr(exc),
                        "paths": {
                            "input": str(input_img),
                            "ref": str(ref_img),
                            "pred": str(pred_path),
                        },
                    }
                    write_jsonl(out_jsonl, row)
                    existing.add(key)
                    n_written += 1

    if dry_run:
        print(f"Dry run complete. Planned scoring calls: {n_planned}; missing generated images: {n_missing}.")
        return

    summary = summarize_normal_scores(out_jsonl)
    write_summary(summary, summary_path)
    print(f"Finished. Wrote {n_written} records to {out_jsonl}")
    print(f"Summary saved to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate normal PhyEditBench samples.")
    parser.add_argument("--bench_root", type=str, default="./bench")
    parser.add_argument("--generated_root", type=str, default="./bench_generated")
    parser.add_argument("--model_name", "--model", type=str, required=True)
    parser.add_argument("--output", "--output_dir", dest="output_dir", type=str, default="./bench_scores")
    parser.add_argument("--scores_path", type=str, default="")
    parser.add_argument("--summary_path", type=str, default="")
    parser.add_argument("--base_url", type=str, default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--api_key", type=str, default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--judge_model", type=str, default="gpt-4o")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--types", type=str, default="all", help="Comma-separated TypeA-TypeE list, or all")
    parser.add_argument("--dimensions", type=str, default="all", help="Comma-separated dimension list, or all")
    parser.add_argument("--save_raw", action="store_true")
    parser.add_argument("--raw_dir", type=str, default="")
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    default_scores_path, default_summary_path = default_output_paths(output_dir, args.model_name)
    scores_path = Path(args.scores_path).resolve() if args.scores_path else default_scores_path
    summary_path = Path(args.summary_path).resolve() if args.summary_path else default_summary_path
    edit_types = parse_choice_list(args.types, TYPE_ORDER, "types")
    dimensions = parse_choice_list(args.dimensions, DIMENSIONS, "dimensions")

    if not args.dry_run and not args.api_key:
        raise SystemExit("Missing API key. Set OPENAI_API_KEY or pass --api_key.")

    eval_model(
        bench_root=Path(args.bench_root).resolve(),
        generated_root=Path(args.generated_root).resolve(),
        model_name=args.model_name,
        out_jsonl=scores_path,
        summary_path=summary_path,
        api_key=args.api_key or None,
        base_url=args.base_url or None,
        judge_model=args.judge_model,
        skip_existing=not args.overwrite,
        limit=args.limit,
        edit_types=edit_types,
        dimensions=dimensions,
        save_raw=args.save_raw,
        raw_dir=Path(args.raw_dir).resolve() if args.raw_dir else None,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
