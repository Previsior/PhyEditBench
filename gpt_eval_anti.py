# gpt_eval_anti.py
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from utils import (
    encode_image_base64,
    load_existing_by_field,
    make_openai_client,
    parse_anti_score_content,
    read_jsonl,
    response_to_jsonable,
    summarize_anti_scores,
    write_jsonl,
    write_summary,
)


prompt_1 = """ 
Use the input image, editing prompt and the Physical Plausibility(the expected result of the editing prompt on the input image) to create the checklist used to judge the image edited on it(mandatory).  
(Note: The editing prompt is related to unreasonable scene that against the real world)
1. VIOLATED PHYSIC LAW: Which physic law is related to the editing prompt?
2. INVOLVED OBJECTS: What are the main objects interacted with the editing prompt within the image?
3. EXPECTED PHENOMENA: What is the expected editing result of the editing prompt with respect to the involved objects(detailed and precise phenomena of the involved objects)?

Guidelines for checklist creation: 
- only target things which are visually observable in the image 
- the statements in checklist needs to be assertive statements instead of questions
- only checklist no other content
"""

rubric_text = """
You are strict VLM-Judge objectively evaluating a generated edited image from a checklist, editing prompt and the input image. 
The checklist is provided as a reference :
1. VIOLATED PHYSIC LAW
2. INVOLVED OBJECTS
3. EXPECTED PHENOMENA

Score each rubric from 1–10: 
a) Instruction Following — Judge how well the prediction satisfies the instruction AND matches the reference target state.
b) Physical Plausibility — expected physical/chemical outcome is present according to the EXPECTED PHENOMENA(check the image in detail).
c) Consistency — whether the involved objects are the same object between the input image and the edited image beside some changes according to the EXPECTED PHENOMENA. 
d) Image Quality — The generated image should exhibit sharp details with minimal artifacts, distortions, or unnatural textures. Lighting, shadows, and reflections must remain physically coherent, while materials should demonstrate photorealistic appearance and visual consistency. Overall, the output should be free from obvious AI-generated artifacts or technical glitches.

Each rating must be supported with clear justification, drawing on specific edited image area and, when provided, the corresponding checklist items.
Important!: Give each rubic at least 1 point even it completely fail.
Scoring rubric for Consistency(1-10):
- 10: Only the intended edits occur; invariants and unrelated regions are preserved extremely well.
- 7-9: Minor unintended changes (small texture shifts, slight lighting drift), but overall consistent.
- 4-6: Noticeable unintended changes (background altered, viewpoint drift, extra objects), partially consistent.
- 1-3: Major unwanted changes; scene identity not preserved.
Scoring rubric for Instruction Following(1-10):
- 10: Prediction matches the reference target very closely and fulfills the instruction precisely.
- 7-9: Mostly correct; small differences from reference target but clearly follows instruction.
- 4-6: Partially correct; key aspects missing or wrong; noticeable mismatch vs reference.
- 1-3: Fails to perform the intended edit; does not resemble the reference target.
Scoring rubric for Image Quality(1-10):
- 10: Highly realistic, sharp where appropriate, no noticeable artifacts.
- 7-9: Minor artifacts or softness, overall high quality.
- 4-6: Clear artifacts, blur, distortions, but still recognizable.
- 1-3: Severe artifacts, unrealistic, degraded output.
"""

output_format = """
Return JSON with fields: 
{ ”scores”: { ”Instruction_Following”:1-10, ”Physical_Plausibility”:1-10, ”Consistency”:1-10, ”Image_Quality”:1-10}, 
 ”explanations”: {”summary”: string, ”issues”: [{"issue_name": string,"score_explanation":string}...]} ## issue_name's value should one of: Instruction Following, Physical Plausibility, Consistency and Image Quality
 }
"""


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "model"


def default_output_paths(output_dir: Path, model_name: str) -> tuple[Path, Path]:
    name = safe_filename(model_name)
    return output_dir / f"{name}_anti_scores.jsonl", output_dir / f"{name}_anti_summary.json"


def save_raw_response(resp: Any, path: Optional[Path]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(response_to_jsonable(resp), ensure_ascii=False, indent=2), encoding="utf-8")


def generate_checklist(
    client: Any,
    judge_model: str,
    editing_prompt: str,
    input_image: str,
    expected_phenomena: str,
    save_raw_path: Optional[Path] = None,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"From my evaluation of editing models I have generated a image using the prompt: {editing_prompt}"},
                {"type": "text", "text": "here is the input image"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{input_image}"},
                },
                {"type": "text", "text": "here is the Physical Plausibility"},
                {"type": "text", "text": expected_phenomena},
                {"type": "text", "text": prompt_1},
            ],
        },
    ]

    completion = client.chat.completions.create(
        model=judge_model,
        messages=messages,
        stream=False,
    )
    save_raw_response(completion, save_raw_path)

    if not completion.choices:
        return ""
    return completion.choices[0].message.content or ""


def vlm_judge(
    client: Any,
    editing_prompt: str,
    edited_image: str,
    input_image: str,
    check_list: str,
    judge_model: str,
    save_raw_path: Optional[Path] = None,
) -> str:
    try:
        completion = client.chat.completions.create(
            model=judge_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Here is the editing prompt :{editing_prompt}"},
                        {"type": "text", "text": "here is the input image:"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{input_image}"},
                        },
                        {"type": "text", "text": "here is the edited image:"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{edited_image}"},
                        },
                        {"type": "text", "text": f"here is the checklist:{check_list}"},
                        {"type": "text", "text": rubric_text},
                        {"type": "text", "text": f"here is the output format:{output_format}"},
                    ],
                },
            ],
            stream=False,
        )
        save_raw_response(completion, save_raw_path)

        if not completion.choices:
            print("Warning: API returned no choices for this image.")
            return json.dumps({
                "scores": {"Instruction_Following": 1, "Physical_Plausibility": 1, "Consistency": 1, "Image_Quality": 1},
                "explanations": {"summary": "API Refusal/Error", "issues": []},
            })

        full_message = completion.choices[0].message
        if getattr(full_message, "refusal", None):
            print(f"Warning: Request refused by safety system: {full_message.refusal}")
            return json.dumps({
                "scores": {"Instruction_Following": 1, "Physical_Plausibility": 1, "Consistency": 1, "Image_Quality": 1},
                "explanations": {"summary": "Content Policy Violation", "issues": []},
            })

        return full_message.content or ""

    except Exception as exc:
        print(f"CRITICAL ERROR in vlm_judge: {exc}")
        return json.dumps({
            "scores": {"Instruction_Following": 0, "Physical_Plausibility": 0, "Consistency": 0, "Image_Quality": 0},
            "explanations": {"summary": f"Python Error: {str(exc)}", "issues": []},
        })


def resolve_pred_path(generated_root: Path, model_name: str, data_id: int) -> Path:
    return generated_root / model_name / "anti-physic" / f"{data_id}.png"


def process_dataset(
    bench_root: Path,
    generated_root: Path,
    model_name: str,
    scores_path: Path,
    summary_path: Path,
    checklist_path: Path,
    api_key: Optional[str],
    base_url: Optional[str],
    judge_model: str = "gpt-4o",
    skip_existing: bool = True,
    limit: int = 0,
    generate_missing_checklists: bool = False,
    save_raw: bool = False,
    raw_dir: Optional[Path] = None,
    dry_run: bool = False,
) -> None:
    meta_file_path = bench_root / "meta.jsonl"
    if not meta_file_path.exists():
        raise FileNotFoundError(meta_file_path)

    meta_rows = read_jsonl(meta_file_path)
    if limit > 0:
        meta_rows = meta_rows[:limit]

    processed_checklists = load_existing_by_field(checklist_path, "data_id")
    processed_scores = load_existing_by_field(scores_path, "data_id") if skip_existing and not dry_run else {}

    client = None
    if not dry_run:
        client = make_openai_client(api_key=api_key, base_url=base_url)

    n_planned = 0
    n_written = 0
    n_missing = 0

    for data in tqdm(meta_rows, desc=f"Eval anti-physics {model_name}", total=len(meta_rows)):
        data_id = data["data_id"]
        edit_prompt = data["edit_prompt"]
        expected_phenomena = data["expected_phenomenon"]
        input_image_path = bench_root / "input_data" / f"data_{data_id}.png"
        output_image_path = resolve_pred_path(generated_root, model_name, data_id)

        if not input_image_path.exists():
            raise FileNotFoundError(input_image_path)
        if not output_image_path.exists():
            n_missing += 1
            continue
        if skip_existing and data_id in processed_scores:
            continue

        n_planned += 1
        if dry_run:
            continue

        checklist_data = processed_checklists.get(data_id)
        if checklist_data is None:
            if not generate_missing_checklists:
                raise RuntimeError(
                    f"Missing checklist for data_id {data_id}. "
                    f"Expected it in {checklist_path}, or rerun with --generate_missing_checklists."
                )
            assert client is not None
            raw_path = None
            if save_raw:
                raw_base = raw_dir or (scores_path.parent / "_raw")
                raw_path = raw_base / model_name / "anti-physic" / f"{data_id}_checklist.json"
            checklist = generate_checklist(
                client=client,
                judge_model=judge_model,
                editing_prompt=edit_prompt,
                input_image=encode_image_base64(input_image_path),
                expected_phenomena=expected_phenomena,
                save_raw_path=raw_path,
            )
            checklist_data = {
                "data_id": data_id,
                "data_type": data["data_type"],
                "sub_id": data["sub_id"],
                "checklist": checklist,
            }
            write_jsonl(checklist_path, checklist_data)
            processed_checklists[data_id] = checklist_data
        else:
            checklist = checklist_data["checklist"]

        assert client is not None
        raw_path = None
        if save_raw:
            raw_base = raw_dir or (scores_path.parent / "_raw")
            raw_path = raw_base / model_name / "anti-physic" / f"{data_id}_score.json"

        score_content = vlm_judge(
            client=client,
            editing_prompt=edit_prompt,
            edited_image=encode_image_base64(output_image_path),
            input_image=encode_image_base64(input_image_path),
            check_list=checklist,
            judge_model=judge_model,
            save_raw_path=raw_path,
        )
        parsed_scores = parse_anti_score_content(score_content)

        row = {
            "_key": f"{model_name}|anti-physic|{data_id}",
            "model_name": model_name,
            "judge_model": judge_model,
            "data_id": data_id,
            "data_type": data["data_type"],
            "sub_id": data["sub_id"],
            "Instruction_Following": parsed_scores["Instruction_Following"],
            "Physical_Plausibility": parsed_scores["Physical_Plausibility"],
            "Consistency": parsed_scores["Consistency"],
            "Image_Quality": parsed_scores["Image_Quality"],
            "summary": parsed_scores["summary"],
            "issues": parsed_scores["issues"],
            "paths": {
                "input": str(input_image_path),
                "pred": str(output_image_path),
            },
        }
        write_jsonl(scores_path, row)
        processed_scores[data_id] = row
        n_written += 1

    if dry_run:
        print(f"Dry run complete. Planned judge calls: {n_planned}; missing generated images: {n_missing}.")
        return

    summary = summarize_anti_scores(scores_path)
    write_summary(summary, summary_path)
    print(f"Finished. Wrote {n_written} records to {scores_path}")
    print(f"Summary saved to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate anti-physics samples in PhyEditBench.")
    parser.add_argument("--bench_root", type=str, default="./bench/anti-physic")
    parser.add_argument("--generated_root", type=str, default="./bench_generated")
    parser.add_argument("--model_name", "--model", type=str, required=True)
    parser.add_argument("--output", "--output_dir", dest="output_dir", type=str, default="./bench_scores")
    parser.add_argument("--scores_path", type=str, default="")
    parser.add_argument("--summary_path", type=str, default="")
    parser.add_argument("--checklist_path", type=str, default="")
    parser.add_argument("--base_url", type=str, default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--api_key", type=str, default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--judge_model", type=str, default="gpt-4o")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--generate_missing_checklists", action="store_true")
    parser.add_argument("--save_raw", action="store_true")
    parser.add_argument("--raw_dir", type=str, default="")
    parser.add_argument("--dry_run", action="store_true")

    args = parser.parse_args()

    bench_root = Path(args.bench_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    default_scores_path, default_summary_path = default_output_paths(output_dir, args.model_name)
    scores_path = Path(args.scores_path).resolve() if args.scores_path else default_scores_path
    summary_path = Path(args.summary_path).resolve() if args.summary_path else default_summary_path
    checklist_path = Path(args.checklist_path).resolve() if args.checklist_path else bench_root / "checklists.jsonl"

    if not args.dry_run and not args.api_key:
        raise SystemExit("Missing API key. Set OPENAI_API_KEY or pass --api_key.")

    process_dataset(
        bench_root=bench_root,
        generated_root=Path(args.generated_root).resolve(),
        model_name=args.model_name,
        scores_path=scores_path,
        summary_path=summary_path,
        checklist_path=checklist_path,
        api_key=args.api_key or None,
        base_url=args.base_url or None,
        judge_model=args.judge_model,
        skip_existing=not args.overwrite,
        limit=args.limit,
        generate_missing_checklists=args.generate_missing_checklists,
        save_raw=args.save_raw,
        raw_dir=Path(args.raw_dir).resolve() if args.raw_dir else None,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

