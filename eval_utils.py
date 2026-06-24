from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from PIL import Image

from image_normalize import normalize_image


SEED = 2026
NUM_INFERENCE_STEPS = 40

CONSTRAINTS_BLOCK = (
    "Constraints (must follow):\n"
    "- Keep the camera viewpoint unchanged.\n"
    "- Keep the background unchanged.\n"
    "- Do not add text, logos, watermarks, UI elements, or borders.\n"
    "- Preserve the overall realism and natural appearance."
)


@dataclass(frozen=True)
class NormalDatapoint:
    sample_id: str
    primary_class: str
    sub_class: str
    input_image: Path
    intermediate_1_image: Path
    intermediate_2_image: Path
    output_image: Path
    step_1_instruction: str
    step_2_instruction: str
    step_3_instruction: str
    global_instruction: str


@dataclass(frozen=True)
class AntiPhysicDatapoint:
    data_id: str
    data_type: str
    input_image: Path
    edit_prompt: str


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_device(device: str) -> str:
    device = str(device).strip()
    if device.isdigit():
        return f"cuda:{device}"
    return device


def discover_normal_datapoints(bench_root: Path) -> List[NormalDatapoint]:
    bench_root = bench_root.resolve()
    samples: List[NormalDatapoint] = []

    for primary_dir in sorted(bench_root.iterdir(), key=lambda p: p.name):
        if not primary_dir.is_dir():
            continue
        if primary_dir.name in {"anti-physic", ".cache"}:
            continue
        for sub_dir in sorted(primary_dir.iterdir(), key=lambda p: p.name):
            if not sub_dir.is_dir():
                continue
            meta_path = sub_dir / "meta.json"
            if not meta_path.exists():
                continue
            rows = _read_json(meta_path)
            if not isinstance(rows, list):
                raise ValueError(f"Expected list in {meta_path}, got {type(rows)}")
            for row in rows:
                frames = row["frames"]
                instruction = row["instruction"]
                steps = instruction["steps"]
                if len(steps) != 3:
                    raise ValueError(
                        f"Expected 3 step instructions in {meta_path} for id={row.get('id')}"
                    )
                sample_id = str(row["id"])
                samples.append(
                    NormalDatapoint(
                        sample_id=sample_id,
                        primary_class=primary_dir.name,
                        sub_class=sub_dir.name,
                        input_image=_resolve_existing(sub_dir / frames["input"]),
                        intermediate_1_image=_resolve_existing(
                            sub_dir / frames["intermediate_1"]
                        ),
                        intermediate_2_image=_resolve_existing(
                            sub_dir / frames["intermediate_2"]
                        ),
                        output_image=_resolve_existing(sub_dir / frames["output"]),
                        step_1_instruction=str(steps[0]),
                        step_2_instruction=str(steps[1]),
                        step_3_instruction=str(steps[2]),
                        global_instruction=str(instruction["global"]),
                    )
                )
    return samples


def discover_anti_physic_datapoints(bench_root: Path) -> List[AntiPhysicDatapoint]:
    bench_root = bench_root.resolve()
    meta_path = bench_root / "anti-physic" / "meta.jsonl"
    samples: List[AntiPhysicDatapoint] = []
    for row in _read_jsonl(meta_path):
        data_id = str(row["data_id"])
        input_image = bench_root / "anti-physic" / "input_data" / f"data_{data_id}.png"
        samples.append(
            AntiPhysicDatapoint(
                data_id=data_id,
                data_type=str(row.get("data_type", "")),
                input_image=_resolve_existing(input_image),
                edit_prompt=str(row["edit_prompt"]),
            )
        )
    return samples


def build_run1_prompt(step_1_instruction: str) -> str:
    return (
        "Task:\n"
        "Edit the provided image to produce the next state (intermediate_1).\n\n"
        "Instruction:\n"
        f"{step_1_instruction}\n\n"
        f"{CONSTRAINTS_BLOCK}\n"
        "Output: one edited image only."
    )


def build_run2_prompt(step_2_instruction: str) -> str:
    return (
        "Task:\n"
        "Edit the provided image to produce the next state (intermediate_2).\n\n"
        "Instruction:\n"
        f"{step_2_instruction}\n\n"
        f"{CONSTRAINTS_BLOCK}\n"
        "Output: one edited image only."
    )


def build_run3_prompt(step_3_instruction: str) -> str:
    return (
        "Task:\n"
        "Edit the provided image to produce the final state (output).\n\n"
        "Instruction:\n"
        f"{step_3_instruction}\n\n"
        f"{CONSTRAINTS_BLOCK}\n"
        "Output: one edited image only."
    )


def build_run4_prompt(
    step_1_instruction: str, step_2_instruction: str, step_3_instruction: str
) -> str:
    return (
        "Task:\n"
        "Starting from the provided input image, apply the following three step instructions sequentially to reach the final output state.\n"
        "Do NOT output intermediate images; output only the final edited image after completing all steps.\n\n"
        "Step 1:\n"
        f"{step_1_instruction}\n\n"
        "Step 2:\n"
        f"{step_2_instruction}\n\n"
        "Step 3:\n"
        f"{step_3_instruction}\n\n"
        f"{CONSTRAINTS_BLOCK}\n"
        "Output: one final edited image only."
    )


def build_run5_prompt(global_instruction: str) -> str:
    return (
        "Task:\n"
        "Edit the provided image to reach the final target state (output).\n\n"
        "Instruction:\n"
        f"{global_instruction}\n\n"
        f"{CONSTRAINTS_BLOCK}\n"
        "Output: one edited image only."
    )


def build_anti_physic_prompt(edit_prompt: str) -> str:
    return (
        "Task:\n"
        "Edit the provided image to create a physically impossible outcome.\n\n"
        "Impossible instruction:\n"
        f"{edit_prompt}\n\n"
        f"{CONSTRAINTS_BLOCK}\n"
        "Output: one edited image only."
    )


def build_normal_run_prompts(sample: NormalDatapoint) -> Dict[str, str]:
    return {
        "run1": build_run1_prompt(sample.step_1_instruction),
        "run2": build_run2_prompt(sample.step_2_instruction),
        "run3": build_run3_prompt(sample.step_3_instruction),
        "run4": build_run4_prompt(
            sample.step_1_instruction,
            sample.step_2_instruction,
            sample.step_3_instruction,
        ),
        "run5": build_run5_prompt(sample.global_instruction),
    }


def normal_output_paths(out_root: Path, model_name: str, sample: NormalDatapoint) -> Dict[str, Path]:
    base = out_root / model_name / sample.primary_class / sample.sub_class
    return {
        "run1": base / "step" / "run1" / f"{sample.sample_id}.png",
        "run2": base / "step" / "run2" / f"{sample.sample_id}.png",
        "run3": base / "step" / "run3" / f"{sample.sample_id}.png",
        "run4": base / "step" / "run4" / f"{sample.sample_id}.png",
        "run5": base / "global" / "run5" / f"{sample.sample_id}.png",
    }


def anti_physic_output_path(out_root: Path, model_name: str, sample: AntiPhysicDatapoint) -> Path:
    return out_root / model_name / "anti-physic" / f"{sample.data_id}.png"


def normalize_to_cache(
    image_path: Path, bench_root: Path, cache_root: Path, long_side: int = 1024
) -> Path:
    bench_root = bench_root.resolve()
    image_path = image_path.resolve()
    cache_root = cache_root.resolve()
    rel = image_path.relative_to(bench_root)
    cache_path = (cache_root / rel).with_suffix(".png")
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        normalize_image(
            in_path=image_path,
            out_path=cache_path,
            long_side=long_side,
            force_square=False,
        )
    return cache_path


def safe_load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def safe_save_image(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(path, format="PNG")


def limit_items(items: Sequence, limit: int | None) -> List:
    if limit is None:
        return list(items)
    return list(items[: max(0, limit)])


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _resolve_existing(path: Path) -> Path:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path
