# utils.py
from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


# -----------------------------
# Evaluation constants
# -----------------------------
DIMENSIONS = ["consistency", "instruction_following", "physical_plausibility", "image_quality"]
TYPE_ORDER = ["TypeA", "TypeB", "TypeC", "TypeD", "TypeE"]

DIMENSION_WEIGHTS = {
    "consistency": 0.2,
    "instruction_following": 0.3,
    "physical_plausibility": 0.4,
    "image_quality": 0.1,
}

ANTI_DIMENSIONS = ["Consistency", "Instruction_Following", "Physical_Plausibility", "Image_Quality"]
ANTI_DIMENSION_WEIGHTS = {
    "Consistency": 0.2,
    "Instruction_Following": 0.3,
    "Physical_Plausibility": 0.4,
    "Image_Quality": 0.1,
}


# -----------------------------
# OpenAI client helpers
# -----------------------------
def make_openai_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout_sec: int = 180,
):
    """
    Create an official OpenAI Python SDK client.

    By default the SDK talks to the official OpenAI API. Set OPENAI_BASE_URL, or
    pass base_url, only when using an OpenAI-compatible endpoint for local tests.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install the `openai` package.") from exc

    resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY")
    resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL")

    kwargs: Dict[str, Any] = {"timeout": timeout_sec}
    if resolved_api_key:
        kwargs["api_key"] = resolved_api_key
    if resolved_base_url:
        kwargs["base_url"] = resolved_base_url.rstrip("/")
    return OpenAI(**kwargs)


def response_to_jsonable(resp: Any) -> Any:
    if isinstance(resp, (dict, list, str, int, float, bool)) or resp is None:
        return resp
    if hasattr(resp, "model_dump"):
        return resp.model_dump()
    if hasattr(resp, "to_dict"):
        return resp.to_dict()
    if hasattr(resp, "model_dump_json"):
        return json.loads(resp.model_dump_json())
    return str(resp)


# -----------------------------
# Prompts
# -----------------------------
CONSTRAINTS_BLOCK = """Constraints (must follow):
- Keep the camera viewpoint unchanged.
- Keep the background unchanged.
- Do not add text, logos, watermarks, UI elements, or borders.
- Preserve the overall realism and natural appearance.
"""


def format_bullets(items: List[str], title: str) -> str:
    if not items:
        return f"{title}:\n- (none)\n"
    lines = "\n".join([f"- {x}" for x in items])
    return f"{title}:\n{lines}\n"


def prompt_consistency(instruction: str, invariants: List[str]) -> str:
    """
    Input: input_img + pred_img + instruction + invariants
    Goal: check whether non-target content is preserved (ignore changes required by instruction).
    """
    return f"""You are a strict evaluator for image editing.

Rate the edited image on **Consistency** from 1 to 10.

Definition:
- Consistency measures whether everything that should remain unchanged (camera viewpoint, background, lighting style, and unrelated objects) stays consistent between the input image and the edited image.
- You MUST ignore changes that are explicitly required by the instruction.

Scoring rubric (1-10):
- 10: Only the intended edits occur; invariants and unrelated regions are preserved extremely well.
- 7-9: Minor unintended changes (small texture shifts, slight lighting drift), but overall consistent.
- 4-6: Noticeable unintended changes (background altered, viewpoint drift, extra objects), partially consistent.
- 1-3: Major unwanted changes; scene identity not preserved.

Instruction (for allowed changes):
{instruction}

{CONSTRAINTS_BLOCK}
{format_bullets(invariants, "Invariants to preserve")}

Return ONLY JSON matching the schema.
"""


def prompt_instruction_following(instruction: str) -> str:
    """
    Input: input_img + ref_img(GT) + pred_img + instruction
    Goal: does pred match the intended edit and align with reference GT target.
    """
    return f"""You are a strict evaluator for image editing.

Rate the edited image on **Instruction Following** from 1 to 10.

You are given:
- An input image (before edit)
- A reference target image (ground-truth goal state)
- A model edited image (prediction)

Judge how well the prediction satisfies the instruction AND matches the reference target state.

Scoring rubric (1-10):
- 10: Prediction matches the reference target very closely and fulfills the instruction precisely.
- 7-9: Mostly correct; small differences from reference target but clearly follows instruction.
- 4-6: Partially correct; key aspects missing or wrong; noticeable mismatch vs reference.
- 1-3: Fails to perform the intended edit; does not resemble the reference target.

Instruction:
{instruction}

{CONSTRAINTS_BLOCK}

Return ONLY JSON matching the schema.
"""


def prompt_physical_plausibility(instruction: str, explain: str) -> str:
    """
    Input: input_img + ref_img(GT) + pred_img + instruction + explain
    Goal: physical correctness / plausibility of the process and outcome.
    """
    return f"""You are a strict evaluator focusing on physical correctness in image editing.

Rate the edited image on **Physical Plausibility** from 1 to 10.

You are given:
- Input image (before)
- Reference target image (ground-truth intended physical outcome)
- Prediction image (model output)

Evaluate whether the prediction is physically plausible and consistent with the described physical mechanism.
You may use the reference target as the intended outcome, but your score should primarily reflect physics correctness:
- correct forces, motion direction, material behavior, fluid behavior, deformation, state change
- no impossible artifacts (e.g., bubbles sinking when they should rise, objects floating without cause, broken gravity, inconsistent contact)

Instruction:
{instruction}

Physical explanation of what should happen:
{explain}

{CONSTRAINTS_BLOCK}

Return ONLY JSON matching the schema.
"""


def prompt_image_quality() -> str:
    """
    Input: pred_img + ref_img(GT)
    Goal: visual realism/clarity/artefacts. ref_img helps judge expected style/quality.
    """
    return f"""You are a strict evaluator for image generation quality.

Rate the **Image Quality** of the prediction from 1 to 10.
You are given:
- Prediction image (model output)
- Reference target image (ground-truth), as a guide for expected realism and visual style.

Consider:
- sharpness, artifacts, distortions, unnatural textures
- coherence of lighting/shadows/reflections
- photorealism and consistency of materials
- absence of obvious AI artifacts or glitches

Scoring rubric (1-10):
- 10: Highly realistic, sharp where appropriate, no noticeable artifacts.
- 7-9: Minor artifacts or softness, overall high quality.
- 4-6: Clear artifacts, blur, distortions, but still recognizable.
- 1-3: Severe artifacts, unrealistic, degraded output.

Return ONLY JSON matching the schema.
"""


# -----------------------------
# JSON schema for structured output
# -----------------------------
def score_json_schema(name: str = "score_response") -> Dict[str, Any]:
    return {
        "type": "json_schema",
        "name": name,
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "score": {"type": "integer", "minimum": 1, "maximum": 10},
                "reason": {"type": "string"},
            },
            "required": ["score", "reason"],
        },
    }


# -----------------------------
# Image helpers
# -----------------------------
NEUTRAL_GRAY = (128, 128, 128)


def _to_srgb(img: Any) -> Any:
    from PIL import ImageCms

    icc = img.info.get("icc_profile", None)
    if not icc:
        return img.convert("RGB")

    try:
        src = ImageCms.ImageCmsProfile(BytesIO(icc))
        dst = ImageCms.createProfile("sRGB")
        return ImageCms.profileToProfile(img, src, dst, outputMode="RGB")
    except Exception:
        return img.convert("RGB")


def normalize_image(
    in_path: Union[str, Path],
    out_path: Union[str, Path],
    long_side: int = 1024,
    force_square: bool = False,
    pad_color: Tuple[int, int, int] = NEUTRAL_GRAY,
) -> Path:
    """
    Normalize image:
      1) sRGB
      2) PNG
      3) resize so that max(width, height) == long_side, keep aspect ratio
      4) if force_square: pad to square with neutral gray (no cropping)

    Returns output path.
    """
    from PIL import Image, ImageOps

    in_path = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(in_path)

    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    img = _to_srgb(img)

    w, h = img.size
    cur_long = max(w, h)
    if cur_long != long_side:
        scale = long_side / float(cur_long)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resample = Image.Resampling.LANCZOS if scale < 1.0 else Image.Resampling.BICUBIC
        img = img.resize((new_w, new_h), resample=resample)

    if force_square:
        w, h = img.size
        s = max(w, h)
        canvas = Image.new("RGB", (s, s), pad_color)
        x = (s - w) // 2
        y = (s - h) // 2
        canvas.paste(img, (x, y))
        img = canvas

    if img.mode != "RGB":
        img = img.convert("RGB")

    img.save(out_path, format="PNG", optimize=True)
    return out_path


def load_image_as_data_url(path: Path, max_side: int = 768, fmt: str = "PNG") -> str:
    """
    For VLM input only: shrink the image to reduce token/cost, keep aspect ratio.
    """
    from PIL import Image

    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = min(1.0, float(max_side) / float(max(w, h)))
    if scale < 1.0:
        img = img.resize((int(round(w * scale)), int(round(h * scale))), Image.Resampling.LANCZOS)

    buf = BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def encode_image_base64(image_path: Union[str, Path]) -> str:
    with Path(image_path).open("rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# -----------------------------
# Responses API client
# -----------------------------
def extract_output_text(resp: Any) -> str:
    if hasattr(resp, "output_text") and isinstance(resp.output_text, str):
        return resp.output_text.strip()

    resp_dict = response_to_jsonable(resp)
    if not isinstance(resp_dict, dict):
        return ""

    parts: List[str] = []
    for item in resp_dict.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                parts.append(content["text"])
    return "\n".join(parts).strip()


def safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Structured outputs should already be clean JSON.
    This is a robust fallback if the platform ever returns extra text.
    """
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"(\{.*\})", t, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(1))


@dataclass
class ResponsesClient:
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = "gpt-4o"
    timeout_sec: int = 180

    def __post_init__(self) -> None:
        self.client = make_openai_client(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout_sec=self.timeout_sec,
        )

    def create_response(
        self,
        prompt_text: str,
        images: List[Tuple[str, Path]],
        temperature: float = 0.0,
        max_output_tokens: int = 256,
        schema_name: str = "score_response",
        extra_instructions: Optional[str] = None,
        save_raw_path: Optional[Path] = None,
    ) -> Any:
        """
        Build one official Responses API request with text + images and json_schema output.
        """
        content_items: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt_text}]
        for _, path in images:
            content_items.append({"type": "input_image", "image_url": load_image_as_data_url(path)})

        payload: Dict[str, Any] = {
            "model": self.model,
            "input": [{"role": "user", "content": content_items}],
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "text": {"format": score_json_schema(schema_name)},
        }
        if extra_instructions:
            payload["instructions"] = extra_instructions

        resp = self.client.responses.create(**payload)

        if save_raw_path:
            save_raw_path.parent.mkdir(parents=True, exist_ok=True)
            save_raw_path.write_text(
                json.dumps(response_to_jsonable(resp), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return resp


# -----------------------------
# Dimension scoring wrapper
# -----------------------------
def build_dimension_prompt(
    dimension: str,
    instruction: str,
    explain: str,
    invariants: List[str],
) -> str:
    if dimension == "consistency":
        return prompt_consistency(instruction=instruction, invariants=invariants)
    if dimension == "instruction_following":
        return prompt_instruction_following(instruction=instruction)
    if dimension == "physical_plausibility":
        return prompt_physical_plausibility(instruction=instruction, explain=explain)
    if dimension == "image_quality":
        return prompt_image_quality()
    raise ValueError(f"Unknown dimension: {dimension}")


def score_one_dimension(
    client: ResponsesClient,
    dimension: str,
    input_img: Optional[Path],
    ref_img: Optional[Path],
    pred_img: Path,
    instruction: str,
    explain: str,
    invariants: List[str],
    save_raw_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Returns dict: {score:int, reason:str}
    """
    prompt = build_dimension_prompt(dimension, instruction, explain, invariants)

    images: List[Tuple[str, Path]] = []
    if dimension == "consistency":
        assert input_img is not None
        images = [("input", input_img), ("pred", pred_img)]
    elif dimension == "instruction_following":
        assert input_img is not None and ref_img is not None
        images = [("input", input_img), ("ref", ref_img), ("pred", pred_img)]
    elif dimension == "physical_plausibility":
        assert input_img is not None and ref_img is not None
        images = [("input", input_img), ("ref", ref_img), ("pred", pred_img)]
    elif dimension == "image_quality":
        assert ref_img is not None
        images = [("pred", pred_img), ("ref", ref_img)]
    else:
        raise ValueError(dimension)

    resp = client.create_response(
        prompt_text=prompt,
        images=images,
        temperature=0.0,
        max_output_tokens=256,
        schema_name=f"{dimension}_score",
        extra_instructions="You are a strict judge. Output must follow the JSON schema exactly.",
        save_raw_path=save_raw_path,
    )

    text = extract_output_text(resp)
    data = safe_json_loads(text)

    score = int(data["score"])
    score = max(1, min(10, score))
    reason = str(data["reason"]).strip()

    return {"score": score, "reason": reason}


# -----------------------------
# JSONL and summary helpers
# -----------------------------
def write_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_existing_keys(jsonl_path: Path) -> set:
    keys = set()
    for obj in read_jsonl(jsonl_path):
        key = obj.get("_key")
        if key:
            keys.add(key)
    return keys


def load_existing_by_field(jsonl_path: Path, field: str) -> Dict[Any, Dict[str, Any]]:
    data: Dict[Any, Dict[str, Any]] = {}
    for obj in read_jsonl(jsonl_path):
        if field in obj:
            data[obj[field]] = obj
    return data


def mean(xs: List[float]) -> Optional[float]:
    return (sum(xs) / len(xs)) if xs else None


def normalized_weights(weights: Dict[str, float], dimensions: Iterable[str]) -> Dict[str, float]:
    dimensions = list(dimensions)
    total = sum(weights.get(d, 0.0) for d in dimensions)
    if total <= 0:
        raise ValueError("dimension weights sum must be > 0")
    return {d: weights.get(d, 0.0) / total for d in dimensions}


def row_edit_type(row: Dict[str, Any]) -> str:
    edit_type = row.get("type")
    if edit_type:
        return str(edit_type)
    return "UNKNOWN"


def weighted_mean_over_dimensions(
    rows: List[Dict[str, Any]],
    dimensions: List[str] = DIMENSIONS,
    weights: Dict[str, float] = DIMENSION_WEIGHTS,
) -> Dict[str, Any]:
    """
    Compute per-dimension mean first, then weighted sum across dimensions.
    Renormalize weights over dimensions that are present (have >=1 sample).
    """
    w = normalized_weights(weights, dimensions)
    dim_means: Dict[str, Dict[str, Any]] = {}
    present: List[str] = []

    for dimension in dimensions:
        xs = [r["score"] for r in rows if r.get("dimension") == dimension and r.get("score") is not None]
        m = mean(xs)
        dim_means[dimension] = {"count": len(xs), "mean": m}
        if m is not None:
            present.append(dimension)

    if not present:
        return {"count": 0, "weighted_mean": None, "dim_means": dim_means}

    wsum = sum(w[d] for d in present)
    weighted = sum(dim_means[d]["mean"] * (w[d] / wsum) for d in present)
    return {"count": len(rows), "weighted_mean": weighted, "dim_means": dim_means}


def summarize_normal_scores(jsonl_path: Path) -> Dict[str, Any]:
    rows = [row for row in read_jsonl(jsonl_path) if "score" in row]

    by_dimension = {}
    for dimension in DIMENSIONS:
        xs = [r["score"] for r in rows if r.get("dimension") == dimension]
        by_dimension[dimension] = {"count": len(xs), "mean": mean(xs)}

    overall_pack = weighted_mean_over_dimensions(rows)
    overall = {"count": overall_pack["count"], "weighted_mean": overall_pack["weighted_mean"]}

    by_type = {}
    for edit_type in TYPE_ORDER:
        sub = [r for r in rows if row_edit_type(r) == edit_type]
        pack = weighted_mean_over_dimensions(sub)
        by_type[edit_type] = {"count": pack["count"], "weighted_mean": pack["weighted_mean"]}

    primary_map: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        primary = row.get("primary", "UNKNOWN")
        primary_map.setdefault(primary, []).append(row)

    by_primary = {}
    for primary, subrows in sorted(primary_map.items()):
        pack = weighted_mean_over_dimensions(subrows)
        by_primary[primary] = {"count": pack["count"], "weighted_mean": pack["weighted_mean"]}

    primary_sub_map: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        key = f"{row.get('primary','UNKNOWN')}/{row.get('sub','UNKNOWN')}"
        primary_sub_map.setdefault(key, []).append(row)

    by_primary_sub = {}
    for key, subrows in sorted(primary_sub_map.items()):
        pack = weighted_mean_over_dimensions(subrows)
        by_primary_sub[key] = {"count": pack["count"], "weighted_mean": pack["weighted_mean"]}

    return {
        "source": str(jsonl_path),
        "weights": normalized_weights(DIMENSION_WEIGHTS, DIMENSIONS),
        "overall": overall,
        "by_dimension": by_dimension,
        "by_type": by_type,
        "by_primary": by_primary,
        "by_primary_sub": by_primary_sub,
        "overall_dim_means": overall_pack["dim_means"],
    }


def write_summary(summary: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def weighted_average_from_wide_rows(
    rows: List[Dict[str, Any]],
    dimensions: List[str],
    weights: Dict[str, float],
) -> Dict[str, Any]:
    dim_means = {}
    for dimension in dimensions:
        xs = [row[dimension] for row in rows if row.get(dimension) is not None]
        dim_means[dimension] = {"count": len(xs), "mean": mean(xs)}

    present = [d for d in dimensions if dim_means[d]["mean"] is not None]
    if not present:
        return {"count": len(rows), "weighted_mean": None, "dim_means": dim_means}

    w = normalized_weights(weights, dimensions)
    wsum = sum(w[d] for d in present)
    weighted = sum(dim_means[d]["mean"] * (w[d] / wsum) for d in present)
    return {"count": len(rows), "weighted_mean": weighted, "dim_means": dim_means}


def summarize_anti_scores(jsonl_path: Path) -> Dict[str, Any]:
    rows = [
        row
        for row in read_jsonl(jsonl_path)
        if any(row.get(dimension) is not None for dimension in ANTI_DIMENSIONS)
    ]

    overall_pack = weighted_average_from_wide_rows(rows, ANTI_DIMENSIONS, ANTI_DIMENSION_WEIGHTS)
    by_dimension = overall_pack["dim_means"]

    type_map: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        data_type = row.get("data_type", "UNKNOWN")
        type_map.setdefault(data_type, []).append(row)

    by_data_type = {}
    for data_type, subrows in sorted(type_map.items()):
        pack = weighted_average_from_wide_rows(subrows, ANTI_DIMENSIONS, ANTI_DIMENSION_WEIGHTS)
        by_data_type[data_type] = {
            "count": pack["count"],
            "weighted_mean": pack["weighted_mean"],
            "by_dimension": pack["dim_means"],
        }

    return {
        "source": str(jsonl_path),
        "weights": normalized_weights(ANTI_DIMENSION_WEIGHTS, ANTI_DIMENSIONS),
        "overall": {"count": overall_pack["count"], "weighted_mean": overall_pack["weighted_mean"]},
        "by_dimension": by_dimension,
        "by_data_type": by_data_type,
    }


def parse_anti_score_content(score_content: str) -> Dict[str, Any]:
    """
    Parse the anti-physics judge response, preserving the original wide score format.
    """
    try:
        score_data = safe_json_loads(score_content)
        scores = score_data.get("scores", {})
        explanations = score_data.get("explanations", {})
        return {
            "Instruction_Following": _coerce_score(scores.get("Instruction_Following")),
            "Physical_Plausibility": _coerce_score(scores.get("Physical_Plausibility")),
            "Consistency": _coerce_score(scores.get("Consistency")),
            "Image_Quality": _coerce_score(scores.get("Image_Quality")),
            "summary": explanations.get("summary", ""),
            "issues": explanations.get("issues", []),
        }
    except Exception:
        parsed_data: Dict[str, Any] = {}
        for field in ["Instruction_Following", "Physical_Plausibility", "Consistency", "Image_Quality"]:
            match = re.search(rf'"{field}"\s*:\s*(\d+)', score_content)
            parsed_data[field] = _coerce_score(match.group(1)) if match else None

        summary_match = re.search(r'"summary"\s*:\s*"([^"]*)"', score_content)
        parsed_data["summary"] = summary_match.group(1) if summary_match else ""

        issues_matches = re.findall(
            r'\{\s*"issue_name"\s*:\s*"([^"]+)"\s*,\s*"score_explanation"\s*:\s*"([^"]*)"',
            score_content,
        )
        parsed_data["issues"] = [
            {"issue_name": issue_name, "score_explanation": explanation}
            for issue_name, explanation in issues_matches
        ]
        return parsed_data


def _coerce_score(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        score = int(value)
    except Exception:
        return None
    return max(0, min(10, score))
