# utils.py
from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image

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

DIMENSIONS = ["consistency", "instruction_following", "physical_plausibility", "image_quality"]


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
# Image encoding helpers
# -----------------------------
def load_image_as_data_url(path: Path, max_side: int = 768, fmt: str = "PNG") -> str:
    """
    For VLM input only: shrink the image to reduce token/cost, keep aspect ratio.
    """
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


def extract_output_text(resp: Dict[str, Any]) -> str:
    """
    NewAPI Responses: resp["output"] is a list; find message->content->output_text.
    """
    parts: List[str] = []
    for item in resp.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for c in item.get("content", []) or []:
            if c.get("type") == "output_text" and isinstance(c.get("text"), str):
                parts.append(c["text"])
    return "\n".join(parts).strip()


def safe_json_loads(text: str) -> Dict[str, Any]:
    """
    Structured outputs should already be clean JSON.
    This is a robust fallback if the platform ever returns extra text.
    """
    t = text.strip()
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"(\{.*\})", t, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(1))


# -----------------------------
# Responses API client (yinli.one / NewAPI)
# -----------------------------
@dataclass
class ResponsesClient:
    base_url: str
    api_key: str
    model: str = "gpt-4o"
    timeout_sec: int = 180

    def __post_init__(self):
        self.session = requests.Session()
        # avoid system proxy surprises
        self.session.trust_env = False

    @property
    def url(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/responses"

    def create_response(
        self,
        prompt_text: str,
        images: List[Tuple[str, Path]],  # (label, path)
        temperature: float = 0.0,
        max_output_tokens: int = 256,
        schema_name: str = "score_response",
        extra_instructions: Optional[str] = None,
        save_raw_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Build a single Responses request with text + images, force json_schema output.
        """
        content_items: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt_text}]

        for _, p in images:
            content_items.append(
                {
                    "type": "input_image",
                    "image_url": load_image_as_data_url(p),
                }
            )

        payload: Dict[str, Any] = {
            "model": self.model,
            "input": [{"role": "user", "content": content_items}],
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "text": {"format": score_json_schema(schema_name)},
        }
        if extra_instructions:
            payload["instructions"] = extra_instructions

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        r = self.session.post(self.url, headers=headers, json=payload, timeout=(10, self.timeout_sec))
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:1200]}")
        resp = r.json()

        if save_raw_path:
            save_raw_path.parent.mkdir(parents=True, exist_ok=True)
            save_raw_path.write_text(json.dumps(resp, ensure_ascii=False, indent=2), encoding="utf-8")

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

    # Choose images per your spec
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

    # defensive cast
    score = int(data["score"])
    score = max(1, min(10, score))
    reason = str(data["reason"]).strip()

    return {"score": score, "reason": reason}