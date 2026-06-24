from pathlib import Path
from typing import Tuple, Union, Optional
from PIL import Image, ImageCms, ImageOps
import io

NEUTRAL_GRAY = (128, 128, 128)

def _to_srgb(img: Image.Image) -> Image.Image:
    """
    Convert image to sRGB if it has an embedded ICC profile.
    Fallback: just convert to RGB.
    """
    icc = img.info.get("icc_profile", None)
    if not icc:
        return img.convert("RGB")

    try:
        src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
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
    in_path = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = Image.open(in_path)

    # Handle EXIF orientation (phone photos etc.)
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    # Convert to sRGB best-effort
    # Note: need io + ImageOps imported
    img = _to_srgb(img)

    # Resize to target long side
    w, h = img.size
    cur_long = max(w, h)
    if cur_long != long_side:
        scale = long_side / float(cur_long)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        # Downscale: LANCZOS, Upscale: BICUBIC
        if scale < 1.0:
            resample = Image.Resampling.LANCZOS
        else:
            resample = Image.Resampling.BICUBIC

        img = img.resize((new_w, new_h), resample=resample)

    # Optional: pad to square
    if force_square:
        w, h = img.size
        s = max(w, h)
        canvas = Image.new("RGB", (s, s), pad_color)
        x = (s - w) // 2
        y = (s - h) // 2
        canvas.paste(img, (x, y))
        img = canvas

    # Save as PNG (sRGB)
    # Ensure no alpha (or keep alpha if you want; here we standardize to RGB)
    if img.mode != "RGB":
        img = img.convert("RGB")

    img.save(out_path, format="PNG", optimize=True)
    return out_path