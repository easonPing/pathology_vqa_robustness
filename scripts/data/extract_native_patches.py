#!/usr/bin/env python3
"""Materialize the frozen native-VLM patches from TCGA and BCNB slides.

The public SlideChat repository specifies the 30-random-patch baseline, but it
does not publish an inverse mapping from its delivered ``patch_name`` values
back to source pixels for these two datasets.  The coordinate interpretations
below preserve Harry's handoff implementation:

* TCGA ``<slide>_<level>_<level0-x>_<level0-y>``: read 512 px at the encoded
  OpenSlide level, then resize to 224 px.
* BCNB ``<slide>_<positive-region-index>_<local-x>_<local-y>``: reconstruct a
  256 px crop from the positive polygon bounding box, apply a white polygon
  mask/padding, then resize to 224 px.

The official BCNB BALNMP preprocessing repository confirms the BCNB naming,
polygon, bounding-box, right/bottom padding, and 256-pixel grid conventions.
It is not pixel-equivalent to Harry's reconstruction, however: BALNMP uses
OpenCV polygon rasterization, writes/reloads an intermediate quality-100 JPEG,
filters patches with blank ratio >= 0.3, and writes quality-100 JPEG patches.
Harry instead reconstructs directly, uses PIL rasterization, resizes to 224,
and writes quality 95.  TCGA's inverse mapping is likewise absent from the
SlideChat release.  Both paths must therefore be checked against delivered
golden checksums before newly reconstructed pixels are accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import openslide
import pandas as pd
import pyvips
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SELECTION_PATH = REPO_ROOT / "data/manifests/native_patch_selection.csv"
DEFAULT_TCGA_ROOT = REPO_ROOT / "data/raw/slides/TCGA"
DEFAULT_BCNB_ROOT = REPO_ROOT / "data/raw/slides/BCNB"
DEFAULT_PATCH_ROOT = REPO_ROOT / "data/patches/native_vlm"
DEFAULT_REPORT_PATH = (
    REPO_ROOT / "data/manifests/native_patch_extraction_report.csv"
)
DEFAULT_PROTOCOL_PATH = REPO_ROOT / "configs/native_patch_protocol.json"

EXPECTED_SLIDES_BY_DATASET = {"TCGA": 353, "BCNB": 54}
SUPPORTED_DATASETS = frozenset(EXPECTED_SLIDES_BY_DATASET)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract the frozen 30 raw patches per SlideBench WSI.",
        epilog=(
            "WARNING: Harry's BCNB reconstruction is not pixel-equivalent to "
            "the public BALNMP preprocessing (OpenCV + intermediate JPEG + "
            "blank filtering). Use --reference-checksums with golden hashes "
            "before accepting newly reconstructed pixels."
        ),
    )
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION_PATH)
    parser.add_argument("--tcga-root", type=Path, default=DEFAULT_TCGA_ROOT)
    parser.add_argument("--bcnb-root", type=Path, default=DEFAULT_BCNB_ROOT)
    parser.add_argument("--patch-root", type=Path, default=DEFAULT_PATCH_ROOT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=DEFAULT_PROTOCOL_PATH,
        help="Single source of truth for the frozen native-patch protocol.",
    )
    parser.add_argument(
        "--reference-checksums",
        type=Path,
        default=None,
        help=(
            "Optional Harry/golden validation CSV. When supplied, every "
            "dataset+slide+patch_name SHA256 must match before acceptance."
        ),
    )
    parser.add_argument(
        "--allow-unverified-reconstruction",
        action="store_true",
        help=(
            "Permit exploratory extraction without golden SHA256 values. "
            "Such output is not accepted for the production experiment."
        ),
    )
    parser.add_argument(
        "--patches-per-slide",
        type=int,
        default=None,
        help="Explicit override; must agree with the protocol config by default.",
    )
    parser.add_argument(
        "--allow-protocol-override",
        action="store_true",
        help=(
            "Acknowledge a CLI/config mismatch. The production study remains "
            "locked to 30 patches."
        ),
    )
    parser.add_argument("--expected-slides", type=int, default=407)
    parser.add_argument("--tcga-source-size", type=int, default=512)
    parser.add_argument("--bcnb-source-size", type=int, default=256)
    parser.add_argument("--output-size", type=int, default=224)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract even when a readable output of the expected size exists.",
    )
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"Path must be inside the repository: {path}") from error


def load_patches_per_slide(
    protocol_path: Path,
    *,
    cli_value: int | None,
    allow_override: bool,
) -> int:
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Missing native-patch protocol: {protocol_path}")
    with protocol_path.open("r", encoding="utf-8") as handle:
        protocol = json.load(handle)
    if not isinstance(protocol, dict):
        raise ValueError("Native-patch protocol must be a JSON object")
    configured = protocol.get("patches_per_slide")
    if not isinstance(configured, int) or isinstance(configured, bool):
        raise ValueError("Protocol patches_per_slide must be an integer")
    if cli_value is not None and cli_value != configured and not allow_override:
        raise ValueError(
            f"CLI patches_per_slide={cli_value} disagrees with protocol "
            f"value {configured}; pass --allow-protocol-override only for an "
            "intentional, audited override"
        )
    selected = cli_value if cli_value is not None else configured
    if selected != 30:
        raise ValueError(
            "This production experiment is locked to the SlideChat baseline "
            "of exactly 30 patches per slide"
        )
    return selected


def normalize_slide_id(value: object) -> str:
    value = str(value).strip()
    if value.endswith(".0"):
        value = value[:-2]
    if value.isdigit():
        return str(int(value))
    return value


def validate_patch_name(patch_name: str) -> None:
    candidate = Path(patch_name)
    if (
        not patch_name
        or candidate.is_absolute()
        or candidate.name != patch_name
        or candidate.suffix.lower() not in {".jpg", ".jpeg"}
    ):
        raise ValueError(f"Unsafe or unsupported patch_name: {patch_name!r}")


def parse_tcga_patch_name(patch_name: str) -> dict[str, object]:
    validate_patch_name(patch_name)
    parts = Path(patch_name).stem.rsplit("_", 3)
    if len(parts) != 4:
        raise ValueError(f"Unexpected TCGA patch name: {patch_name}")
    slide, level, x, y = parts
    try:
        parsed = {"slide": slide, "level": int(level), "x": int(x), "y": int(y)}
    except ValueError as error:
        raise ValueError(f"Non-integer TCGA coordinates: {patch_name}") from error
    if parsed["x"] < 0 or parsed["y"] < 0:
        raise ValueError(f"Negative TCGA level-0 coordinates: {patch_name}")
    return parsed


def parse_bcnb_patch_name(patch_name: str) -> dict[str, object]:
    validate_patch_name(patch_name)
    parts = Path(patch_name).stem.rsplit("_", 3)
    if len(parts) != 4:
        raise ValueError(f"Unexpected BCNB patch name: {patch_name}")
    slide, region_index, local_x, local_y = parts
    try:
        parsed = {
            "slide": normalize_slide_id(slide),
            "region_index": int(region_index),
            "local_x": int(local_x),
            "local_y": int(local_y),
        }
    except ValueError as error:
        raise ValueError(f"Non-integer BCNB coordinates: {patch_name}") from error
    return parsed


def numpy_slice_bounds(start: int, length: int, total_length: int) -> tuple[int, int]:
    """Preserve the NumPy negative-slice behavior used in Harry's handoff."""
    normalized_start, normalized_stop, step = slice(
        start, start + length
    ).indices(total_length)
    if step != 1:
        raise ValueError(f"Unexpected slice step: {step}")
    return normalized_start, normalized_stop


def vips_crop_to_rgb_array(vips_crop: pyvips.Image) -> np.ndarray:
    if vips_crop.format != "uchar":
        vips_crop = vips_crop.cast("uchar")
    try:
        if vips_crop.interpretation not in {"srgb", "rgb"}:
            vips_crop = vips_crop.colourspace("srgb")
    except pyvips.Error:
        # Some valid three-band images do not expose a convertible profile.
        pass

    array = np.frombuffer(vips_crop.write_to_memory(), dtype=np.uint8).reshape(
        vips_crop.height, vips_crop.width, vips_crop.bands
    )
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.shape[2] >= 3:
        array = array[:, :, :3]
    else:
        raise ValueError(f"Unsupported image band count: {array.shape[2]}")
    return array.copy()


def create_polygon_mask(
    polygon_points: np.ndarray, *, width: int, height: int
) -> np.ndarray:
    points = [
        (int(point[0]), int(point[1]))
        for point in np.asarray(polygon_points, dtype=np.int32)
    ]
    if len(points) < 3:
        raise ValueError("A BCNB positive polygon must contain at least 3 vertices")
    mask_image = Image.new("L", (width, height), color=0)
    ImageDraw.Draw(mask_image).polygon(points, fill=255)
    return np.asarray(mask_image, dtype=np.uint8)


def calculate_statistics(image: Image.Image) -> dict[str, float]:
    image_array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    mean_rgb = image_array.mean(axis=(0, 1))
    return {
        "mean_red": float(mean_rgb[0]),
        "mean_green": float(mean_rgb[1]),
        "mean_blue": float(mean_rgb[2]),
        "white_fraction": float((image_array.mean(axis=2) > 235).mean()),
        "exact_white_fraction": float(
            np.all(image_array == 255, axis=2).mean()
        ),
    }


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_is_valid(path: Path, *, output_size: int) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        with Image.open(path) as image:
            image.load()
            return image.size == (output_size, output_size) and image.mode in {
                "RGB",
                "L",
            }
    except (OSError, ValueError):
        return False


def save_jpeg_atomic(
    image: Image.Image, path: Path, *, quality: int, output_size: int
) -> None:
    if image.size != (output_size, output_size):
        raise ValueError(f"Refusing to save unexpected image size {image.size}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    image.convert("RGB").save(
        temporary, format="JPEG", quality=quality, subsampling=0
    )
    temporary.replace(path)


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def build_bcnb_file_map(root: Path, suffixes: set[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        slide = normalize_slide_id(path.stem)
        if slide in result:
            raise RuntimeError(
                f"Duplicate BCNB files for slide {slide}: {result[slide]} and {path}"
            )
        result[slide] = path.resolve()
    return result


def build_tcga_svs_map(root: Path, required_slides: set[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(root.rglob("*.svs")):
        slide = path.stem
        if slide not in required_slides:
            continue
        if slide in result:
            raise RuntimeError(
                f"Duplicate exact SVS matches for {slide}: {result[slide]} and {path}"
            )
        result[slide] = path.resolve()
    missing = required_slides - set(result)
    if missing:
        raise FileNotFoundError(f"Missing exact-match TCGA SVS files: {sorted(missing)}")
    return result


def validate_nonempty_sources(paths: dict[str, Path], *, label: str) -> None:
    empty = [slide for slide, path in paths.items() if path.stat().st_size <= 0]
    if empty:
        raise ValueError(f"Zero-byte or incomplete {label} files: {sorted(empty)}")


def load_and_validate_selection(
    path: Path, *, patches_per_slide: int, expected_slides: int
) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing patch selection: {path}")
    selection = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "slide" not in selection.columns and "slide_id" in selection.columns:
        selection = selection.rename(columns={"slide_id": "slide"})

    required = {"dataset", "slide", "patch_index", "patch_name"}
    missing = required - set(selection.columns)
    if missing:
        raise ValueError(f"Missing selection columns: {sorted(missing)}")

    selection = selection.copy()
    selection["dataset"] = selection["dataset"].str.strip().str.upper()
    selection["slide"] = selection["slide"].map(normalize_slide_id)
    selection["patch_name"] = selection["patch_name"].str.strip()
    unsupported = set(selection["dataset"]) - SUPPORTED_DATASETS
    if unsupported:
        raise ValueError(f"Unsupported dataset values: {sorted(unsupported)}")
    for patch_name in selection["patch_name"]:
        validate_patch_name(patch_name)

    numeric = pd.to_numeric(selection["patch_index"], errors="raise")
    if not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("patch_index values must be integers")
    selection["patch_index"] = numeric.astype(np.int64)

    duplicated = selection.duplicated(
        subset=["dataset", "slide", "patch_index"], keep=False
    )
    if duplicated.any():
        raise ValueError(
            "Duplicate patch indexes: "
            + selection.loc[
                duplicated, ["dataset", "slide", "patch_index"]
            ].head(20).to_string(index=False)
        )
    duplicated_names = selection.duplicated(
        subset=["dataset", "slide", "patch_name"], keep=False
    )
    if duplicated_names.any():
        raise ValueError("A slide contains duplicate selected patch names")

    groups = selection.groupby(["dataset", "slide"], sort=True)
    if groups.ngroups != expected_slides:
        raise ValueError(f"Expected {expected_slides} slides, found {groups.ngroups}")
    expected_indexes = list(range(patches_per_slide))
    bad_indexes = [
        f"{dataset}/{slide}"
        for (dataset, slide), group in groups
        if sorted(group["patch_index"].tolist()) != expected_indexes
    ]
    if bad_indexes:
        raise ValueError(
            "Every slide must have patch indexes 0..29 exactly; bad slides: "
            f"{bad_indexes[:20]}"
        )

    slides = selection[["dataset", "slide"]].drop_duplicates()
    slide_counts = slides["dataset"].value_counts().to_dict()
    if slide_counts != EXPECTED_SLIDES_BY_DATASET:
        raise ValueError(
            "Frozen slide counts changed; expected "
            f"{EXPECTED_SLIDES_BY_DATASET}, found {slide_counts}"
        )
    # Do not propagate Harry's machine-specific patch_path/absolute_path fields
    # into the canonical report.  Only portable selection provenance survives.
    canonical_columns = [
        column
        for column in (
            "dataset",
            "slide",
            "patch_index",
            "patch_name",
            "eligible_patch_count",
            "ordered_patch_pool_sha256",
            "master_seed",
            "slide_seed",
            "selection_algorithm",
            "pandas_version",
            "feature_path",
        )
        if column in selection.columns
    ]
    return selection[canonical_columns].sort_values(
        ["dataset", "slide", "patch_index"]
    ).reset_index(drop=True)


def load_reference_checksums(
    path: Path, selection: pd.DataFrame
) -> dict[tuple[str, str, str], str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing reference-checksum CSV: {path}")
    reference = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "slide" not in reference.columns and "slide_id" in reference.columns:
        reference = reference.rename(columns={"slide_id": "slide"})
    if "sha256" not in reference.columns and "patch_sha256" in reference.columns:
        reference = reference.rename(columns={"patch_sha256": "sha256"})
    required = {"dataset", "slide", "patch_name", "sha256"}
    missing = required - set(reference.columns)
    if missing:
        raise ValueError(f"Missing reference-checksum columns: {sorted(missing)}")

    reference["dataset"] = reference["dataset"].str.strip().str.upper()
    reference["slide"] = reference["slide"].map(normalize_slide_id)
    reference["patch_name"] = reference["patch_name"].str.strip()
    reference["sha256"] = reference["sha256"].str.strip().str.lower()
    invalid_hashes = reference.loc[
        ~reference["sha256"].str.fullmatch(r"[0-9a-f]{64}"), "sha256"
    ]
    if not invalid_hashes.empty:
        raise ValueError(
            f"Invalid reference SHA256 values: {invalid_hashes.head(10).tolist()}"
        )

    keys = ["dataset", "slide", "patch_name"]
    if reference.duplicated(subset=keys).any():
        raise ValueError("Duplicate keys in reference-checksum CSV")
    mapping = {
        (row.dataset, row.slide, row.patch_name): row.sha256
        for row in reference[[*keys, "sha256"]].itertuples(index=False)
    }
    expected_keys = {
        (row.dataset, row.slide, row.patch_name)
        for row in selection[keys].itertuples(index=False)
    }
    missing_keys = expected_keys - set(mapping)
    if missing_keys:
        raise ValueError(
            "Reference-checksum CSV is missing selected patches: "
            f"{sorted(missing_keys)[:20]}"
        )
    return mapping


def materialize_tcga_patch(
    *,
    slide_handle: Any,
    slide: str,
    patch_name: str,
    output_path: Path,
    source_size: int,
    output_size: int,
    jpeg_quality: int,
    overwrite: bool,
) -> tuple[Image.Image, str, dict[str, object]]:
    parsed = parse_tcga_patch_name(patch_name)
    if parsed["slide"] != slide:
        raise ValueError(f"Patch slide mismatch: {patch_name} vs {slide}")

    level = int(parsed["level"])
    x = int(parsed["x"])
    y = int(parsed["y"])
    if not 0 <= level < slide_handle.level_count:
        raise ValueError(f"Invalid OpenSlide level {level} for {patch_name}")
    level_width, level_height = slide_handle.level_dimensions[level]
    downsample = float(slide_handle.level_downsamples[level])
    level_x = int(x / downsample)
    level_y = int(y / downsample)
    if (
        level_x < 0
        or level_y < 0
        or level_x + source_size > level_width
        or level_y + source_size > level_height
    ):
        raise ValueError(f"TCGA crop outside slide bounds: {patch_name}")

    metadata = {
        "source_patch_width": source_size,
        "source_patch_height": source_size,
        "level": level,
        "global_x": x,
        "global_y": y,
        "extraction_method": "openslide_read_region_resize",
    }
    if output_is_valid(output_path, output_size=output_size) and not overwrite:
        with Image.open(output_path) as image:
            return image.convert("RGB"), "reused", metadata

    image = slide_handle.read_region(
        location=(x, y), level=level, size=(source_size, source_size)
    ).convert("RGB")
    image = image.resize((output_size, output_size), Image.Resampling.LANCZOS)
    save_jpeg_atomic(
        image, output_path, quality=jpeg_quality, output_size=output_size
    )
    return image, "extracted", metadata


def materialize_bcnb_patch(
    *,
    image_handle: pyvips.Image,
    positive_regions: list[object],
    slide: str,
    patch_name: str,
    output_path: Path,
    source_size: int,
    output_size: int,
    jpeg_quality: int,
    overwrite: bool,
) -> tuple[Image.Image, str, dict[str, object]]:
    parsed = parse_bcnb_patch_name(patch_name)
    if parsed["slide"] != slide:
        raise ValueError(f"Patch slide mismatch: {patch_name} vs {slide}")

    region_index = int(parsed["region_index"])
    local_x = int(parsed["local_x"])
    local_y = int(parsed["local_y"])
    if not 0 <= region_index < len(positive_regions):
        raise ValueError(f"Invalid BCNB region index for {patch_name}")
    region = positive_regions[region_index]
    if not isinstance(region, dict) or "vertices" not in region:
        raise ValueError(f"BCNB positive region lacks vertices: {patch_name}")
    vertices = np.asarray(region["vertices"], dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 2 or len(vertices) < 3:
        raise ValueError(f"Invalid polygon vertices for {patch_name}")

    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    bbox_x, bbox_y = int(minimum[0]), int(minimum[1])
    bbox_width = int(maximum[0] - minimum[0])
    bbox_height = int(maximum[1] - minimum[1])
    if bbox_width <= 0 or bbox_height <= 0:
        raise ValueError(f"Invalid BCNB bounding box for {patch_name}")

    region_x0, region_x1 = numpy_slice_bounds(
        bbox_x, bbox_width, image_handle.width
    )
    region_y0, region_y1 = numpy_slice_bounds(
        bbox_y, bbox_height, image_handle.height
    )
    actual_width = region_x1 - region_x0
    actual_height = region_y1 - region_y0
    if actual_width <= 0 or actual_height <= 0:
        raise ValueError(f"BCNB annotation does not overlap image: {patch_name}")

    padded_width = math.ceil(bbox_width / source_size) * source_size
    padded_height = math.ceil(bbox_height / source_size) * source_size
    if local_x % source_size or local_y % source_size:
        raise ValueError(f"Unaligned BCNB local coordinates: {patch_name}")
    if not 0 <= local_x < padded_width or not 0 <= local_y < padded_height:
        raise ValueError(f"BCNB local coordinates outside padded region: {patch_name}")

    valid_width = max(0, min(source_size, actual_width - local_x))
    valid_height = max(0, min(source_size, actual_height - local_y))
    if valid_width <= 0 or valid_height <= 0:
        raise ValueError(f"BCNB patch is entirely padding: {patch_name}")

    source_x = region_x0 + local_x
    source_y = region_y0 + local_y
    if (
        source_x < 0
        or source_y < 0
        or source_x + valid_width > image_handle.width
        or source_y + valid_height > image_handle.height
    ):
        raise ValueError(f"Normalized BCNB crop outside image: {patch_name}")

    metadata = {
        "source_patch_width": source_size,
        "source_patch_height": source_size,
        "region_index": region_index,
        "local_x": local_x,
        "local_y": local_y,
        "global_x": source_x,
        "global_y": source_y,
        "bbox_original_x": bbox_x,
        "bbox_original_y": bbox_y,
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "bbox_normalized_x": region_x0,
        "bbox_normalized_y": region_y0,
        "polygon_mask_applied": True,
        "numpy_negative_index_normalized": bbox_x < 0 or bbox_y < 0,
        "mask_implementation": "PIL.ImageDraw.polygon",
        "extraction_method": (
            "bcnb_bbox_polygon_mask_right_bottom_white_padding_resize"
        ),
    }
    if output_is_valid(output_path, output_size=output_size) and not overwrite:
        with Image.open(output_path) as image:
            return image.convert("RGB"), "reused", metadata

    crop = image_handle.crop(source_x, source_y, valid_width, valid_height)
    crop_array = vips_crop_to_rgb_array(crop)
    source_array = np.full((source_size, source_size, 3), 255, dtype=np.uint8)
    source_array[0:valid_height, 0:valid_width] = crop_array

    patch_polygon = (
        vertices
        - np.asarray([bbox_x, bbox_y], dtype=np.int32)
        - np.asarray([local_x, local_y], dtype=np.int32)
    )
    polygon_mask = create_polygon_mask(
        patch_polygon, width=source_size, height=source_size
    )
    source_array[polygon_mask != 255] = (255, 255, 255)
    image = Image.fromarray(source_array).resize(
        (output_size, output_size), Image.Resampling.LANCZOS
    )
    save_jpeg_atomic(
        image, output_path, quality=jpeg_quality, output_size=output_size
    )
    return image, "extracted", metadata


def base_record(
    row: pd.Series,
    *,
    image_path: Path,
    raw_image_path: Path | None,
    annotation_path: Path | None,
    output_size: int,
) -> dict[str, object]:
    record = row.to_dict()
    record.update(
        {
            "image_path": repo_relative(image_path),
            "raw_image_path": (
                repo_relative(raw_image_path) if raw_image_path is not None else ""
            ),
            "annotation_path": (
                repo_relative(annotation_path) if annotation_path is not None else ""
            ),
            "output_width": output_size,
            "output_height": output_size,
            "status": "",
            "extraction_status": "",
            "sha256": "",
            "reference_sha256": "",
            "checksum_matches_reference": "",
            "file_size_bytes": "",
            "error": "",
        }
    )
    return record


def main() -> None:
    args = parse_args()
    if args.expected_slides != 407:
        raise ValueError("This script is frozen to the 407-slide study")
    if min(
        args.tcga_source_size,
        args.bcnb_source_size,
        args.output_size,
        args.jpeg_quality,
    ) <= 0:
        raise ValueError("Patch sizes and JPEG quality must be positive")
    if args.jpeg_quality > 100:
        raise ValueError("JPEG quality cannot exceed 100")

    selection_path = resolve_repo_path(args.selection)
    tcga_root = resolve_repo_path(args.tcga_root)
    bcnb_root = resolve_repo_path(args.bcnb_root)
    patch_root = resolve_repo_path(args.patch_root)
    report_path = resolve_repo_path(args.report)
    protocol_path = resolve_repo_path(args.protocol_config)
    reference_path = (
        resolve_repo_path(args.reference_checksums)
        if args.reference_checksums is not None
        else None
    )
    # Outputs and recorded inputs are intentionally repository-local.
    for path in (selection_path, tcga_root, bcnb_root, patch_root, report_path):
        repo_relative(path)
    repo_relative(protocol_path)
    if reference_path is not None:
        repo_relative(reference_path)
    if not tcga_root.is_dir():
        raise FileNotFoundError(f"Missing TCGA raw-slide root: {tcga_root}")
    if not bcnb_root.is_dir():
        raise FileNotFoundError(f"Missing BCNB raw-slide root: {bcnb_root}")

    patches_per_slide = load_patches_per_slide(
        protocol_path,
        cli_value=args.patches_per_slide,
        allow_override=args.allow_protocol_override,
    )

    selection = load_and_validate_selection(
        selection_path,
        patches_per_slide=patches_per_slide,
        expected_slides=args.expected_slides,
    )
    reference_checksums = (
        load_reference_checksums(reference_path, selection)
        if reference_path is not None
        else {}
    )
    if reference_path is None and not args.allow_unverified_reconstruction:
        raise ValueError(
            "Production reconstruction requires --reference-checksums because "
            "the public repositories do not define a pixel-equivalent inverse "
            "mapping. Use --allow-unverified-reconstruction only for audited "
            "exploration."
        )
    if reference_path is None:
        print(
            "WARNING: unverified exploratory reconstruction. The script will "
            "record output hashes but this output is not accepted for the "
            "production experiment."
        )
    patch_root.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    required_tcga = set(
        selection.loc[selection["dataset"] == "TCGA", "slide"].unique()
    )
    required_bcnb = set(
        selection.loc[selection["dataset"] == "BCNB", "slide"].unique()
    )
    tcga_map = build_tcga_svs_map(tcga_root, required_tcga)
    bcnb_image_map = build_bcnb_file_map(bcnb_root, {".jpg", ".jpeg"})
    bcnb_json_map = build_bcnb_file_map(bcnb_root, {".json"})
    missing_images = required_bcnb - set(bcnb_image_map)
    missing_jsons = required_bcnb - set(bcnb_json_map)
    if missing_images:
        raise FileNotFoundError(f"Missing BCNB images: {sorted(missing_images)}")
    if missing_jsons:
        raise FileNotFoundError(f"Missing BCNB annotations: {sorted(missing_jsons)}")
    validate_nonempty_sources(tcga_map, label="TCGA SVS")
    validate_nonempty_sources(
        {slide: bcnb_image_map[slide] for slide in required_bcnb},
        label="BCNB image",
    )
    validate_nonempty_sources(
        {slide: bcnb_json_map[slide] for slide in required_bcnb},
        label="BCNB annotation",
    )

    records: list[dict[str, object]] = []
    grouped = list(selection.groupby(["dataset", "slide"], sort=True))
    for slide_number, ((dataset, slide), rows) in enumerate(grouped, start=1):
        print(f"[{slide_number}/{len(grouped)}] {dataset} — {slide}")
        slide_handle = None
        bcnb_handle = None
        positive_regions: list[object] | None = None
        raw_path: Path | None = None
        annotation_path: Path | None = None

        try:
            if dataset == "TCGA":
                raw_path = tcga_map[slide]
                slide_handle = openslide.OpenSlide(str(raw_path))
            elif dataset == "BCNB":
                raw_path = bcnb_image_map[slide]
                annotation_path = bcnb_json_map[slide]
                bcnb_handle = pyvips.Image.new_from_file(str(raw_path), access="random")
                with annotation_path.open("r", encoding="utf-8") as handle:
                    annotation = json.load(handle)
                if not isinstance(annotation, dict):
                    raise ValueError(f"BCNB annotation must be an object: {annotation_path}")
                positive_regions = annotation.get("positive", [])
                if not isinstance(positive_regions, list):
                    raise ValueError(
                        f"BCNB annotation positive field must be a list: {annotation_path}"
                    )
            else:  # Protected by selection validation.
                raise ValueError(f"Unsupported dataset: {dataset}")

            for _, row in rows.sort_values("patch_index").iterrows():
                patch_name = row["patch_name"]
                output_path = patch_root / dataset / slide / patch_name
                record = base_record(
                    row,
                    image_path=output_path,
                    raw_image_path=raw_path,
                    annotation_path=annotation_path,
                    output_size=args.output_size,
                )
                try:
                    if dataset == "TCGA":
                        image, extraction_status, metadata = materialize_tcga_patch(
                            slide_handle=slide_handle,
                            slide=slide,
                            patch_name=patch_name,
                            output_path=output_path,
                            source_size=args.tcga_source_size,
                            output_size=args.output_size,
                            jpeg_quality=args.jpeg_quality,
                            overwrite=args.overwrite,
                        )
                    else:
                        image, extraction_status, metadata = materialize_bcnb_patch(
                            image_handle=bcnb_handle,
                            positive_regions=positive_regions,
                            slide=slide,
                            patch_name=patch_name,
                            output_path=output_path,
                            source_size=args.bcnb_source_size,
                            output_size=args.output_size,
                            jpeg_quality=args.jpeg_quality,
                            overwrite=args.overwrite,
                        )
                    if not output_is_valid(output_path, output_size=args.output_size):
                        raise RuntimeError(f"Output validation failed: {output_path}")
                    output_sha256 = sha256_file(output_path)
                    reference_sha256 = reference_checksums.get(
                        (dataset, slide, patch_name), ""
                    )
                    record.update(
                        {
                            "sha256": output_sha256,
                            "reference_sha256": reference_sha256,
                            "checksum_matches_reference": (
                                output_sha256 == reference_sha256
                                if reference_sha256
                                else ""
                            ),
                            "file_size_bytes": output_path.stat().st_size,
                        }
                    )
                    if reference_sha256 and output_sha256 != reference_sha256:
                        raise RuntimeError(
                            "Output checksum differs from the supplied golden "
                            f"checksum for {dataset}/{slide}/{patch_name}: "
                            f"{output_sha256} != {reference_sha256}"
                        )
                    record.update(metadata)
                    record.update(calculate_statistics(image))
                    record.update(
                        {
                            "status": "found",
                            "extraction_status": extraction_status,
                        }
                    )
                except Exception as error:  # Record every patch failure for audit.
                    record["status"] = "error"
                    record["extraction_status"] = "error"
                    record["error"] = repr(error)
                records.append(record)

        except Exception as error:
            completed = {
                (record["dataset"], record["slide"], record["patch_index"])
                for record in records
            }
            for _, row in rows.sort_values("patch_index").iterrows():
                key = (dataset, slide, row["patch_index"])
                if key in completed:
                    continue
                output_path = patch_root / dataset / slide / row["patch_name"]
                record = base_record(
                    row,
                    image_path=output_path,
                    raw_image_path=raw_path,
                    annotation_path=annotation_path,
                    output_size=args.output_size,
                )
                record.update(
                    {
                        "status": "error",
                        "extraction_status": "error",
                        "error": f"Slide-level error: {error!r}",
                    }
                )
                records.append(record)
        finally:
            if slide_handle is not None:
                slide_handle.close()

        checkpoint = pd.DataFrame.from_records(records).sort_values(
            ["dataset", "slide", "patch_index"]
        )
        write_csv_atomic(checkpoint, report_path)
        successes = checkpoint["status"].eq("found").sum()
        errors = checkpoint["status"].eq("error").sum()
        print(f"  Successful: {successes} | Errors: {errors}")

    report = pd.DataFrame.from_records(records).sort_values(
        ["dataset", "slide", "patch_index"]
    )
    write_csv_atomic(report, report_path)
    expected_patches = args.expected_slides * patches_per_slide
    successful_count = int(report["status"].eq("found").sum())
    error_count = int(report["status"].eq("error").sum())
    print("\nFinal status counts:")
    print(report["status"].value_counts(dropna=False).to_string())
    if len(report) != expected_patches:
        raise RuntimeError(
            f"Expected {expected_patches} report rows, found {len(report)}"
        )
    if successful_count != expected_patches or error_count:
        print(
            report.loc[
                report["status"] == "error",
                ["dataset", "slide", "patch_name", "error"],
            ].head(50).to_string(index=False)
        )
        raise RuntimeError("Patch extraction did not complete successfully")
    print(f"Saved: {repo_relative(report_path)}")
    print(f"Protocol: {repo_relative(protocol_path)}")


if __name__ == "__main__":
    main()
