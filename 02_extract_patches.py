# 02_extract_patches.py

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import pyvips
import openslide
from PIL import Image, ImageDraw


PROJECT_DIR = Path.cwd()
HDD_ROOT = Path("/Volumes/TOSHIBA EXT/exp2")

HANDOFF_ROOT = (
    HDD_ROOT
    / "native_vlm_handoff"
)

PATCH_ROOT = (
    HANDOFF_ROOT
    / "patches"
)

TCGA_RAW_ROOT = (
    HDD_ROOT
    / "raw_slides"
    / "TCGA"
)

BCNB_RAW_ROOT = (
    HDD_ROOT
    / "raw_slides"
    / "BCNB_required"
)

METADATA_DIR = (
    PROJECT_DIR
    / "patch_metadata"
)

SELECTION_PATH = (
    METADATA_DIR
    / "native_patch_selection_407slides_final.csv"
)

EXTRACTION_REPORT_PATH = (
    METADATA_DIR
    / "native_patch_extraction_report.csv"
)

TCGA_SOURCE_SIZE = 512
BCNB_SOURCE_SIZE = 256
OUTPUT_SIZE = 224

EXPECTED_PATCHES = 12210
EXPECTED_SLIDES = 407

OVERWRITE_EXISTING = False


def normalize_slide_id(value: object) -> str:
    value = str(value).strip()

    if value.endswith(".0"):
        value = value[:-2]

    if value.isdigit():
        return str(int(value))

    return value


def parse_tcga_patch_name(
    patch_name: str,
) -> dict:
    stem = Path(
        str(patch_name)
    ).stem

    parts = stem.rsplit(
        "_",
        3,
    )

    if len(parts) != 4:
        raise ValueError(
            f"Unexpected TCGA patch name: "
            f"{patch_name}"
        )

    slide_id, level, x, y = parts

    return {
        "slide_id": slide_id,
        "level": int(level),
        "x": int(x),
        "y": int(y),
    }


def parse_bcnb_patch_name(
    patch_name: str,
) -> dict:
    stem = Path(
        str(patch_name)
    ).stem

    parts = stem.rsplit(
        "_",
        3,
    )

    if len(parts) != 4:
        raise ValueError(
            f"Unexpected BCNB patch name: "
            f"{patch_name}"
        )

    (
        slide_id,
        region_index,
        local_x,
        local_y,
    ) = parts

    return {
        "slide_id": normalize_slide_id(
            slide_id
        ),
        "region_index": int(
            region_index
        ),
        "local_x": int(local_x),
        "local_y": int(local_y),
    }


def numpy_slice_bounds(
    start: int,
    length: int,
    total_length: int,
) -> tuple[int, int]:
    (
        normalized_start,
        normalized_stop,
        step,
    ) = slice(
        start,
        start + length,
    ).indices(total_length)

    if step != 1:
        raise ValueError(
            f"Unexpected slice step: {step}"
        )

    return (
        normalized_start,
        normalized_stop,
    )


def vips_crop_to_rgb_array(
    vips_crop: pyvips.Image,
) -> np.ndarray:
    if vips_crop.format != "uchar":
        vips_crop = vips_crop.cast(
            "uchar"
        )

    try:
        if vips_crop.interpretation not in {
            "srgb",
            "rgb",
        }:
            vips_crop = (
                vips_crop
                .colourspace("srgb")
            )
    except Exception:
        pass

    memory = (
        vips_crop
        .write_to_memory()
    )

    array = np.frombuffer(
        memory,
        dtype=np.uint8,
    ).reshape(
        vips_crop.height,
        vips_crop.width,
        vips_crop.bands,
    )

    if array.shape[2] == 1:
        array = np.repeat(
            array,
            3,
            axis=2,
        )

    elif array.shape[2] >= 3:
        array = array[:, :, :3]

    else:
        raise ValueError(
            "Unsupported image band count: "
            f"{array.shape[2]}"
        )

    return array.copy()


def create_polygon_mask(
    polygon_points: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    points = [
        (
            int(point[0]),
            int(point[1]),
        )
        for point in np.asarray(
            polygon_points,
            dtype=np.int32,
        )
    ]

    mask_image = Image.new(
        "L",
        (
            width,
            height,
        ),
        color=0,
    )

    if len(points) >= 3:
        ImageDraw.Draw(
            mask_image
        ).polygon(
            points,
            fill=255,
        )

    return np.asarray(
        mask_image,
        dtype=np.uint8,
    )


def calculate_statistics(
    image: Image.Image,
) -> dict:
    image_array = np.asarray(
        image.convert("RGB"),
        dtype=np.uint8,
    )

    mean_rgb = image_array.mean(
        axis=(0, 1)
    )

    return {
        "mean_red": float(
            mean_rgb[0]
        ),
        "mean_green": float(
            mean_rgb[1]
        ),
        "mean_blue": float(
            mean_rgb[2]
        ),
        "white_fraction": float(
            (
                image_array.mean(axis=2)
                > 235
            ).mean()
        ),
        "exact_white_fraction": float(
            np.all(
                image_array == 255,
                axis=2,
            ).mean()
        ),
    }


def output_is_valid(
    path: Path,
) -> bool:
    if not path.exists():
        return False

    try:
        with Image.open(path) as image:
            image.load()

            return (
                image.size
                == (
                    OUTPUT_SIZE,
                    OUTPUT_SIZE,
                )
            )

    except Exception:
        return False


def build_bcnb_file_map(
    root: Path,
    suffixes: set[str],
) -> dict[str, Path]:
    file_map = {}

    for path in root.rglob("*"):
        if (
            not path.is_file()
            or path.suffix.lower()
            not in suffixes
        ):
            continue

        slide_id = normalize_slide_id(
            path.stem
        )

        if slide_id in file_map:
            raise RuntimeError(
                f"Duplicate BCNB files for "
                f"slide {slide_id}:\n"
                f"{file_map[slide_id]}\n"
                f"{path}"
            )

        file_map[slide_id] = path

    return file_map


def build_tcga_svs_map(
    root: Path,
    slide_ids: list[str],
) -> dict[str, Path]:
    all_svs = [
        path
        for path in root.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            == ".svs"
        )
    ]

    svs_map = {}

    for slide_id in slide_ids:
        exact_matches = [
            path
            for path in all_svs
            if path.stem == slide_id
        ]

        matches = (
            exact_matches
            or [
                path
                for path in all_svs
                if slide_id in path.stem
            ]
        )

        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one SVS for "
                f"{slide_id}; found {len(matches)}: "
                f"{matches}"
            )

        svs_map[slide_id] = matches[0]

    return svs_map


def main() -> None:
    assert SELECTION_PATH.exists(), (
        SELECTION_PATH
    )

    assert TCGA_RAW_ROOT.exists(), (
        TCGA_RAW_ROOT
    )

    assert BCNB_RAW_ROOT.exists(), (
        BCNB_RAW_ROOT
    )

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PATCH_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    selection = pd.read_csv(
        SELECTION_PATH,
        dtype=str,
        keep_default_na=False,
    )

    required_columns = {
        "dataset",
        "slide_id",
        "patch_index",
        "patch_name",
    }

    missing_columns = (
        required_columns
        - set(selection.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing selection columns: "
            f"{sorted(missing_columns)}"
        )

    selection["dataset"] = (
        selection["dataset"]
        .str.strip()
        .str.upper()
    )

    selection["slide_id"] = (
        selection["slide_id"]
        .map(normalize_slide_id)
    )

    selection["patch_index_numeric"] = (
        pd.to_numeric(
            selection["patch_index"],
            errors="raise",
        )
    )

    selection = (
        selection
        .sort_values(
            [
                "dataset",
                "slide_id",
                "patch_index_numeric",
            ]
        )
        .reset_index(drop=True)
    )

    per_slide_counts = (
        selection
        .groupby(
            [
                "dataset",
                "slide_id",
            ]
        )
        .size()
    )

    if len(selection) != EXPECTED_PATCHES:
        raise ValueError(
            f"Expected {EXPECTED_PATCHES} "
            f"selected patches, "
            f"found {len(selection)}"
        )

    if len(per_slide_counts) != EXPECTED_SLIDES:
        raise ValueError(
            f"Expected {EXPECTED_SLIDES} slides, "
            f"found {len(per_slide_counts)}"
        )

    if not (
        per_slide_counts == 30
    ).all():
        raise ValueError(
            "At least one slide does not "
            "have exactly 30 patches"
        )

    bcnb_image_map = build_bcnb_file_map(
        BCNB_RAW_ROOT,
        {
            ".jpg",
            ".jpeg",
        },
    )

    bcnb_json_map = build_bcnb_file_map(
        BCNB_RAW_ROOT,
        {
            ".json",
        },
    )

    required_bcnb_ids = set(
        selection.loc[
            selection["dataset"]
            == "BCNB",
            "slide_id",
        ]
    )

    missing_bcnb_images = (
        required_bcnb_ids
        - set(bcnb_image_map)
    )

    missing_bcnb_jsons = (
        required_bcnb_ids
        - set(bcnb_json_map)
    )

    if missing_bcnb_images:
        raise FileNotFoundError(
            "Missing BCNB images: "
            f"{sorted(missing_bcnb_images)}"
        )

    if missing_bcnb_jsons:
        raise FileNotFoundError(
            "Missing BCNB JSON files: "
            f"{sorted(missing_bcnb_jsons)}"
        )

    required_tcga_ids = sorted(
        selection.loc[
            selection["dataset"]
            == "TCGA",
            "slide_id",
        ].unique()
    )

    tcga_svs_map = build_tcga_svs_map(
        TCGA_RAW_ROOT,
        required_tcga_ids,
    )

    records = []

    grouped_slides = list(
        selection.groupby(
            [
                "dataset",
                "slide_id",
            ],
            sort=True,
        )
    )

    for slide_number, (
        (
            dataset,
            slide_id,
        ),
        slide_rows,
    ) in enumerate(
        grouped_slides,
        start=1,
    ):
        print(
            f"[{slide_number}/"
            f"{len(grouped_slides)}] "
            f"{dataset} — {slide_id}"
        )

        slide_rows = (
            slide_rows
            .sort_values(
                "patch_index_numeric"
            )
        )

        tcga_slide = None
        bcnb_image = None
        positive_regions = None
        raw_image_path = None
        json_path = None

        try:
            if dataset == "TCGA":
                raw_image_path = (
                    tcga_svs_map[
                        slide_id
                    ]
                )

                tcga_slide = (
                    openslide.OpenSlide(
                        str(
                            raw_image_path
                        )
                    )
                )

            elif dataset == "BCNB":
                raw_image_path = (
                    bcnb_image_map[
                        slide_id
                    ]
                )

                json_path = (
                    bcnb_json_map[
                        slide_id
                    ]
                )

                bcnb_image = (
                    pyvips.Image
                    .new_from_file(
                        str(
                            raw_image_path
                        ),
                        access="random",
                    )
                )

                with open(
                    json_path,
                    "r",
                    encoding="utf-8",
                ) as file:
                    annotation_data = (
                        json.load(file)
                    )

                positive_regions = (
                    annotation_data
                    .get(
                        "positive",
                        [],
                    )
                )

            else:
                raise ValueError(
                    f"Unsupported dataset: "
                    f"{dataset}"
                )

            for _, selection_row in (
                slide_rows.iterrows()
            ):
                patch_name = (
                    selection_row[
                        "patch_name"
                    ]
                )

                relative_path = (
                    Path("patches")
                    / dataset
                    / slide_id
                    / patch_name
                ).as_posix()

                output_path = (
                    HANDOFF_ROOT
                    / relative_path
                )

                output_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                record = (
                    selection_row
                    .drop(
                        labels=[
                            "patch_index_numeric"
                        ],
                        errors="ignore",
                    )
                    .to_dict()
                )

                record.update(
                    {
                        "patch_path": (
                            relative_path
                        ),
                        "absolute_patch_path": str(
                            output_path
                        ),
                        "raw_image_path": str(
                            raw_image_path
                        ),
                        "annotation_json_path": (
                            str(json_path)
                            if json_path
                            else ""
                        ),
                        "source_patch_width": "",
                        "source_patch_height": "",
                        "output_width": OUTPUT_SIZE,
                        "output_height": OUTPUT_SIZE,
                        "level": "",
                        "region_index": "",
                        "local_x": "",
                        "local_y": "",
                        "global_x": "",
                        "global_y": "",
                        "bbox_original_x": "",
                        "bbox_original_y": "",
                        "bbox_width": "",
                        "bbox_height": "",
                        "bbox_normalized_x": "",
                        "bbox_normalized_y": "",
                        "source_x_normalized": "",
                        "source_y_normalized": "",
                        "polygon_mask_applied": False,
                        "numpy_negative_index_normalized": False,
                        "mask_implementation": "",
                        "extraction_method": "",
                        "status": "",
                        "error": "",
                    }
                )

                try:
                    if dataset == "TCGA":
                        parsed = (
                            parse_tcga_patch_name(
                                patch_name
                            )
                        )

                        if (
                            parsed["slide_id"]
                            != slide_id
                        ):
                            raise ValueError(
                                "Patch slide mismatch: "
                                f"{patch_name} vs "
                                f"{slide_id}"
                            )

                        if not (
                            0
                            <= parsed["level"]
                            < tcga_slide.level_count
                        ):
                            raise ValueError(
                                "Invalid OpenSlide "
                                f"level "
                                f"{parsed['level']} "
                                f"for {patch_name}"
                            )

                        (
                            level_width,
                            level_height,
                        ) = (
                            tcga_slide
                            .level_dimensions[
                                parsed["level"]
                            ]
                        )

                        downsample = float(
                            tcga_slide
                            .level_downsamples[
                                parsed["level"]
                            ]
                        )

                        level_x = int(
                            parsed["x"]
                            / downsample
                        )

                        level_y = int(
                            parsed["y"]
                            / downsample
                        )

                        if (
                            level_x < 0
                            or level_y < 0
                            or (
                                level_x
                                + TCGA_SOURCE_SIZE
                                > level_width
                            )
                            or (
                                level_y
                                + TCGA_SOURCE_SIZE
                                > level_height
                            )
                        ):
                            raise ValueError(
                                "TCGA crop outside "
                                "slide bounds: "
                                f"{patch_name}"
                            )

                        record.update(
                            {
                                "source_patch_width": (
                                    TCGA_SOURCE_SIZE
                                ),
                                "source_patch_height": (
                                    TCGA_SOURCE_SIZE
                                ),
                                "level": (
                                    parsed["level"]
                                ),
                                "global_x": (
                                    parsed["x"]
                                ),
                                "global_y": (
                                    parsed["y"]
                                ),
                                "extraction_method": (
                                    "openslide_read_"
                                    "region_resize"
                                ),
                            }
                        )

                        if (
                            output_is_valid(
                                output_path
                            )
                            and not (
                                OVERWRITE_EXISTING
                            )
                        ):
                            with Image.open(
                                output_path
                            ) as image:
                                model_patch = (
                                    image
                                    .convert("RGB")
                                )

                            record["status"] = (
                                "REUSED"
                            )

                        else:
                            model_patch = (
                                tcga_slide
                                .read_region(
                                    location=(
                                        parsed["x"],
                                        parsed["y"],
                                    ),
                                    level=(
                                        parsed[
                                            "level"
                                        ]
                                    ),
                                    size=(
                                        TCGA_SOURCE_SIZE,
                                        TCGA_SOURCE_SIZE,
                                    ),
                                )
                                .convert("RGB")
                            )

                            model_patch = (
                                model_patch
                                .resize(
                                    (
                                        OUTPUT_SIZE,
                                        OUTPUT_SIZE,
                                    ),
                                    Image.Resampling
                                    .LANCZOS,
                                )
                            )

                            model_patch.save(
                                output_path,
                                format="JPEG",
                                quality=95,
                                subsampling=0,
                            )

                            record["status"] = (
                                "EXTRACTED"
                            )

                    else:
                        parsed = (
                            parse_bcnb_patch_name(
                                patch_name
                            )
                        )

                        if (
                            parsed["slide_id"]
                            != slide_id
                        ):
                            raise ValueError(
                                "Patch slide mismatch: "
                                f"{patch_name} vs "
                                f"{slide_id}"
                            )

                        region_index = (
                            parsed[
                                "region_index"
                            ]
                        )

                        local_x = (
                            parsed["local_x"]
                        )

                        local_y = (
                            parsed["local_y"]
                        )

                        if not (
                            0
                            <= region_index
                            < len(
                                positive_regions
                            )
                        ):
                            raise ValueError(
                                "Invalid BCNB region "
                                f"index "
                                f"{region_index} "
                                f"for {patch_name}"
                            )

                        vertices = np.asarray(
                            positive_regions[
                                region_index
                            ]["vertices"],
                            dtype=np.int32,
                        )

                        if (
                            vertices.ndim != 2
                            or (
                                vertices.shape[1]
                                != 2
                            )
                        ):
                            raise ValueError(
                                "Invalid polygon "
                                "vertices for "
                                f"{patch_name}"
                            )

                        minimum = (
                            vertices.min(
                                axis=0
                            )
                        )

                        maximum = (
                            vertices.max(
                                axis=0
                            )
                        )

                        bbox_x = int(
                            minimum[0]
                        )

                        bbox_y = int(
                            minimum[1]
                        )

                        bbox_width = int(
                            maximum[0]
                            - minimum[0]
                        )

                        bbox_height = int(
                            maximum[1]
                            - minimum[1]
                        )

                        if (
                            bbox_width <= 0
                            or bbox_height <= 0
                        ):
                            raise ValueError(
                                "Invalid BCNB "
                                "bounding box for "
                                f"{patch_name}: "
                                f"{bbox_width}x"
                                f"{bbox_height}"
                            )

                        (
                            region_x0,
                            region_x1,
                        ) = numpy_slice_bounds(
                            bbox_x,
                            bbox_width,
                            bcnb_image.width,
                        )

                        (
                            region_y0,
                            region_y1,
                        ) = numpy_slice_bounds(
                            bbox_y,
                            bbox_height,
                            bcnb_image.height,
                        )

                        actual_region_width = (
                            region_x1
                            - region_x0
                        )

                        actual_region_height = (
                            region_y1
                            - region_y0
                        )

                        if (
                            actual_region_width
                            <= 0
                            or actual_region_height
                            <= 0
                        ):
                            raise ValueError(
                                "BCNB annotation "
                                "does not overlap "
                                "image: "
                                f"{patch_name}"
                            )

                        padded_width = (
                            math.ceil(
                                bbox_width
                                / BCNB_SOURCE_SIZE
                            )
                            * BCNB_SOURCE_SIZE
                        )

                        padded_height = (
                            math.ceil(
                                bbox_height
                                / BCNB_SOURCE_SIZE
                            )
                            * BCNB_SOURCE_SIZE
                        )

                        if (
                            local_x
                            % BCNB_SOURCE_SIZE
                            != 0
                        ):
                            raise ValueError(
                                "Unexpected local x "
                                "coordinate for "
                                f"{patch_name}: "
                                f"{local_x}"
                            )

                        if (
                            local_y
                            % BCNB_SOURCE_SIZE
                            != 0
                        ):
                            raise ValueError(
                                "Unexpected local y "
                                "coordinate for "
                                f"{patch_name}: "
                                f"{local_y}"
                            )

                        if not (
                            0
                            <= local_x
                            < padded_width
                        ):
                            raise ValueError(
                                "Local x outside "
                                "padded region: "
                                f"{patch_name}"
                            )

                        if not (
                            0
                            <= local_y
                            < padded_height
                        ):
                            raise ValueError(
                                "Local y outside "
                                "padded region: "
                                f"{patch_name}"
                            )

                        valid_width = max(
                            0,
                            min(
                                BCNB_SOURCE_SIZE,
                                (
                                    actual_region_width
                                    - local_x
                                ),
                            ),
                        )

                        valid_height = max(
                            0,
                            min(
                                BCNB_SOURCE_SIZE,
                                (
                                    actual_region_height
                                    - local_y
                                ),
                            ),
                        )

                        if (
                            valid_width <= 0
                            or valid_height <= 0
                        ):
                            raise ValueError(
                                "BCNB patch is "
                                "entirely padding: "
                                f"{patch_name}"
                            )

                        source_x = (
                            region_x0
                            + local_x
                        )

                        source_y = (
                            region_y0
                            + local_y
                        )

                        if (
                            source_x < 0
                            or source_y < 0
                            or (
                                source_x
                                + valid_width
                                > bcnb_image.width
                            )
                            or (
                                source_y
                                + valid_height
                                > bcnb_image.height
                            )
                        ):
                            raise ValueError(
                                "Normalized BCNB "
                                "crop outside image: "
                                f"{patch_name}"
                            )

                        record.update(
                            {
                                "source_patch_width": (
                                    BCNB_SOURCE_SIZE
                                ),
                                "source_patch_height": (
                                    BCNB_SOURCE_SIZE
                                ),
                                "region_index": (
                                    region_index
                                ),
                                "local_x": local_x,
                                "local_y": local_y,
                                "global_x": source_x,
                                "global_y": source_y,
                                "bbox_original_x": (
                                    bbox_x
                                ),
                                "bbox_original_y": (
                                    bbox_y
                                ),
                                "bbox_width": (
                                    bbox_width
                                ),
                                "bbox_height": (
                                    bbox_height
                                ),
                                "bbox_normalized_x": (
                                    region_x0
                                ),
                                "bbox_normalized_y": (
                                    region_y0
                                ),
                                "source_x_normalized": (
                                    source_x
                                ),
                                "source_y_normalized": (
                                    source_y
                                ),
                                "polygon_mask_applied": True,
                                "numpy_negative_index_normalized": (
                                    bbox_x < 0
                                    or bbox_y < 0
                                ),
                                "mask_implementation": (
                                    "PIL.ImageDraw."
                                    "polygon"
                                ),
                                "extraction_method": (
                                    "bcnb_bbox_"
                                    "polygon_mask_"
                                    "right_bottom_"
                                    "white_padding_"
                                    "resize"
                                ),
                            }
                        )

                        if (
                            output_is_valid(
                                output_path
                            )
                            and not (
                                OVERWRITE_EXISTING
                            )
                        ):
                            with Image.open(
                                output_path
                            ) as image:
                                model_patch = (
                                    image
                                    .convert("RGB")
                                )

                            record["status"] = (
                                "REUSED"
                            )

                        else:
                            valid_crop_vips = (
                                bcnb_image.crop(
                                    source_x,
                                    source_y,
                                    valid_width,
                                    valid_height,
                                )
                            )

                            valid_crop_array = (
                                vips_crop_to_rgb_array(
                                    valid_crop_vips
                                )
                            )

                            source_patch_array = (
                                np.full(
                                    (
                                        BCNB_SOURCE_SIZE,
                                        BCNB_SOURCE_SIZE,
                                        3,
                                    ),
                                    255,
                                    dtype=np.uint8,
                                )
                            )

                            source_patch_array[
                                0:valid_height,
                                0:valid_width,
                            ] = valid_crop_array

                            patch_polygon = (
                                vertices
                                - np.asarray(
                                    [
                                        bbox_x,
                                        bbox_y,
                                    ],
                                    dtype=np.int32,
                                )
                                - np.asarray(
                                    [
                                        local_x,
                                        local_y,
                                    ],
                                    dtype=np.int32,
                                )
                            )

                            polygon_mask = (
                                create_polygon_mask(
                                    patch_polygon,
                                    width=(
                                        BCNB_SOURCE_SIZE
                                    ),
                                    height=(
                                        BCNB_SOURCE_SIZE
                                    ),
                                )
                            )

                            source_patch_array[
                                polygon_mask
                                != 255
                            ] = (
                                255,
                                255,
                                255,
                            )

                            source_patch = (
                                Image.fromarray(
                                    source_patch_array
                                )
                            )

                            model_patch = (
                                source_patch.resize(
                                    (
                                        OUTPUT_SIZE,
                                        OUTPUT_SIZE,
                                    ),
                                    Image.Resampling
                                    .LANCZOS,
                                )
                            )

                            model_patch.save(
                                output_path,
                                format="JPEG",
                                quality=95,
                                subsampling=0,
                            )

                            record["status"] = (
                                "EXTRACTED"
                            )

                    if not output_is_valid(
                        output_path
                    ):
                        raise RuntimeError(
                            "Output validation "
                            "failed: "
                            f"{output_path}"
                        )

                    record.update(
                        calculate_statistics(
                            model_patch
                        )
                    )

                except Exception as patch_error:
                    record["status"] = (
                        "ERROR"
                    )

                    record["error"] = repr(
                        patch_error
                    )

                records.append(record)

        except Exception as slide_error:
            completed_keys = {
                (
                    record["dataset"],
                    record["slide_id"],
                    record["patch_name"],
                )
                for record in records
            }

            for _, selection_row in (
                slide_rows.iterrows()
            ):
                key = (
                    dataset,
                    slide_id,
                    selection_row[
                        "patch_name"
                    ],
                )

                if key in completed_keys:
                    continue

                record = (
                    selection_row
                    .drop(
                        labels=[
                            "patch_index_numeric"
                        ],
                        errors="ignore",
                    )
                    .to_dict()
                )

                record.update(
                    {
                        "patch_path": "",
                        "absolute_patch_path": "",
                        "raw_image_path": (
                            str(raw_image_path)
                            if raw_image_path
                            else ""
                        ),
                        "annotation_json_path": (
                            str(json_path)
                            if json_path
                            else ""
                        ),
                        "status": "ERROR",
                        "error": (
                            "Slide-level error: "
                            f"{repr(slide_error)}"
                        ),
                    }
                )

                records.append(record)

        finally:
            if tcga_slide is not None:
                tcga_slide.close()

        checkpoint = pd.DataFrame(
            records
        )

        checkpoint.to_csv(
            EXTRACTION_REPORT_PATH,
            index=False,
        )

        successes = (
            checkpoint["status"]
            .isin(
                [
                    "EXTRACTED",
                    "REUSED",
                ]
            )
            .sum()
        )

        errors = (
            checkpoint["status"]
            .eq("ERROR")
            .sum()
        )

        print(
            f"  Successful: {successes} | "
            f"Errors: {errors}"
        )

    report = pd.DataFrame(
        records
    )

    report["patch_index_numeric"] = (
        pd.to_numeric(
            report["patch_index"],
            errors="coerce",
        )
    )

    report = (
        report
        .sort_values(
            [
                "dataset",
                "slide_id",
                "patch_index_numeric",
            ]
        )
        .drop(
            columns=[
                "patch_index_numeric"
            ]
        )
    )

    report.to_csv(
        EXTRACTION_REPORT_PATH,
        index=False,
    )

    successful_count = (
        report["status"]
        .isin(
            [
                "EXTRACTED",
                "REUSED",
            ]
        )
        .sum()
    )

    error_count = (
        report["status"]
        .eq("ERROR")
        .sum()
    )

    print(
        "\nFinal status counts:"
    )

    print(
        report["status"]
        .value_counts(
            dropna=False
        )
    )

    print(
        "\nSuccessful patches:",
        successful_count,
    )

    print(
        "Remaining errors:",
        error_count,
    )

    if (
        successful_count
        != EXPECTED_PATCHES
        or error_count != 0
    ):
        errors = report.loc[
            report["status"]
            == "ERROR",
            [
                "dataset",
                "slide_id",
                "patch_name",
                "error",
            ],
        ]

        print(
            errors
            .head(50)
            .to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Patch extraction did not "
            "complete successfully"
        )

    print(
        f"Saved: "
        f"{EXTRACTION_REPORT_PATH}"
    )


if __name__ == "__main__":
    main()
