# Native Patch Images

This directory is reserved for the fixed raw-pixel patch sample used by
InternVL3-8B and Qwen3-VL-8B-Instruct.

Expected local payloads are `TCGA/<slide>/<patch_name>` and
`BCNB/<slide>/<patch_name>`. The generated `native_patch_manifest.csv` and
`patch_acquisition_report.json` are tracked, while image and WSI payloads remain
local. See `docs/phase2_patch_acquisition.md` for the complete handoff.
