"""
ngam.downloader
===============
Model weight management and Hugging Face artifact caching for the ngam
Universal Decision Engine. Supports automatic downloading, local offline
caching in ~/.cache/ngam/models, and linking from local dev repositories.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("ngam.downloader")

DEFAULT_REPO_ID = "inferenceprince/laya-onnx-int8"
FALLBACK_REPO_ID = "inferenceprince/laya-onnx"

# Default standalone cache directory: ~/.cache/ngam/models/ngam_onnx
DEFAULT_CACHE_DIR = (Path.home() / ".cache" / "ngam" / "models" / "ngam_onnx").resolve()
DEFAULT_MODEL_DIR = str(DEFAULT_CACHE_DIR)

# Local fallback sources
LEGACY_CACHE_DIR = (Path.home() / ".cache" / "ngam" / "models" / "laya_onnx").resolve()
LOCAL_DEV_DIR = Path("c:/dev/dev/models/laya_onnx")

REQUIRED_FILES = [
    "model.onnx",
    "model.onnx.data",
    "rl_agent_config.json",
    os.path.join("tokenizer", "tokenizer.json"),
]

OPTIONAL_FILES = [
    "config.json",
    os.path.join("tokenizer", "tokenizer_config.json"),
]


def _has_required_files(path: Path) -> bool:
    """Check if all required files exist with non-zero size."""
    if not path.is_dir():
        return False
    for rel_path in REQUIRED_FILES:
        fpath = path / rel_path
        if not fpath.is_file() or fpath.stat().st_size == 0:
            return False
    return True


def _link_or_copy(src: Path, dst: Path) -> None:
    """Attempt hardlink first; fallback to copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() and dst.stat().st_size > 0:
        return
    try:
        os.link(str(src), str(dst))
    except Exception:
        shutil.copy2(str(src), str(dst))


def _sync_from_local_source(src_dir: Path, target_dir: Path) -> bool:
    """Copy or hardlink local model files from a local source directory."""
    if not _has_required_files(src_dir):
        return False
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        for rel_file in REQUIRED_FILES + OPTIONAL_FILES:
            src_file = src_dir / rel_file
            if src_file.is_file():
                dest_file = target_dir / rel_file
                _link_or_copy(src_file, dest_file)
        return _has_required_files(target_dir)
    except Exception as e:
        logger.warning(f"Could not link or copy from local source {src_dir}: {e}")
        return False


def is_model_available(model_dir: Optional[str] = None) -> bool:
    """
    Check whether all critical ONNX model and tokenizer assets exist locally.
    Auto-links from existing local model caches if target is default cache.
    """
    target = Path(model_dir or os.environ.get("NGAM_MODEL_DIR") or os.environ.get("AGY_DECIDE_MODEL_DIR") or DEFAULT_CACHE_DIR).resolve()
    
    if _has_required_files(target):
        return True

    # If checking the default cache and it is not yet populated, check local fallback sources
    if target == DEFAULT_CACHE_DIR:
        for candidate in [LEGACY_CACHE_DIR, LOCAL_DEV_DIR]:
            if candidate.is_dir() and _has_required_files(candidate):
                if _sync_from_local_source(candidate, target):
                    return True

    return False


def ensure_model(
    model_dir: Optional[str] = None,
    repo_id: str = DEFAULT_REPO_ID,
    force_download: bool = False,
    offline_mode: bool = False,
) -> str:
    """
    Ensure the ngam ONNX model weights, tokenizer, and calibration configs are available.
    
    Args:
        model_dir: Directory where model weights are stored (default ~/.cache/ngam/models/ngam_onnx).
        repo_id: Hugging Face repository id (defaults to DEFAULT_REPO_ID).
        force_download: If True, re-download assets even if present.
        offline_mode: If True, never attempt network requests; error if assets are missing.
        
    Returns:
        Absolute path to the validated model directory.
    """
    env_dir = os.environ.get("NGAM_MODEL_DIR") or os.environ.get("AGY_DECIDE_MODEL_DIR")
    target = Path(model_dir or env_dir or DEFAULT_CACHE_DIR).resolve()

    if not force_download:
        if _has_required_files(target):
            logger.info(f"Using cached ngam model from: {target} (offline ready)")
            return str(target)

        # Check local fallback sources only when targeting default cache dir
        if target == DEFAULT_CACHE_DIR:
            for candidate in [LEGACY_CACHE_DIR, LOCAL_DEV_DIR]:
                if candidate.is_dir() and _has_required_files(candidate):
                    logger.info(f"Populating cache from local source: {candidate} -> {target}")
                    if _sync_from_local_source(candidate, target):
                        return str(target)

    if offline_mode:
        raise FileNotFoundError(
            f"Offline mode requested, but required model files are missing in: {target}. "
            f"Expected files: {REQUIRED_FILES}"
        )

    target.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading ngam model from Hugging Face '{repo_id}' into '{target}'...")

    try:
        from huggingface_hub import hf_hub_download

        files_to_fetch = REQUIRED_FILES + OPTIONAL_FILES
        for rel_file in files_to_fetch:
            dest_file = target / rel_file
            if not force_download and dest_file.is_file() and dest_file.stat().st_size > 0:
                continue

            dest_file.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Fetching {rel_file}...")
            hub_filename = rel_file.replace("\\", "/")

            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=hub_filename,
                    local_dir=str(target),
                )
            except Exception as dl_err:
                if rel_file in REQUIRED_FILES:
                    logger.error(f"Failed downloading required asset '{rel_file}' from '{repo_id}': {dl_err}")
                    raise dl_err
                else:
                    logger.warning(f"Optional asset '{rel_file}' not found on '{repo_id}', skipping.")

    except Exception as e:
        if not is_model_available(str(target)):
            raise RuntimeError(
                f"Failed to acquire ngam model from '{repo_id}'. Error: {e}"
            ) from e

    if not is_model_available(str(target)):
        raise RuntimeError(f"Model directory verification failed after download: {target}")

    logger.info(f"Successfully verified ngam model at: {target}")
    return str(target)
