#!/usr/bin/env python3
"""get_models.py - download the model files config.yaml points at, into models/ (gitignored).

  python tools/get_models.py          # what the current config needs (detector + MediaPipe model)
  python tools/get_models.py --all    # also buffalo_l (SCRFD det_10g and the re-ID embedder, Task 12)

Downloads are skipped when the files are already there.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import load_config, repo_path  # noqa: E402

MEDIAPIPE_URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
                 "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite")


def get_mediapipe(path: Path) -> None:
    if path.is_file():
        print(f"have     {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"download {MEDIAPIPE_URL}\n      -> {path}")
    urllib.request.urlretrieve(MEDIAPIPE_URL, path)


def get_insightface_pack(root: Path, pack: str) -> None:
    target = root / "models" / pack
    if target.is_dir() and any(target.glob("*.onnx")):
        print(f"have     {target}")
        return
    from insightface.utils.storage import ensure_available   # downloads and unzips the pack

    root.mkdir(parents=True, exist_ok=True)
    print(f"download InsightFace pack {pack} -> {target}")
    ensure_available("models", pack, root=str(root))
    zip_path = root / "models" / f"{pack}.zip"
    if zip_path.is_file():          # insightface leaves the archive behind (buffalo_l: 280 MB)
        zip_path.unlink()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--all", action="store_true", help="also fetch buffalo_l (det_10g + re-ID embedder)")
    p.add_argument("--config", help="config.yaml to read (default: the repo's)")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    det = cfg["detect"]
    get_mediapipe(repo_path(det.get("mediapipe", {}).get("model_path", "models/blaze_face_short_range.tflite")))
    scrfd = det.get("scrfd", {})
    root = repo_path(scrfd.get("root", "models/insightface"))
    packs = {scrfd.get("pack", "buffalo_sc")}
    if args.all:
        packs |= {"buffalo_l", cfg.get("track", {}).get("reid", {}).get("model", "buffalo_l")}
    for pack in sorted(packs):
        get_insightface_pack(root, pack)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
