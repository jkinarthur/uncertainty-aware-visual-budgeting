"""Prepare the RSVQA-LR (real remote-sensing VQA) dataset for UAViB.

Downloads the public ``dmarsili/RSVQA-LR-2k`` subset (CC-BY-4.0, a 2k sample of
the RSVQA-LR validation split), keeps the yes/no questions, and writes a
UAViB-compatible on-disk layout:

    <out>/rsvqa/
        images/000000.png ...
        calib.jsonl      (held-out pool for training the calibrator)
        test.jsonl       (evaluation split)

Each manifest line:
    {"dataset": "rsvqa-lr", "image": "images/000000.png",
     "question": "...", "candidates": ["yes","no"], "answer": "yes"}

Usage:
    python deploy/prepare_rsvqa.py --out data/raw --calib 80 --test 200
"""

from __future__ import annotations

import argparse
import json
import os
import random


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw", help="root data dir")
    ap.add_argument("--calib", type=int, default=80, help="calibration samples")
    ap.add_argument("--test", type=int, default=200, help="test samples")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset("dmarsili/RSVQA-LR-2k", split="validation")

    # Keep clean binary questions (RSVQA presence/comparison answers).
    idx = [i for i, a in enumerate(ds["answer"]) if str(a).lower() in ("yes", "no")]
    random.Random(args.seed).shuffle(idx)

    need = args.calib + args.test
    if len(idx) < need:
        raise SystemExit(f"Only {len(idx)} yes/no samples available, need {need}")
    calib_idx = idx[: args.calib]
    test_idx = idx[args.calib: args.calib + args.test]

    ds_dir = os.path.join(args.out, "rsvqa")
    img_dir = os.path.join(ds_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    def write_split(indices, split_name):
        path = os.path.join(ds_dir, f"{split_name}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for j, i in enumerate(indices):
                rec = ds[i]
                img = rec["image"].convert("RGB")
                rel = f"images/{split_name}_{j:05d}.png"
                img.save(os.path.join(ds_dir, rel))
                fh.write(json.dumps({
                    "dataset": "rsvqa-lr",
                    "image": rel,
                    "question": rec["question"],
                    "candidates": ["yes", "no"],
                    "answer": str(rec["answer"]).lower(),
                }) + "\n")
        print(f"  wrote {len(indices)} -> {path}")

    print(f"Preparing RSVQA-LR under {ds_dir} ...")
    write_split(calib_idx, "calib")
    write_split(test_idx, "test")
    print("Done.")


if __name__ == "__main__":
    main()
