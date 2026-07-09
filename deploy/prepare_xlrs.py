"""Prepare XLRS-Bench-lite (ultra-high-res RS, multiple-choice) for UAViB.

Streams ``initiacms/XLRS-Bench-lite_VLM``, decodes the base64 images, caps the
longest side (100MP originals would OOM the GPU / fill disk), and writes the
UAViB on-disk layout under ``<out>/xlrs_bench/``:
    images/*.png, calib.jsonl, test.jsonl

Each manifest line:
    {"dataset": "xlrs-bench", "image": "images/....png",
     "question": "<q>\n<options>\nAnswer with the option's letter only.",
     "candidates": ["A","B","C","D"], "answer": "B",
     "meta": {"category": "..."}}

Multiple-choice is scored by the letter, so options are embedded in the prompt.
Per-category caps keep the splits diverse. Sequential streaming (no shuffle
buffer) keeps memory bounded.

Usage:
  python deploy/prepare_xlrs.py --calib 60 --test 120 --per-cat 30 --max-side 2048
"""
from __future__ import annotations

import argparse, io, base64, json, os
from collections import defaultdict
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # allow the 100MP originals


def decode_img(v):
    if isinstance(v, list):
        v = v[0]
    if isinstance(v, Image.Image):
        return v.convert("RGB")
    if isinstance(v, (bytes, bytearray)):
        return Image.open(io.BytesIO(v)).convert("RGB")
    return Image.open(io.BytesIO(base64.b64decode(v))).convert("RGB")


def parse_letters(opts):
    return [L for L in "ABCDEFGH" if f"({L})" in opts] or ["A", "B", "C", "D"]


def cap_side(img, max_side):
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    s = max_side / m
    return img.resize((int(w * s), int(h * s)), Image.BILINEAR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--calib", type=int, default=60)
    ap.add_argument("--test", type=int, default=120)
    ap.add_argument("--per-cat", type=int, default=30, help="cap per category")
    ap.add_argument("--max-side", type=int, default=2048, help="cap longest side px")
    ap.add_argument("--scan-cap", type=int, default=800,
                    help="stop after scanning this many records (avoids stalling on rare cats)")
    ap.add_argument("--cats", default="",
                    help="comma-separated substrings; keep only categories matching one")
    ap.add_argument("--exclude-cats", default="",
                    help="comma-separated substrings; drop categories matching one")
    ap.add_argument("--skip", type=int, default=0,
                    help="skip this many leading records before collecting")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hf-id", default="initiacms/XLRS-Bench-lite_VLM")
    args = ap.parse_args()

    allow = [c.strip().lower() for c in args.cats.split(",") if c.strip()]
    block = [c.strip().lower() for c in args.exclude_cats.split(",") if c.strip()]

    from datasets import load_dataset
    ds = load_dataset(args.hf_id, split="train", streaming=True)

    ds_dir = os.path.join(args.out, "xlrs_bench")
    img_dir = os.path.join(ds_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    need = args.calib + args.test
    rows = []
    cat_count = defaultdict(int)
    scanned = 0
    for r in ds:
        scanned += 1
        if scanned <= args.skip:
            continue
        if scanned > args.scan_cap:
            print(f"  scan-cap {args.scan_cap} reached with {len(rows)} rows", flush=True)
            break
        opts = str(r["multi-choice options"])
        ans = str(r["answer"]).strip().upper()[:1]
        letters = parse_letters(opts)
        if ans not in letters:
            continue
        cat = str(r.get("category", "?"))
        cl = cat.lower()
        if allow and not any(a in cl for a in allow):
            continue
        if block and any(b in cl for b in block):
            continue
        if cat_count[cat] >= args.per_cat:
            continue
        try:
            img = cap_side(decode_img(r["image"]), args.max_side)
        except Exception:
            continue
        cat_count[cat] += 1
        rows.append((img, str(r["question"]), opts, letters, ans, cat))
        if len(rows) >= need:
            break
        if len(rows) % 20 == 0:
            print(f"  collected {len(rows)}/{need} scanned={scanned} (cats={dict(cat_count)})", flush=True)

    # Stratified split: shuffle (fixed seed) so calib and test span the same
    # category mix instead of calib=early-blocks / test=late-blocks.
    import random
    random.Random(args.seed).shuffle(rows)
    calib_rows = rows[: args.calib]
    test_rows = rows[args.calib: args.calib + args.test]

    def write_split(split_rows, split_name):
        path = os.path.join(ds_dir, f"{split_name}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for j, (img, q, opts, letters, ans, cat) in enumerate(split_rows):
                rel = f"images/{split_name}_{j:05d}.png"
                img.save(os.path.join(ds_dir, rel))
                prompt = f"{q}\n{opts}\nAnswer with the option's letter only."
                fh.write(json.dumps({
                    "dataset": "xlrs-bench",
                    "image": rel,
                    "question": prompt,
                    "candidates": letters,
                    "answer": ans,
                    "meta": {"category": cat},
                }) + "\n")
        print(f"  wrote {len(split_rows)} -> {path}", flush=True)

    print(f"Preparing xlrs-bench under {ds_dir} (max_side={args.max_side}) ...", flush=True)
    write_split(calib_rows, "calib")
    write_split(test_rows, "test")
    print(f"Done. category distribution: {dict(cat_count)}", flush=True)


if __name__ == "__main__":
    main()
