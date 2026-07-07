"""Convert a results.json (from run_eval) into LaTeX table rows that can be
pasted into paper/main.tex to replace the assumed values with measured ones.

    python scripts/results_to_latex.py --results outputs/results.json --out outputs/tables.tex
"""

from __future__ import annotations

import argparse
import json

_DISPLAY = {
    "full-resolution": "Full-resolution (bound)",
    "coarse-only": "Coarse-only (bound)",
    "fixed-512": "Fixed budget (512)",
    "tiling": "Sliding-window/tiling",
    "tome": "ToMe~\\cite{bolya2023tome}",
    "fastv": "FastV~\\cite{chen2024fastv}",
    "dynamicvit": "DynamicViT~\\cite{rao2021dynamicvit}",
    "prato": "PrATo~\\cite{dutta2025prato}",
    "grasp": "GRASP~\\cite{sun2026grasp}",
    "dualcomp": "DualComp~\\cite{li2026dualcomp}",
    "random-regional": "Random regional",
    "entropy-only": "Entropy-only (no spatial)",
    "attention-only": "Attention-only attribution",
    "uavib": "\\textbf{\\method~(ours)}",
    "oracle": "\\textit{Oracle budget (upper bound)}",
}

_DATASET_ORDER = ["rsvqa-lr", "rsvqa-hr", "rsivqa", "xlrs-bench", "acdc", "isic", "mvtec-ad"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="outputs/results.json")
    ap.add_argument("--out", default="outputs/tables.tex")
    args = ap.parse_args()

    with open(args.results, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    res = data["results"]

    lines = ["% Auto-generated main accuracy--cost table rows"]
    for m in _DISPLAY:
        if m not in res:
            continue
        row = res[m]
        allm = row["_all"]
        cells = []
        for d in _DATASET_ORDER:
            cells.append(f"{row[d]['accuracy']:.1f}" if d in row else "--")
        lines.append(
            f"{_DISPLAY[m]} & {allm['avg_tokens']:.0f} & "
            + " & ".join(cells)
            + f" & {allm['accuracy']:.1f} \\\\"
        )

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Wrote {args.out} ({len(lines)-1} rows)")


if __name__ == "__main__":
    main()
