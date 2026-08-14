#!/usr/bin/env python3
"""
Step 1: Fix left-handed fiber frames in a Continuity/CHeart-style fiber
export.

Continuity's per-sample fiber export ("...fibres.txt"-style, one row per
sample point, with a 3x3 dY_dMatl orientation matrix per row) sometimes
contains elements where the stored normal is the *negative* of
fiber x sheet -- i.e. a left-handed frame (det(dY_dMatl) = -1 instead of
+1). This is a real, systematic issue we found affecting ~40% of elements
in one template file, all consistent per-element (never mixed within a
single element), consistent with elements that were fit/labeled with a
flipped local convention.

The fix is exact and lossless: recompute normal = fiber x sheet everywhere.
For already-correct (right-handed) elements this is a no-op, since
fiber x sheet already equals the stored normal there; for left-handed
elements it corrects the sign.

Expected input column layout (Continuity/CHeart fiber export convention):
  col 0        : Elem
  col 1-3      : Z[0..2]      (material coords)
  col 4-6      : xi[0..2]     (element-local coords)
  col 7-9      : X[0..2]      (spatial coords)
  col 10-12    : Y[0..2]
  col 13-15    : y[0..2]
  col 16-24    : dY_dMatl[0,0..2], dY_dMatl[1,0..2], dY_dMatl[2,0..2]
                 (row 0 = fiber/longitudinal, row 1 = sheet/transversal,
                 row 2 = normal -- confirm this convention against your
                 own Continuity export documentation before trusting it
                 for a different source file)
  col 25       : Index

Usage:
  python 01_fix_fiber_handedness.py --input fibers_all.txt --output fibers_all_fixed.txt
"""

import argparse
import numpy as np

FIBER_COLS = slice(16, 19)
SHEET_COLS = slice(19, 22)
NORMAL_COLS = slice(22, 25)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Input Continuity fiber export (e.g. fibers_all.txt)")
    p.add_argument("--output", required=True, help="Output path for the handedness-corrected file")
    args = p.parse_args()

    with open(args.input) as f:
        header = f.readline()

    print(f"Loading {args.input} ...")
    data = np.loadtxt(args.input, skiprows=1)
    elem = data[:, 0].astype(np.int64)
    fiber = data[:, FIBER_COLS]
    sheet = data[:, SHEET_COLS]
    normal = data[:, NORMAL_COLS]

    cross = np.cross(fiber, sheet)
    det = np.einsum("ij,ij->i", cross, normal)  # (fiber x sheet) . normal == det(dY_dMatl)
    negative = det < 0

    print(f"Total rows: {len(data)}")
    print(f"Rows with det=-1 (left-handed): {negative.sum()}")
    print(f"Rows with det=+1 (right-handed): {(~negative).sum()}")

    flipped_elems = []
    mixed_elems = []
    for e in np.unique(elem):
        m = elem == e
        signs = np.sign(det[m])
        if signs.min() != signs.max():
            mixed_elems.append(int(e))
        elif signs[0] < 0:
            flipped_elems.append(int(e))
    print(f"Elements entirely left-handed: {len(flipped_elems)} -> {flipped_elems}")
    if mixed_elems:
        print(f"WARNING: elements with MIXED handedness (unexpected, inspect manually): {mixed_elems}")

    data[:, NORMAL_COLS] = cross  # no-op where already correct, fixes the rest

    print(f"Writing {args.output} ...")
    with open(args.output, "w") as f:
        f.write(header)
        np.savetxt(f, data, fmt="%.6g", delimiter="\t")
    print("Done.")


if __name__ == "__main__":
    main()
