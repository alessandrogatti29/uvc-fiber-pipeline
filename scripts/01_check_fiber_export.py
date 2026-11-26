#!/usr/bin/env python3
"""
Step 1: Check a Continuity fiber export before using it. This step only
reads the file. Nothing later needs the file changed, so it is never
rewritten.

Column layout of the export (0-based), one row per sample point:
  col 0        : Elem
  col 1-3      : Z[0..2]      (material coords)
  col 4-6      : xi[0..2]     (element-local coords)
  col 7-9      : X[0..2]      (spatial coords)
  col 10-15    : Y[0..2], y[0..2]
  col 16-24    : dY_dMatl, written row by row: dY_dMatl[i,j] is in
                 column 16 + 3*i + j
  col 25       : Index

The basis vectors are the COLUMNS of dY_dMatl (confirmed by the author of
the export):
  column 0 = fiber          -> file columns 16, 19, 22
  column 1 = sheet-normal   -> file columns 17, 20, 23
  column 2 = sheet          -> file columns 18, 21, 24
Row 0 (file columns 16-18) is NOT the fiber. It holds the x-components of
all three vectors. Steps 2a and 2c read the fiber and sheet columns and
recompute normal = fiber x sheet, so the sign of det(dY_dMatl) does not
matter downstream.

What is checked:
  - the header puts X and dY_dMatl in the columns above (fails otherwise)
  - every dY_dMatl is orthonormal (fails above --tol)
  - the sign of det(dY_dMatl) (reported only)
  - the frame varies within elements. If every element carries one
    constant frame, the file has the Continuity export bug that affected
    our first template export: this fails, and you need a corrected export
  - fiber continuity across element faces: the angle between the fiber
    axes of coincident samples that belong to different elements. A
    correct export gives 0 deg; the buggy one gave a median of about
    40 deg. A median above 5 deg is reported as a warning.

Exits with status 1 if any check fails.

Usage:
  python 01_check_fiber_export.py --fibers fibers_all_corrected.txt
"""

import argparse
import sys

import numpy as np

EXPECTED_HEADER = {0: "Elem", 7: "X[0]", 16: "dY_dMatl[0,0]", 24: "dY_dMatl[2,2]"}


def axis_angle_deg(a, b):
    """Angle between two axes, in degrees (f and -f count as the same fiber)."""
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip(np.abs(np.sum(a * b, axis=1)), 0.0, 1.0)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fibers", required=True, help="Continuity fiber export (e.g. fibers_all_corrected.txt)")
    p.add_argument("--tol", type=float, default=1e-3,
                   help="Largest allowed entry of |dY_dMatl^T dY_dMatl - I| (default 1e-3)")
    p.add_argument("--weld-decimals", type=int, default=5,
                   help="Coordinates are rounded to this many decimals to find coincident samples "
                        "(the same rounding step 2a uses to weld elements; default 5)")
    args = p.parse_args()

    failed = False

    with open(args.fibers) as fh:
        header = fh.readline().split()
    for col, name in EXPECTED_HEADER.items():
        if col >= len(header) or header[col] != name:
            found = header[col] if col < len(header) else "nothing"
            print(f"FAIL: expected column {col} to be '{name}', found {found}. "
                  "The column numbers used by steps 2a/2c do not apply to this file.")
            sys.exit(1)

    print(f"Loading {args.fibers} ...")
    data = np.loadtxt(args.fibers, skiprows=1)
    elem = data[:, 0].astype(np.int64)
    X = data[:, 7:10]
    A = data[:, 16:25].reshape(-1, 3, 3)  # A[r, i, j] = dY_dMatl[i, j] of sample r
    del data
    fiber = A[:, :, 0]
    elem_ids, elem_first, elem_inv = np.unique(elem, return_index=True, return_inverse=True)
    elem_inv = elem_inv.ravel()
    print(f"  {len(A)} samples, {len(elem_ids)} elements")

    G = np.einsum("rki,rkj->rij", A, A)  # dY_dMatl^T dY_dMatl
    orth_err = np.abs(G - np.eye(3)).max(axis=(1, 2))
    print(f"Orthonormality: max |dY_dMatl^T dY_dMatl - I| = {orth_err.max():.2e}")
    if orth_err.max() > args.tol:
        print(f"  FAIL: {np.sum(orth_err > args.tol)} samples exceed --tol {args.tol}")
        failed = True

    det = np.linalg.det(A)
    n_neg = int(np.sum(det < 0))
    print(f"det(dY_dMatl): {len(det) - n_neg} samples +1, {n_neg} samples -1"
          + (" (harmless here: the normal is recomputed as fiber x sheet)" if n_neg else ""))

    print("Variation of the fiber within each element...")
    within = axis_angle_deg(fiber, fiber[elem_first[elem_inv]])
    max_within = np.zeros(len(elem_ids))
    np.maximum.at(max_within, elem_inv, within)
    n_const = int(np.sum(max_within < 1e-3))
    print(f"  elements with one constant frame: {n_const} / {len(elem_ids)}; "
          f"median over elements of the largest in-element deviation: {np.median(max_within):.1f} deg")
    if n_const == len(elem_ids):
        print("  FAIL: every element has a single constant frame. This is the Continuity export "
              "bug; fibers jump at every element face. Ask for a corrected export.")
        failed = True
    elif n_const:
        print(f"  NOTE: constant-frame elements: {elem_ids[max_within < 1e-3].tolist()}")

    print("Fiber continuity across element faces...")
    key = np.round(X, args.weld_decimals)
    _, weld_first, weld_inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    weld_ref = weld_first[weld_inv.ravel()]
    across = elem != elem[weld_ref]
    if not across.any():
        print("  no coincident samples from different elements found (check --weld-decimals)")
    else:
        jump = axis_angle_deg(fiber[across], fiber[weld_ref[across]])
        print(f"  {across.sum()} face samples: median {np.median(jump):.2f} deg, "
              f"p95 {np.percentile(jump, 95):.2f} deg, max {jump.max():.2f} deg")
        if np.median(jump) > 5.0:
            print("  WARNING: the fiber jumps at element faces. A correct export is continuous there.")

    print("\nFiber = dY_dMatl column 0 (file columns 16, 19, 22); "
          "sheet = column 2 (file columns 18, 21, 24).")
    if failed:
        print("RESULT: FAIL. Do not use this file.")
        sys.exit(1)
    print("RESULT: OK")


if __name__ == "__main__":
    main()
