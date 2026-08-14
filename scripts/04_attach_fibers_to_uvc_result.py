#!/usr/bin/env python3
"""
Step 2c: Re-attach fiber/sheet/normal orientation onto a Cobiveco UVC
result.

If your mesh-quality-optimization step (03) ran between building the
template mesh and running Cobiveco, the fiber/sheet/normal point data
gets lost -- mmg3d's mesh format round-trip only carries geometry, not
arbitrary point data. This recovers it by nearest-neighbor lookup against
the original dense fiber sample cloud (the same file used in step 2),
which is normally far denser than the mesh, so this is accurate even
without needing the Cobiveco output to share exact point correspondence
with the pre-optimization mesh.

Usage:
  python 04_attach_fibers_to_uvc_result.py \\
      --uvc-result result_template/result.vtu \\
      --fibers fibers_all_fixed.txt \\
      --output template_with_fibers.vtu
"""

import argparse
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--uvc-result", required=True, help="Cobiveco/CobivecoX result.vtu for the template")
    p.add_argument("--fibers", required=True, help="Dense fiber sample cloud (fibers_all_fixed.txt)")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    print("Loading UVC result...")
    result = pv.read(args.uvc_result)
    print(f"  {result.n_points} points, fields: {list(result.point_data.keys())}")

    print("Loading dense fiber sample cloud...")
    data = np.loadtxt(args.fibers, skiprows=1)
    fiber_pts = data[:, 7:10]
    fiber = data[:, 16:19]
    sheet = data[:, 19:22]
    print(f"  {len(fiber_pts)} sample points")

    print("Nearest-neighbor matching mesh points to fiber samples...")
    tree = cKDTree(fiber_pts)
    dist, idx = tree.query(result.points)
    print(f"  match distance: min={dist.min():.4f} mean={dist.mean():.4f} max={dist.max():.4f}")
    if dist.mean() > 1.0:
        print("  WARNING: mean match distance is large relative to a typical mesh edge length -- "
              "double check --fibers is the right file for this mesh.")

    matched_fiber = fiber[idx]
    matched_sheet = sheet[idx]
    matched_fiber = matched_fiber / np.linalg.norm(matched_fiber, axis=1, keepdims=True)
    matched_sheet = matched_sheet - np.sum(matched_sheet * matched_fiber, axis=1, keepdims=True) * matched_fiber
    matched_sheet = matched_sheet / np.linalg.norm(matched_sheet, axis=1, keepdims=True)
    matched_normal = np.cross(matched_fiber, matched_sheet)

    result.point_data["fiber"] = matched_fiber
    result.point_data["sheet"] = matched_sheet
    result.point_data["normal"] = matched_normal

    result.save(args.output)
    print(f"\nwrote {args.output}: {result.n_points} points, fields: {list(result.point_data.keys())}")


if __name__ == "__main__":
    main()
