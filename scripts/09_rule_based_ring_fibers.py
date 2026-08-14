#!/usr/bin/env python3
"""
Step 6: Fill in a rule-based fiber direction for any region left
unassigned (fiberValid=0) after step 08's spatial transfer -- typically
the area right around valve/ring annuli that got cut away from the source
mesh and so were never covered by the transfer.

The rule: blend from the (already-present, UVC/spatially-mapped) fiber
direction into a direction parallel to the nearest ring, weighted by
proximity to that ring. Far from any ring, you keep the existing fiber
direction; right at a ring, you get pure circumferential (ring-parallel);
in between, a smooth blend. This matches the physiological expectation
that fibers run roughly parallel to a fibrous annulus as they approach it.
Only points with fiberValid=0 are touched.

Ring geometry is fit per surface with PCA (best-fit plane + circle) rather
than traced from the mesh boundary directly, which is more robust to
irregular/coarse ring meshes.

Usage:
  python 09_rule_based_ring_fibers.py \\
      --input full_mesh_fibers_partial.vtu \\
      --rings av.ply mv.ply pv.ply tv.ply \\
      --transition-distance 15.0 \\
      --output full_mesh_fibers_complete.vtu
"""

import argparse
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree


def fit_ring(points):
    """PCA-fit a plane+circle to a ring's point cloud."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    e1, e2 = vt[0], vt[1]
    proj = centered @ np.vstack([e1, e2]).T
    radius = np.linalg.norm(proj, axis=1).mean()
    return centroid, e1, e2, radius


def ring_tangent(query_points, centroid, e1, e2):
    """In-plane tangent of the fitted circle at each query point's
    angular position."""
    rel = query_points - centroid
    x = rel @ e1
    y = rel @ e2
    theta = np.arctan2(y, x)
    tangent = (-np.sin(theta))[:, None] * e1 + (np.cos(theta))[:, None] * e2
    return tangent / np.linalg.norm(tangent, axis=1, keepdims=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Mesh with partial fiber/sheet + fiberValid (from step 08)")
    p.add_argument("--rings", required=True, nargs="+",
                   help="Ring/annulus surface .ply files (any number, e.g. valve surfaces)")
    p.add_argument("--transition-distance", type=float, default=15.0,
                   help="Distance (mesh units) over which the blend goes from 0 to 1; "
                        "tune based on visual inspection (color by the output 'ruleWeight' field)")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    print("Loading partial fiber field...")
    m = pv.read(args.input)
    if "fiberValid" not in m.point_data:
        raise SystemExit("Input mesh has no 'fiberValid' field -- run step 08 first.")
    valid = m.point_data["fiberValid"].astype(bool)
    fiber = m.point_data["fiber"].copy()
    sheet = m.point_data["sheet"].copy()
    print(f"  {m.n_points} points, {valid.sum()} valid, {(~valid).sum()} invalid")

    if (~valid).sum() == 0:
        print("Nothing to fill in -- writing input through unchanged.")
        m.save(args.output)
        return

    print("Extending valid fiber field to invalid points (nearest-valid copy)...")
    valid_tree = cKDTree(m.points[valid])
    _, nn_idx = valid_tree.query(m.points[~valid])
    valid_idx = np.where(valid)[0]
    fiber_ext = fiber.copy()
    sheet_ext = sheet.copy()
    fiber_ext[~valid] = fiber[valid_idx[nn_idx]]
    sheet_ext[~valid] = sheet[valid_idx[nn_idx]]

    print(f"Fitting rings to {len(args.rings)} surface(s)...")
    rings, ring_trees = {}, {}
    for path in args.rings:
        vm = pv.read(path)
        rings[path] = fit_ring(vm.points)
        ring_trees[path] = cKDTree(vm.points)
        print(f"  {path}: {vm.n_points} points, radius={rings[path][3]:.2f}")

    print("Computing distance and tangent to nearest ring for invalid points...")
    inv_idx = np.where(~valid)[0]
    inv_pts = m.points[inv_idx]

    best_dist = np.full(len(inv_pts), np.inf)
    best_tangent = np.zeros((len(inv_pts), 3))
    for path in args.rings:
        d, _ = ring_trees[path].query(inv_pts)
        centroid, e1, e2, _ = rings[path]
        tang = ring_tangent(inv_pts, centroid, e1, e2)
        better = d < best_dist
        best_dist[better] = d[better]
        best_tangent[better] = tang[better]

    print(f"  distance to nearest ring: min={best_dist.min():.2f} "
          f"median={np.median(best_dist):.2f} max={best_dist.max():.2f}")

    w = np.clip(1 - best_dist / args.transition_distance, 0, 1)
    print(f"  blend weight: {np.sum(w > 0)} points affected (w>0), {np.sum(w == 1)} fully ring-parallel")

    fiber_inv = fiber_ext[inv_idx]
    sign = np.sign(np.sum(best_tangent * fiber_inv, axis=1))
    sign[sign == 0] = 1
    tangent_signed = best_tangent * sign[:, None]

    blended = (1 - w[:, None]) * fiber_inv + w[:, None] * tangent_signed
    blended = blended / np.linalg.norm(blended, axis=1, keepdims=True)
    fiber[inv_idx] = blended

    print("Re-deriving sheet/normal for all invalid points...")
    sheet_inv = sheet_ext[inv_idx]
    sheet_inv = sheet_inv - np.sum(sheet_inv * blended, axis=1, keepdims=True) * blended
    sheet_inv = sheet_inv / np.linalg.norm(sheet_inv, axis=1, keepdims=True)
    sheet[inv_idx] = sheet_inv

    normal = np.cross(fiber, sheet)

    m.point_data["fiber"] = fiber
    m.point_data["sheet"] = sheet
    m.point_data["normal"] = normal
    m.point_data["ruleWeight"] = np.zeros(m.n_points)
    m.point_data["ruleWeight"][inv_idx] = w

    fn = np.linalg.norm(fiber, axis=1)
    dot_fs = np.sum(fiber * sheet, axis=1)
    print(f"\nfinal check: fiber norm min={fn.min():.4f} max={fn.max():.4f}, "
          f"max |fiber.sheet|={np.abs(dot_fs).max():.6f}, any NaN={np.isnan(fiber).any()}")

    m.save(args.output)
    print(f"wrote {args.output}: {m.n_points} points, all fibers assigned")


if __name__ == "__main__":
    main()
