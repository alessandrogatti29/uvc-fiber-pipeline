#!/usr/bin/env python3
"""
Step 3b (optional): Repair small near-duplicate-point defects in a set of
boundary-patch surfaces (e.g. from a manual ParaView cut) before
tetrahedralizing them.

Symptom this fixes: tetgen either produces severely degenerate
(radius_ratio > 1e6) elements, or refuses outright with "self-intersecting
mesh", clustered at one or a few specific spots. That's usually a handful
of near-duplicate points left over from however the cut/cap was made
(e.g. ParaView's Clip filter capping a plane cut). A single global weld
tolerance tends to either miss these (too tight) or start merging
unrelated nearby geometry elsewhere and creating NEW self-intersections
(too loose) -- see the README for how we found this out the hard way.
The fix here is surgical: weld points only within a small radius of each
known defect location, leaving everything else untouched.

How to find your own defect locations: try tetrahedralizing first (step
06); if it fails or produces bad quality clustered in one place, the
error/quality report will tell you roughly where (element centroids), or
open the merged surface in ParaView and look for visibly overlapping
geometry.

Usage:
  python 05_repair_surface_defects.py \\
      --surfaces cut_endo_lv.ply cut_endo_rv.ply cut_epi.ply cut_base.ply \\
      --defect-center -10.85 7.70 -232.82 \\
      --defect-radius 5.0 --weld-tolerance 0.03 \\
      --output-suffix _repaired
"""

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--surfaces", required=True, nargs="+", help="Input .ply boundary patches (any number)")
    p.add_argument("--defect-center", type=float, nargs=3, action="append", default=None,
                   metavar=("X", "Y", "Z"),
                   help="A known defect location; repeat this flag for multiple defects")
    p.add_argument("--defect-radius", type=float, default=5.0,
                   help="Radius (mesh units) around each defect center to consider for local welding")
    p.add_argument("--weld-tolerance", type=float, default=0.03,
                   help="Distance below which two points within a defect radius get merged")
    p.add_argument("--global-weld-tolerance", type=float, default=1e-3,
                   help="Baseline weld tolerance applied everywhere (should be tight -- just enough "
                        "to close exact/near-exact shared-border seams between patches)")
    p.add_argument("--output-dir", default=None, help="Default: same directory as each input file")
    p.add_argument("--output-suffix", default="_repaired")
    args = p.parse_args()

    meshes = [pv.read(s).triangulate() for s in args.surfaces]
    merged = meshes[0].merge(meshes[1:]).clean(tolerance=args.global_weld_tolerance)
    merged = pv.PolyData(merged.points, merged.faces).clean(tolerance=1e-6)
    print(f"baseline (tol={args.global_weld_tolerance}): {merged.n_points} points, {merged.n_cells} faces, "
          f"is_all_triangles={merged.is_all_triangles}")

    points = merged.points
    faces = merged.faces.reshape(-1, 4)[:, 1:].copy()

    if args.defect_center:
        parent = {}

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i, j):
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

        n_merged_total = 0
        for center in args.defect_center:
            center = np.array(center)
            dist_to_defect = np.linalg.norm(points - center, axis=1)
            local_idx = np.where(dist_to_defect < args.defect_radius)[0]
            for i in local_idx:
                parent.setdefault(i, i)
            print(f"defect at {center}: {len(local_idx)} points within radius {args.defect_radius}")

            local_pts = points[local_idx]
            tree = cKDTree(local_pts)
            pairs = tree.query_pairs(r=args.weld_tolerance)
            for a, b in pairs:
                union(local_idx[a], local_idx[b])

        remap = np.arange(len(points))
        for i in parent:
            r = find(i)
            if r != i:
                remap[i] = r
                n_merged_total += 1
        print(f"total points merged: {n_merged_total}")

        new_faces = remap[faces]
        keep = (new_faces[:, 0] != new_faces[:, 1]) & \
               (new_faces[:, 1] != new_faces[:, 2]) & \
               (new_faces[:, 0] != new_faces[:, 2])
        print(f"degenerate faces dropped: {(~keep).sum()}")
        faces = new_faces[keep]

        used = np.unique(faces)
        compact = -np.ones(len(points), dtype=np.int64)
        compact[used] = np.arange(len(used))
        points = points[used]
        faces = compact[faces]
    else:
        print("No --defect-center given, skipping local repair (just the baseline weld above).")

    f = np.hstack([np.full((len(faces), 1), 3), faces]).astype(np.int64).flatten()
    repaired = pv.PolyData(points, f)
    print(f"result: {repaired.n_points} points, {repaired.n_cells} faces")

    edges = repaired.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                            manifold_edges=False, non_manifold_edges=False)
    nonman = repaired.extract_feature_edges(boundary_edges=False, feature_edges=False,
                                             manifold_edges=False, non_manifold_edges=True)
    print(f"open boundary edges: {edges.n_cells}, non-manifold edges: {nonman.n_cells}")
    if edges.n_cells or nonman.n_cells:
        print("WARNING: not watertight/manifold -- do not proceed to tetrahedralization until fixed.")

    out_dir = Path(args.output_dir) if args.output_dir else Path(args.surfaces[0]).parent
    out_path = out_dir / f"merged{args.output_suffix}.vtp"
    repaired.save(str(out_path))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
