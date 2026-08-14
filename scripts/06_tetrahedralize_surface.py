#!/usr/bin/env python3
"""
Step 3c: Tetrahedralize a closed, watertight boundary surface into a
volume mesh with tetgen, preserving the input boundary exactly (tetgen's
-Y switch -- no re-triangulation of the surface you give it, so the
result's boundary is bit-identical to your input, and any labeled patches
you built the surface from stay valid).

Accepts either a single already-merged watertight surface (e.g. the
output of step 05), or a list of separate boundary patches to merge here.

Usage:
  python 06_tetrahedralize_surface.py --merged-surface merged_repaired.vtp --output volume.vtu
  # or:
  python 06_tetrahedralize_surface.py --surfaces endo_lv.ply endo_rv.ply epi.ply base.ply --output volume.vtu
"""

import argparse
import numpy as np
import pyvista as pv
import tetgen


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--surfaces", nargs="+", default=None, help="Boundary patches to merge (.ply)")
    p.add_argument("--merged-surface", default=None, help="Already-merged watertight surface (.vtp/.ply)")
    p.add_argument("--output", required=True, help="Output volume .vtu")
    p.add_argument("--weld-tolerance", type=float, default=1e-3,
                   help="Only used with --surfaces, to weld shared patch borders")
    p.add_argument("--tetgen-switches", default="pYq1.4/10",
                   help="tetgen switches (default: -Y boundary-preserving, -q1.4 quality bound)")
    args = p.parse_args()

    if args.merged_surface:
        merged = pv.read(args.merged_surface)
    elif args.surfaces:
        meshes = [pv.read(s).triangulate() for s in args.surfaces]
        merged = meshes[0].merge(meshes[1:]).clean(tolerance=args.weld_tolerance)
    else:
        raise SystemExit("Provide either --merged-surface or --surfaces")

    # clean() can leave stray vert/line cells from welding degenerate geometry;
    # tetgen requires a pure-triangle mesh
    merged = pv.PolyData(merged.points, merged.faces).clean(tolerance=1e-6)
    print(f"input surface: {merged.n_points} points, {merged.n_cells} faces, "
          f"is_all_triangles={merged.is_all_triangles}")

    edges = merged.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                          manifold_edges=False, non_manifold_edges=False)
    print(f"open boundary edges: {edges.n_cells}")
    if edges.n_cells != 0:
        raise RuntimeError("Surface is not watertight, refusing to tetrahedralize. "
                            "Run step 05 (repair_surface_defects) first, or fix your surface.")

    print(f"Tetrahedralizing (switches='{args.tetgen_switches}')...")
    tgen = tetgen.TetGen(merged)
    tgen.tetrahedralize(switches=args.tetgen_switches)
    grid = tgen.grid
    grid.save(args.output)
    print(f"wrote {args.output}: {grid.n_points} points, {grid.n_cells} cells")

    q = grid.cell_quality(quality_measure="radius_ratio")
    v = q.cell_data["radius_ratio"]
    print(f"quality: min={v.min():.4f} median={np.median(v):.4f} mean={v.mean():.2f} "
          f"n>100={np.sum(v>100)} n>1e6={np.sum(v>1e6)}")
    if np.sum(v > 1e6) > 0:
        print("There are severely degenerate elements -- consider step 03 "
              "(optimize_mesh_quality) next, or step 05 if quality issues cluster "
              "at a specific location (may indicate an unrepaired surface defect).")

    surf = grid.extract_surface(algorithm="dataset_surface")
    print(f"volume boundary: {surf.n_points} pts / {surf.n_cells} faces "
          f"(input surface had {merged.n_points} pts / {merged.n_cells} faces)")


if __name__ == "__main__":
    main()
