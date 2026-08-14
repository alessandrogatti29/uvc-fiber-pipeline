#!/usr/bin/env python3
"""
Step 2b / 3b: Improve tetrahedral element quality of a volume mesh with
mmg3d, then re-derive labeled endo_LV/endo_RV/epi/base boundary surfaces
from the result.

Use this on any boundary-labeled tet mesh whose quality needs fixing
(e.g. straight after step 2's template build, or after cutting/rebuilding
a valve-containing biventricular mesh in step 3). It lets mmg3d adapt the
surface as well as the interior (not just the interior with the boundary
locked), since in practice the boundary triangulation is usually where the
worst-quality elements originate (e.g. from a naive diagonal-split of a
structured grid, or from whatever tool produced your surface cut) --
locking it just means mmg3d can't fix what's actually wrong. Region labels
aren't passed through mmg3d at all; they're re-derived afterward by
cutting the new boundary along sharp (dihedral-angle) creases and taking
connected components, since that's robust to mmg3d changing the exact
point set.

LV vs RV disambiguation: if the mesh segments into more than 2 regions
besides base/epi, or you need to know which of two candidate regions is
LV vs RV, pass --tv-reference pointing at any existing Cobiveco/CobivecoX
result.vtu for the same anatomy (it just needs a 'tv' point-data field,
1=LV/0=RV) -- points get matched by nearest-neighbor to disambiguate.
Without it, ambiguous extra regions are written as endo_0, endo_1, ...
for manual inspection.

Usage:
  python 03_optimize_mesh_quality.py \\
      --volume template_volume.vtu \\
      --out-dir template_geometry --out-prefix template \\
      --mmg3d-bin /path/to/mmg3d_O3
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def write_medit_volume(path, points, tets):
    with open(path, "w") as f:
        f.write("MeshVersionFormatted 2\nDimension 3\n")
        f.write("Vertices\n")
        f.write(f"{len(points)}\n")
        for p in points:
            f.write(f"{p[0]:.10f} {p[1]:.10f} {p[2]:.10f} 0\n")
        f.write("Tetrahedra\n")
        f.write(f"{len(tets)}\n")
        for t in tets:
            f.write(f"{t[0]+1} {t[1]+1} {t[2]+1} {t[3]+1} 1\n")
        f.write("End\n")


def read_medit_volume(path):
    with open(path) as f:
        lines = [l.strip() for l in f.readlines()]
    points = tets = None
    i = 0
    while i < len(lines):
        line = lines[i]
        if line == "Vertices":
            n = int(lines[i + 1])
            data = np.array([list(map(float, lines[i + 2 + k].split())) for k in range(n)])
            points = data[:, :3]
            i += 2 + n
        elif line == "Tetrahedra":
            n = int(lines[i + 1])
            data = np.array([list(map(int, lines[i + 2 + k].split())) for k in range(n)])
            tets = data[:, :4].astype(np.int64) - 1
            i += 2 + n
        else:
            i += 1
    return points, tets


def quality_report(grid, label):
    q = grid.cell_quality(quality_measure="radius_ratio")
    v = q.cell_data["radius_ratio"]
    print(f"{label}: n_cells={grid.n_cells}  radius_ratio min={v.min():.4f} "
          f"median={np.median(v):.4f} mean={v.mean():.2f}  n>100={np.sum(v>100)}  n>1e6={np.sum(v>1e6)}")


def region_grow(surf, sharp_edges_set):
    faces = surf.faces.reshape(-1, 4)[:, 1:]
    n_faces = len(faces)
    edge_to_faces = {}
    for fi, f in enumerate(faces):
        for a, b in zip(f, np.roll(f, -1)):
            key = (a, b) if a < b else (b, a)
            edge_to_faces.setdefault(key, []).append(fi)
    rows, cols, n_open = [], [], 0
    for key, fl in edge_to_faces.items():
        if len(fl) == 1:
            n_open += 1
            continue
        if key in sharp_edges_set or len(fl) != 2:
            continue
        rows += [fl[0], fl[1]]
        cols += [fl[1], fl[0]]
    adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_faces, n_faces))
    n_comp, labels = connected_components(adj, directed=False)
    return n_comp, labels, faces, n_open


def classify_and_write(grid, out_dir, prefix, feature_angle, tv_reference):
    surf = grid.extract_surface(algorithm="dataset_surface")
    sharp = surf.extract_feature_edges(feature_angle=feature_angle, boundary_edges=False,
                                        manifold_edges=False, non_manifold_edges=False)
    tree = cKDTree(surf.points)
    sharp_lines = sharp.lines.reshape(-1, 3)[:, 1:]
    sharp_coords = sharp.points[sharp_lines]
    _, idx_a = tree.query(sharp_coords[:, 0])
    _, idx_b = tree.query(sharp_coords[:, 1])
    sharp_edges_set = set((a, b) if a < b else (b, a) for a, b in zip(idx_a, idx_b))

    n_comp, labels, faces, n_open = region_grow(surf, sharp_edges_set)
    print(f"boundary: {surf.n_points} pts, {surf.n_cells} faces, "
          f"{n_comp} region(s), {n_open} open edges, {len(sharp_edges_set)} sharp edges")

    bboxes = {}
    for r in range(n_comp):
        verts = np.unique(faces[labels == r])
        pts = surf.points[verts]
        bboxes[r] = pts.max(axis=0) - pts.min(axis=0)
        print(f"  region {r}: {(labels==r).sum()} faces, bbox={bboxes[r].round(2)}")

    if n_comp < 3:
        print(f"  WARNING: only {n_comp} region(s) found, expected at least base+epi+endo. "
              f"Inspect before trusting the output.")

    base_region = min(bboxes, key=lambda r: bboxes[r][0])
    epi_region = max(bboxes, key=lambda r: np.prod(bboxes[r]))
    remaining = [r for r in bboxes if r not in (base_region, epi_region)]

    masks = {f"{prefix}_base": labels == base_region, f"{prefix}_epi": labels == epi_region}

    if len(remaining) == 2 and tv_reference is not None:
        tv_src = pv.read(tv_reference)
        tv_tree = cKDTree(tv_src.points)
        for r in remaining:
            verts = np.unique(faces[labels == r])
            pts = surf.points[verts]
            _, idx = tv_tree.query(pts)
            frac_lv = np.round(tv_src.point_data["tv"][idx]).mean()
            name = f"{prefix}_endo_lv" if frac_lv > 0.5 else f"{prefix}_endo_rv"
            print(f"  region {r}: {frac_lv*100:.1f}% tv==LV -> classified as {name}")
            masks[name] = labels == r
    elif len(remaining) == 2:
        remaining_sorted = sorted(remaining, key=lambda r: np.prod(bboxes[r]), reverse=True)
        print("  2 remaining regions, no --tv-reference given -- labeling by size "
          "(larger=endo_lv, smaller=endo_rv). Verify this is right for your anatomy, "
          "or re-run with --tv-reference for a reliable check.")
        masks[f"{prefix}_endo_lv"] = labels == remaining_sorted[0]
        masks[f"{prefix}_endo_rv"] = labels == remaining_sorted[1]
    else:
        print(f"  {len(remaining)} remaining region(s) besides base/epi -- writing as "
              f"{prefix}_endo_0, {prefix}_endo_1, ... for manual inspection.")
        for i, r in enumerate(remaining):
            masks[f"{prefix}_endo_{i}"] = labels == r

    for name, mask in masks.items():
        quads = faces[mask]
        f2 = np.hstack([np.full((len(quads), 1), 3), quads]).astype(np.int64).flatten()
        poly = pv.PolyData(surf.points, f2)
        used = np.unique(poly.faces.reshape(-1, 4)[:, 1:])
        remap = -np.ones(poly.n_points, dtype=np.int64)
        remap[used] = np.arange(len(used))
        sub_points = poly.points[used]
        sub_faces = remap[poly.faces.reshape(-1, 4)[:, 1:]]
        f3 = np.hstack([np.full((len(sub_faces), 1), 3), sub_faces]).astype(np.int64).flatten()
        out = pv.PolyData(sub_points, f3)
        out_path = out_dir / f"{name}.ply"
        out.save(str(out_path), binary=False)
        print(f"  wrote {out_path.name}: {out.n_points} points, {out.n_cells} cells")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--volume", required=True, help="Input tet volume .vtu")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--out-prefix", default="mesh", help="Filename prefix for outputs (default: mesh)")
    p.add_argument("--out-volume-name", default=None,
                   help="Output volume filename (default: <out-prefix>_volume.vtu)")
    p.add_argument("--mmg3d-bin", required=True, help="Path to the mmg3d_O3 executable")
    p.add_argument("--hausd", type=float, default=0.05,
                   help="mmg3d Hausdorff distance tolerance (mm) -- how far the optimized surface "
                        "is allowed to deviate from the input; smaller keeps geometry closer to "
                        "the original but limits how much bad triangles can be fixed")
    p.add_argument("--feature-angle", type=float, default=40.0,
                   help="Dihedral angle (deg) threshold for post-optimization region segmentation")
    p.add_argument("--tv-reference", default=None,
                   help="Optional: a Cobiveco/CobivecoX result.vtu (with a 'tv' point-data field) "
                        "for the same anatomy, used to reliably disambiguate LV vs RV if exactly "
                        "2 endocardial regions are found")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    volume_stem = Path(args.volume).parent

    grid = pv.read(args.volume)
    quality_report(grid, "BEFORE")

    tets = grid.cells_dict[pv.CellType.TETRA]
    in_mesh = volume_stem / "_mmg3d_in.mesh"
    out_mesh = volume_stem / "_mmg3d_out.mesh"
    write_medit_volume(in_mesh, grid.points, tets)

    print("\nRunning mmg3d -optim (surface allowed to adapt)...")
    result = subprocess.run(
        [args.mmg3d_bin, str(in_mesh), str(out_mesh), "-optim", "-hausd", str(args.hausd)],
        capture_output=True, text=True,
    )
    print(result.stdout[-2000:])
    if result.returncode != 0:
        print(result.stderr[-3000:])
        raise RuntimeError("mmg3d failed")

    points, tets2 = read_medit_volume(out_mesh)
    cells = np.hstack([np.full((len(tets2), 1), 4), tets2]).astype(np.int64).flatten()
    cell_types = np.full(len(tets2), pv.CellType.TETRA)
    improved = pv.UnstructuredGrid(cells, cell_types, points)
    quality_report(improved, "AFTER")

    out_vol_name = args.out_volume_name or f"{args.out_prefix}_volume.vtu"
    out_vol_path = out_dir / out_vol_name
    improved.save(str(out_vol_path))
    print(f"\nwrote {out_vol_path}: {improved.n_points} points, {improved.n_cells} cells")

    print("\nRe-deriving labeled patches from the new boundary...")
    classify_and_write(improved, out_dir, args.out_prefix, args.feature_angle, args.tv_reference)

    in_mesh.unlink(missing_ok=True)
    out_mesh.unlink(missing_ok=True)
    Path(str(out_mesh) + ".sol").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
