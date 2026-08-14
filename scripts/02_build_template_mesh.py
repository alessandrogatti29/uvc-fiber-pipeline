#!/usr/bin/env python3
"""
Step 2a: Build a solid tetrahedral template mesh (with fiber/sheet/normal
carried as point data) from a Continuity structured-grid fiber export, and
classify its boundary into endo_LV / endo_RV / epi / base surfaces ready
for Cobiveco.

Assumes the export samples each element on a regular structured grid
(xi0 x xi1 x xi2), which is Continuity's convention for this kind of file.
Reconstructs real hexahedral geometry from that grid (rather than treating
the rows as a bare point cloud), welds points shared between neighbouring
elements, and segments the resulting closed surface into anatomical
regions by cutting along sharp (dihedral-angle) creases.

Apex handling (only relevant if your template has a coordinate-singularity
pole, e.g. a prolate-spheroidal-style parametrization where several
elements converge to a point): elements whose xi1=0 face collapses to a
near-point get isolated as their own tiny, topologically awkward region by
the automatic segmentation. If that happens, use --apex-inner-elements /
--apex-outer-elements to say which elements are the endocardial-side and
epicardial-side wedges converging there (found by inspecting where the
segmentation goes wrong -- see the README for how we identified these for
our own template), and the two sides get capped independently so endo and
epi close smoothly without touching each other. If your template has no
such singularity, pass --no-apex-fix.

Usage:
  python 02_build_template_mesh.py \\
      --fibers fibers_all_fixed.txt \\
      --out-dir template_geometry \\
      --grid-shape 10 50 50 \\
      --num-elements 128 \\
      --apex-inner-elements 49 65 \\
      --apex-outer-elements 113 129
"""

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def load(fibers_path):
    return np.loadtxt(fibers_path, skiprows=1)


def build_hex_mesh(data, grid_shape, num_elements):
    n0, n1, n2 = grid_shape
    ppe = n0 * n1 * n2

    all_hexes = []
    all_elem_ids = []
    raw_points, raw_fiber, raw_sheet, raw_normal = [], [], [], []
    global_offset = 0

    for e in range(num_elements):
        block = data[e * ppe:(e + 1) * ppe]
        X = block[:, 7:10].reshape(n0, n1, n2, 3)
        dY = block[:, 16:25].reshape(n0, n1, n2, 3, 3)
        idx = np.arange(n0 * n1 * n2).reshape(n0, n1, n2) + global_offset

        raw_points.append(X.reshape(-1, 3))
        raw_fiber.append(dY[..., 0, :].reshape(-1, 3))
        raw_sheet.append(dY[..., 1, :].reshape(-1, 3))
        raw_normal.append(dY[..., 2, :].reshape(-1, 3))
        global_offset += n0 * n1 * n2

        for i in range(n0 - 1):
            for j in range(n1 - 1):
                for k in range(n2 - 1):
                    all_hexes.append([
                        idx[i, j, k], idx[i + 1, j, k], idx[i + 1, j + 1, k], idx[i, j + 1, k],
                        idx[i, j, k + 1], idx[i + 1, j, k + 1], idx[i + 1, j + 1, k + 1], idx[i, j + 1, k + 1],
                    ])
                    all_elem_ids.append(e + 1)
        print(f"  element {e + 1}/{num_elements} done", end="\r")
    print()

    raw_points = np.vstack(raw_points)
    raw_fiber = np.vstack(raw_fiber)
    raw_sheet = np.vstack(raw_sheet)
    raw_normal = np.vstack(raw_normal)

    rounded = np.round(raw_points, 5)
    unique_pts, first_idx, inverse = np.unique(rounded, axis=0, return_index=True, return_inverse=True)
    fiber = raw_fiber[first_idx]
    sheet = raw_sheet[first_idx]
    normal = raw_normal[first_idx]
    hexes = inverse[np.array(all_hexes, dtype=np.int64)]
    return unique_pts, hexes, np.array(all_elem_ids), fiber, sheet, normal


def region_grow(surf, sharp_edges_set):
    faces = surf.faces.reshape(-1, 5)[:, 1:]
    n_faces = len(faces)
    edge_to_faces = {}
    for fi, f in enumerate(faces):
        for a, b in zip(f, np.roll(f, -1)):
            key = (a, b) if a < b else (b, a)
            edge_to_faces.setdefault(key, []).append(fi)
    rows, cols = [], []
    for key, fl in edge_to_faces.items():
        if key in sharp_edges_set or len(fl) != 2:
            continue
        rows += [fl[0], fl[1]]
        cols += [fl[1], fl[0]]
    adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_faces, n_faces))
    n_comp, labels = connected_components(adj, directed=False)
    return n_comp, labels, faces


def cap_hole_near_point(poly, hint_point):
    """Closes the boundary loop of `poly` closest to hint_point with a
    triangle fan to a new centroid point. Other boundary loops (e.g. the
    base-side opening) are left untouched."""
    edges = poly.extract_feature_edges(boundary_edges=True, feature_edges=False,
                                        manifold_edges=False, non_manifold_edges=False)
    if edges.n_cells == 0:
        return poly

    lines = edges.lines.reshape(-1, 3)[:, 1:]
    n_pts = edges.n_points
    adj = coo_matrix((np.ones(len(lines)), (lines[:, 0], lines[:, 1])), shape=(n_pts, n_pts))
    adj = adj + adj.T
    n_comp, comp_labels = connected_components(adj, directed=False)

    best_comp, best_dist = None, None
    for c in range(n_comp):
        idx = np.where(comp_labels == c)[0]
        centroid = edges.points[idx].mean(axis=0)
        d = np.linalg.norm(centroid - hint_point)
        if best_dist is None or d < best_dist:
            best_dist, best_comp = d, c

    loop_local_idx = np.where(comp_labels == best_comp)[0]
    comp_lines = lines[np.isin(lines[:, 0], loop_local_idx) & np.isin(lines[:, 1], loop_local_idx)]

    adj_map = {}
    for a, b in comp_lines:
        adj_map.setdefault(a, []).append(b)
        adj_map.setdefault(b, []).append(a)

    start = int(comp_lines[0, 0])
    order = [start]
    prev, cur = None, start
    while True:
        nbrs = [x for x in adj_map[cur] if x != prev]
        if not nbrs:
            break
        nxt = nbrs[0]
        if nxt == start:
            break
        order.append(nxt)
        prev, cur = cur, nxt

    loop_coords = edges.points[order]
    tree = cKDTree(poly.points)
    _, orig_idx = tree.query(loop_coords)

    centroid = loop_coords.mean(axis=0)
    new_idx = poly.n_points
    new_points = np.vstack([poly.points, centroid])

    fan = []
    n = len(orig_idx)
    for i in range(n):
        a, b = int(orig_idx[i]), int(orig_idx[(i + 1) % n])
        fan.append([3, a, b, new_idx])

    orig_faces = poly.faces.reshape(-1, 4)
    all_faces = np.vstack([orig_faces, np.array(fan, dtype=orig_faces.dtype)])
    capped = pv.PolyData(new_points, all_faces.flatten())
    capped = capped.compute_normals(consistent_normals=True, auto_orient_normals=True,
                                     non_manifold_traversal=False)
    print(f"    capped loop of {n} points near {hint_point}, dist={best_dist:.3f}")
    return capped


def write_ply(points, quads, out_path, cap_hint=None):
    faces = np.hstack([np.full((len(quads), 1), 4), quads]).astype(np.int64).flatten()
    poly = pv.PolyData(points, faces)
    poly = poly.triangulate()
    if cap_hint is not None:
        poly = cap_hole_near_point(poly, np.array(cap_hint))
    used = np.unique(poly.faces.reshape(-1, 4)[:, 1:])
    remap = -np.ones(poly.n_points, dtype=np.int64)
    remap[used] = np.arange(len(used))
    sub_points = poly.points[used]
    sub_faces = remap[poly.faces.reshape(-1, 4)[:, 1:]]
    faces2 = np.hstack([np.full((len(sub_faces), 1), 3), sub_faces]).astype(np.int64).flatten()
    out = pv.PolyData(sub_points, faces2)
    out.save(str(out_path), binary=False)
    print(f"  wrote {out_path.name}: {out.n_points} points, {out.n_cells} cells")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fibers", required=True, help="Handedness-corrected fiber export (output of step 1)")
    p.add_argument("--out-dir", required=True, help="Output directory for the template mesh + surfaces")
    p.add_argument("--out-prefix", default="template", help="Filename prefix for outputs (default: template)")
    p.add_argument("--grid-shape", type=int, nargs=3, default=[10, 50, 50],
                   metavar=("N0", "N1", "N2"), help="Per-element structured sample grid shape")
    p.add_argument("--num-elements", type=int, default=128)
    p.add_argument("--feature-angle", type=float, default=40.0,
                   help="Dihedral angle (deg) threshold for sharp-edge region segmentation")
    p.add_argument("--apex-inner-elements", type=int, nargs=2, metavar=("START", "END_EXCLUSIVE"),
                   help="Element id range (Python-style, end-exclusive) converging at the endocardial "
                        "side of an apex pole singularity, e.g. 49 65. Omit with --no-apex-fix.")
    p.add_argument("--apex-outer-elements", type=int, nargs=2, metavar=("START", "END_EXCLUSIVE"),
                   help="Same as above, for the epicardial side, e.g. 113 129.")
    p.add_argument("--no-apex-fix", action="store_true",
                   help="Skip apex-pole handling entirely (use if your template has no such singularity)")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    apex_inner = set(range(*args.apex_inner_elements)) if args.apex_inner_elements else set()
    apex_outer = set(range(*args.apex_outer_elements)) if args.apex_outer_elements else set()
    if not args.no_apex_fix and not (apex_inner and apex_outer):
        print("NOTE: no --apex-inner-elements/--apex-outer-elements given and --no-apex-fix not set. "
              "If your template has an apex-pole singularity, the automatic segmentation below may "
              "isolate a small spurious region there; inspect the output region list and re-run with "
              "those two element ranges if so.")

    print("Loading fiber file...")
    data = load(args.fibers)

    print(f"Building hex mesh ({args.num_elements} elements, grid {args.grid_shape})...")
    points, hexes, elem_ids, fiber, sheet, normal = build_hex_mesh(data, args.grid_shape, args.num_elements)
    print(f"  {len(points)} welded points, {len(hexes)} hexahedra")

    cells = np.hstack([np.full((len(hexes), 1), 8), hexes]).astype(np.int64).flatten()
    cell_types = np.full(len(hexes), pv.CellType.HEXAHEDRON)
    grid = pv.UnstructuredGrid(cells, cell_types, points)
    grid.cell_data["elem"] = elem_ids
    grid.point_data["fiber"] = fiber
    grid.point_data["sheet"] = sheet
    grid.point_data["normal"] = normal

    print("Extracting exterior surface...")
    surf = grid.extract_surface(algorithm="dataset_surface", pass_cellid=True)
    surf_elem = elem_ids[surf.cell_data["vtkOriginalCellIds"]]
    print(f"  surface: {surf.n_points} points, {surf.n_cells} faces")

    print(f"Detecting sharp feature edges (>{args.feature_angle} deg)...")
    sharp = surf.extract_feature_edges(feature_angle=args.feature_angle, boundary_edges=False,
                                        manifold_edges=False, non_manifold_edges=False)
    tree = cKDTree(surf.points)
    sharp_lines = sharp.lines.reshape(-1, 3)[:, 1:]
    sharp_coords = sharp.points[sharp_lines]
    _, idx_a = tree.query(sharp_coords[:, 0])
    _, idx_b = tree.query(sharp_coords[:, 1])
    sharp_edges_set = set((a, b) if a < b else (b, a) for a, b in zip(idx_a, idx_b))
    print(f"  sharp edges: {len(sharp_edges_set)}")

    print("Segmenting surface into regions...")
    n_comp, labels, faces = region_grow(surf, sharp_edges_set)
    print(f"  {n_comp} region(s) found")

    region_elems = {}
    bboxes = {}
    for r in range(n_comp):
        m = labels == r
        els = sorted(set(surf_elem[m].tolist()))
        region_elems[r] = els
        verts = np.unique(faces[m])
        pts = surf.points[verts]
        bboxes[r] = pts.max(axis=0) - pts.min(axis=0)
        print(f"  region {r}: {m.sum()} faces, {len(els)} elements, bbox={bboxes[r].round(2)}")

    apex_hint_lv = apex_hint_epi = None
    if apex_inner and apex_outer:
        apex_region = None
        for r, els in region_elems.items():
            if set(els) & apex_inner and set(els) & apex_outer:
                apex_region = r
                break
        if apex_region is not None:
            apex_mask = labels == apex_region
            apex_verts = np.unique(faces[apex_mask])
            apex_hint = surf.points[apex_verts].mean(axis=0)
            apex_hint_lv = apex_hint_epi = apex_hint
            del bboxes[apex_region]
            print(f"  apex region identified (region {apex_region}), will discard its "
                  f"{apex_mask.sum()} degenerate faces and cap endo_LV/epi independently instead")
        else:
            print("  WARNING: --apex-inner/outer-elements given but no region contains both; "
                  "check the element ranges against the region list above")

    base_region = min(bboxes, key=lambda r: bboxes[r][0])
    epi_region = max(bboxes, key=lambda r: np.prod(bboxes[r]))
    remaining = [r for r in bboxes if r not in (base_region, epi_region)]
    remaining_sorted = sorted(remaining, key=lambda r: np.prod(bboxes[r]), reverse=True)
    lv_region = remaining_sorted[0]
    rv_regions = remaining_sorted[1:]
    print(f"\nFinal mapping: base=region{base_region}, epi=region{epi_region}, "
          f"endo_LV=region{lv_region}, endo_RV=regions{rv_regions}")

    lv_mask = labels == lv_region
    epi_mask = labels == epi_region
    rv_mask = np.isin(labels, rv_regions)
    base_mask = labels == base_region

    prefix = args.out_prefix
    for name, mask, cap in [
        (f"{prefix}_endo_lv", lv_mask, apex_hint_lv),
        (f"{prefix}_endo_rv", rv_mask, None),
        (f"{prefix}_epi", epi_mask, apex_hint_epi),
        (f"{prefix}_base", base_mask, None),
    ]:
        quads = faces[mask]
        write_ply(surf.points, quads, out_dir / f"{name}.ply", cap_hint=cap)

    print("\nConverting volume to tetrahedra...")
    tet_grid = grid.triangulate()
    out_vol = out_dir / f"{prefix}_volume.vtu"
    tet_grid.save(str(out_vol))
    print(f"  wrote {out_vol.name}: {tet_grid.n_points} points, {tet_grid.n_cells} cells")

    print(f"\nDone. Template mesh + surfaces written to {out_dir}")
    print("Next: run Cobiveco (plain, not CobivecoX -- there are no valve landmarks here) "
          "on this endo_lv/endo_rv/epi/base + volume set, then step 3.")


if __name__ == "__main__":
    main()
