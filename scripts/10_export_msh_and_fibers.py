#!/usr/bin/env python3
"""
Step 7: Final export for the solver -- converts a tet mesh with per-point
fiber/sheet/normal into:

  - a Gmsh MSH 4.1 file (one physical volume group, one discrete entity,
    $Nodes, $Elements of 4-node tets)
  - a matching per-element fiber text file: first line = element count,
    then one line per element: "elem_id fx fy fz sx sy sz nx ny nz"

Element order is guaranteed identical between the two output files, so
element N in the .msh is described by line N (after the header) in the
fiber file.

Fiber/sheet/normal are expected as per-POINT data; the solver needs
per-ELEMENT values, so each tet's fiber/sheet are the average of its 4
corner nodes' vectors, re-normalized (fiber), re-orthogonalized +
re-normalized (sheet), with normal recomputed as fiber x sheet. The
average is sign-aware (fibers are axes, so f and -f are made to agree
before averaging).

--fibers-only writes just the fiber file and leaves an existing .msh
alone. Use it when only the fiber field changed (e.g. a new template
export): element order follows the input VTU, exactly as in a full
export, so the old .msh and anything derived from it (renumbered meshes,
tag files) stay valid.

Usage:
  python 10_export_msh_and_fibers.py \\
      --input full_mesh_fibers_complete.vtu \\
      --out-msh final_mesh.msh \\
      --out-fibers final_fibers.txt \\
      --physical-name interior

  python 10_export_msh_and_fibers.py --fibers-only \\
      --input full_mesh_fibers_complete.vtu \\
      --out-fibers final_fibers.txt
"""

import argparse
import sys
import numpy as np
import pyvista as pv
import gmsh


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Tet mesh with point_data fiber/sheet (.vtu)")
    p.add_argument("--out-msh", help="Output .msh (required unless --fibers-only)")
    p.add_argument("--out-fibers", required=True)
    p.add_argument("--physical-name", default="interior", help="Name for the single volume physical group")
    p.add_argument("--fibers-only", action="store_true",
                   help="Write only the fiber file; do not write a .msh")
    args = p.parse_args()
    if not args.fibers_only and not args.out_msh:
        p.error("--out-msh is required unless --fibers-only is given")

    print("Loading mesh with fibers...")
    mesh = pv.read(args.input)
    print(f"  {mesh.n_points} points, {mesh.n_cells} cells, point_data: {list(mesh.point_data.keys())}")

    if "fiber" not in mesh.point_data or "sheet" not in mesh.point_data:
        raise SystemExit("Input mesh must have point_data 'fiber' and 'sheet'.")

    celltypes = np.unique(mesh.celltypes)
    print("  cell types:", celltypes)
    if 10 not in celltypes:
        raise RuntimeError("No tetrahedral cells found.")
    if len(celltypes) != 1:
        print(f"  WARNING: non-tet cell types present ({celltypes}), extracting tets only")

    tet_mesh = mesh.extract_cells(mesh.celltypes == 10)
    points = np.array(tet_mesh.points)
    cells = tet_mesh.cells.reshape(-1, 5)
    tets = cells[:, 1:5]
    print(f"  extracted: {len(points)} nodes, {len(tets)} tets")

    fiber_pt = tet_mesh.point_data["fiber"]
    sheet_pt = tet_mesh.point_data["sheet"]

    print("Averaging node fiber/sheet onto elements...")
    # Fibers are axes (f and -f are the same fiber), and mapped fields flip
    # sign between neighbouring nodes in places, so each corner is first
    # flipped to agree with corner 0; a plain mean of f and -f would cancel.
    fc = np.asarray(fiber_pt, dtype=np.float64)[tets]
    sc = np.asarray(sheet_pt, dtype=np.float64)[tets]
    sgn_f = np.sign(np.einsum("eci,ei->ec", fc, fc[:, 0]) + 1e-30)
    sgn_s = np.sign(np.einsum("eci,ei->ec", sc, sc[:, 0]) + 1e-30)
    print(f"  corner vectors sign-flipped before averaging: fiber {100*(sgn_f < 0).mean():.2f}%, "
          f"sheet {100*(sgn_s < 0).mean():.2f}%")
    f = (fc * sgn_f[:, :, None]).mean(axis=1)
    s = (sc * sgn_s[:, :, None]).mean(axis=1)
    del fc, sc
    f = f / np.linalg.norm(f, axis=1, keepdims=True)
    s = s - np.sum(s * f, axis=1, keepdims=True) * f
    s = s / np.linalg.norm(s, axis=1, keepdims=True)
    n = np.cross(f, s)

    bad = np.isnan(f).any(axis=1) | np.isnan(s).any(axis=1)
    print(f"  degenerate elements (NaN after averaging): {bad.sum()}")
    if bad.sum() > 0:
        f[bad] = fiber_pt[tets[bad, 0]]
        s[bad] = sheet_pt[tets[bad, 0]]
        s[bad] = s[bad] - np.sum(s[bad] * f[bad], axis=1, keepdims=True) * f[bad]
        s[bad] = s[bad] / np.linalg.norm(s[bad], axis=1, keepdims=True)
        n[bad] = np.cross(f[bad], s[bad])

    print(f"Writing {args.out_fibers}...")
    with open(args.out_fibers, "w") as fh:
        fh.write(f"{len(tets)}\n")
        for i in range(len(tets)):
            fh.write(
                f"{i+1} "
                f"{f[i,0]:.6f} {f[i,1]:.6f} {f[i,2]:.6f} "
                f"{s[i,0]:.6f} {s[i,1]:.6f} {s[i,2]:.6f} "
                f"{n[i,0]:.6f} {n[i,1]:.6f} {n[i,2]:.6f}\n"
            )

    if args.fibers_only:
        print(f"\nDone (--fibers-only). {len(tets)} elements; no .msh written.")
        return

    print("Building Gmsh model...")
    gmsh.initialize(sys.argv)
    gmsh.model.add("export")

    volume = gmsh.model.addDiscreteEntity(3)
    gmsh.model.addPhysicalGroup(3, [volume], 1, args.physical_name)

    node_tags = np.arange(1, len(points) + 1)
    gmsh.model.mesh.addNodes(3, volume, node_tags.tolist(), points.flatten().tolist())

    elem_tags = np.arange(1, len(tets) + 1)
    gmsh.model.mesh.addElementsByType(volume, 4, elem_tags.tolist(), (tets + 1).flatten().tolist())

    gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
    gmsh.option.setNumber("Mesh.SaveParametric", 0)
    gmsh.option.setNumber("Mesh.SaveAll", 1)

    print(f"Writing {args.out_msh}...")
    gmsh.write(args.out_msh)
    gmsh.finalize()

    print(f"\nDone. {len(points)} nodes, {len(tets)} elements in both files.")


if __name__ == "__main__":
    main()
