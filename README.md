# UVC-based fiber orientation pipeline

Assigns myocardial fiber orientation to a patient-specific biventricular
mesh (including the valve regions) by transferring fibers from a generic
template through [Cobiveco](https://github.com/KIT-IBT/Cobiveco) Universal
Ventricular Coordinates (UVC), then filling in the valve-adjacent region
Cobiveco can't reach with a simple rule-based direction.

## Why this pipeline exists

A generic fiber template (e.g. a Continuity-fitted cubic-Hermite
biventricular model) usually has **no valves** — it's a simple truncated
ventricular shape. A real patient mesh usually **has valves** — actual
openings for the mitral/aortic/tricuspid/pulmonary annuli. That mismatch
means you can't directly compare UVC coordinates between the two: Cobiveco
computes different `ab`/`rt` fields depending on whether valve landmarks
are available, and the template can't tell you anything about a region it
doesn't model in the first place.

This pipeline resolves that with a two-hop transfer instead of a naive
one-shot mapping:

```
template mesh  --Cobiveco-->  template UVC
                                   |
                                   |  UVC-based mapping (different shapes,
                                   v   only comparable via UVC)
patient mesh, valves cut off  --Cobiveco-->  cut-mesh UVC  --> fibers on cut mesh
                                   |
                                   |  Cartesian mapping (same anatomy,
                                   v   just two different meshes of it)
patient mesh, WITH valves  <---------------  fibers transferred back
                                   |
                                   |  rule-based fill only for the
                                   v   valve-adjacent gap left uncovered
patient mesh, WITH valves, fibers everywhere
```

The key idea at each hop: use UVC mapping only where geometries are
genuinely different shapes (template vs. patient), and use plain spatial
interpolation where they're the same real anatomy just discretized twice
(the cut mesh and the original full mesh) — spatial interpolation is
simpler and more accurate there, and doesn't need Cobiveco to have been
run with matching landmark availability.

## Pipeline steps

| # | Script | What it does |
|---|---|---|
| 1 | `01_fix_fiber_handedness.py` | Fixes left-handed fiber frames in the raw template fiber export |
| 2a | `02_build_template_mesh.py` | Builds a solid template mesh from the fiber export, classifies endo_LV/endo_RV/epi/base |
| 2b | `03_optimize_mesh_quality.py` | Fixes tet element quality (reused again in step 3) |
| 2c | `04_attach_fibers_to_uvc_result.py` | Re-attaches fiber data onto the template's Cobiveco output |
| 3a | *(your own cut, e.g. in ParaView)* | Remove the valve/atrial region from your patient mesh |
| 3b | `05_repair_surface_defects.py` | *(optional)* Fixes near-duplicate-point defects from the cut |
| 3c | `06_tetrahedralize_surface.py` | Builds a clean tet volume from the cut surfaces |
| 3d | `03_optimize_mesh_quality.py` | Fixes tet element quality on the cut mesh |
| 4 | `map_fibers_uvc.m` | UVC-based fiber transfer: template → cut patient mesh |
| 5 | `transfer_fibers_cartesian.m` | Cartesian transfer: cut mesh → full patient mesh (with valves) |
| 6 | `09_rule_based_ring_fibers.py` | Rule-based fiber fill for the valve-adjacent gap |
| 7 | `10_export_msh_and_fibers.py` | Final export: Gmsh MSH 4.1 + per-element fiber `.txt` |

Cobiveco itself runs **outside** this repo — steps 2 and 3 each end with
"now run Cobiveco/CobivecoX on this" as a manual step in MATLAB, using
your own Cobiveco installation. Step 2 uses plain Cobiveco (no valve
landmarks, since the template has none); step 3 typically uses CobivecoX
(with valve landmarks) if you want the fullest-quality UVC on the cut
mesh — either works for the mapping in step 4, as long as it produces a
`result.vtu` with `ab/rt/tm/tv` and an `R.mat` alignment matrix.

## Prerequisites

- Python 3 with `numpy`, `scipy`, `pyvista`, `tetgen`, `gmsh`
- MATLAB, with a working [Cobiveco](https://github.com/KIT-IBT/Cobiveco)
  installation (its `dependencies/` must be built — see Cobiveco's own
  `dependencies/install_cobiveco.sh`)
- The `mmg3d_O3` binary from Cobiveco's `dependencies/mmg/build/bin/`
  (used directly by `03_optimize_mesh_quality.py`)

## Example end-to-end run

```bash
# 1. Fix fiber handedness
python 01_fix_fiber_handedness.py \
    --input fibers_all.txt --output fibers_all_fixed.txt

# 2a. Build template geometry (apex-element ranges are specific to our
#     template -- see "Adapting to a different template" below)
python 02_build_template_mesh.py \
    --fibers fibers_all_fixed.txt --out-dir template_geometry \
    --apex-inner-elements 49 65 --apex-outer-elements 113 129

# 2b. Fix its element quality
python 03_optimize_mesh_quality.py \
    --volume template_geometry/template_volume.vtu \
    --out-dir template_geometry --out-prefix template \
    --mmg3d-bin /path/to/Cobiveco/dependencies/mmg/build/bin/mmg3d_O3

# --- run Cobiveco (plain) on template_geometry/{template_endo_lv,endo_rv,epi,base}.ply
#     + template_geometry/template_volume.vtu in MATLAB, producing result_template/

# 2c. Re-attach fiber data (lost during quality optimization)
python 04_attach_fibers_to_uvc_result.py \
    --uvc-result result_template/result.vtu --fibers fibers_all_fixed.txt \
    --output template_geometry/template_with_fibers.vtu

# 3a. Cut the valve region off your patient mesh yourself (e.g. ParaView Clip),
#     producing cut_endo_lv.ply, cut_endo_rv.ply, cut_epi.ply, cut_base.ply

# 3b. (optional, only if step 3c fails or reports bad quality clustered
#      at a specific spot -- see script docstring for how to find defect coordinates)
python 05_repair_surface_defects.py \
    --surfaces cut_endo_lv.ply cut_endo_rv.ply cut_epi.ply cut_base.ply \
    --defect-center X Y Z --output-suffix _repaired

# 3c. Build the cut mesh's volume
python 06_tetrahedralize_surface.py \
    --merged-surface merged_repaired.vtp --output cut_volume.vtu
    # (or --surfaces cut_endo_lv.ply ... directly, if step 3b wasn't needed)

# 3d. Fix its element quality (tv-reference lets it reliably tell LV from RV)
python 03_optimize_mesh_quality.py \
    --volume cut_volume.vtu --out-dir cut_geometry --out-prefix cut \
    --mmg3d-bin /path/to/mmg3d_O3 --tv-reference result_patient_full/result.vtu

# --- run Cobiveco/CobivecoX on cut_geometry/{cut_endo_lv,endo_rv,epi,base}.ply
#     + the optimized cut volume in MATLAB, producing result_cut/

# 4. UVC-based fiber mapping: template -> cut mesh   (in MATLAB)
map_fibers_uvc('template_geometry/template_with_fibers.vtu', 'result_cut/result.vtu', ...
    'result_template/R.mat', 'result_cut/R.mat', 'cut_with_fibers.vtu')

# 5. Cartesian transfer: cut mesh -> full patient mesh with valves   (in MATLAB)
transfer_fibers_cartesian('cut_with_fibers.vtu', 'patient_full_with_valves.vtu', ...
    'patient_fibers_partial.vtu')

# 6. Rule-based fill for the valve-adjacent gap
python 09_rule_based_ring_fibers.py \
    --input patient_fibers_partial.vtu \
    --rings patient_av.ply patient_mv.ply patient_pv.ply patient_tv.ply \
    --transition-distance 15.0 --output patient_fibers_complete.vtu

# 7. Final export for the solver
python 10_export_msh_and_fibers.py \
    --input patient_fibers_complete.vtu \
    --out-msh patient_final.msh --out-fibers patient_final_fibers.txt
```

## Adapting to a different template or patient

A few things are specific to the particular files we developed this
against, and you'll need to re-determine them for a different template or
patient:

- **Apex singularity elements** (`--apex-inner-elements`/
  `--apex-outer-elements` in step 2a): only relevant if your template uses
  a coordinate parametrization with a pole singularity (e.g.
  prolate-spheroidal-style), where several elements converge to a point.
  Run step 2a once without these flags; if the printed region list shows
  a small, oddly-shaped extra region, that's the pole — its element IDs
  are what to pass in on the next run. If your template doesn't have this
  issue, pass `--no-apex-fix`.
- **Surface defect coordinates** (`--defect-center` in step 3b): only
  needed if step 3c fails or reports severely degenerate elements
  clustered in one place. The error/quality report tells you roughly
  where; open the merged surface in ParaView to confirm.
- **Rule-based transition distance** (`--transition-distance` in step 6):
  a physiologically reasonable annulus transition width; we started at
  15mm and adjusted after visually checking the output's `ruleWeight`
  field in ParaView (color by it, glyph by `fiber`, look at whether the
  blend looks smooth around each ring).
- **LV/RV disambiguation** (`--tv-reference` in step 3d): needs *any*
  existing Cobiveco/CobivecoX result for the same patient with a `tv`
  field — usually you'll already have this from a prior full-mesh
  (valve-containing) Cobiveco run.

## A faster shortcut for a second patient

Once you have one patient's mesh with complete fibers everywhere
(`patient_fibers_complete.vtu`, step 6's output), you can skip most of
the pipeline for the *next* patient: run CobivecoX on that
already-complete mesh to get its UVC, run CobivecoX on the new patient's
mesh the normal way, and map fibers between them directly via UVC (step
4's script, `map_fibers_uvc.m`) in one shot — no cutting, no separate
rule-based valve step, since the valve regions transfer through the UVC
correspondence automatically. The valve-region fibers you get this way
are inherited from the *first* patient's rule-based approximation rather
than freshly derived for the new patient's specific valve geometry, which
is a reasonable simplification but worth knowing about.

## Validating each step

Every script prints numeric sanity checks as it runs (point/element
counts, quality distributions, unmapped-point counts, orthonormality of
the fiber/sheet/normal frames) — read them; a script completing without
an error doesn't by itself mean the geometry is right. Things worth
specifically checking in ParaView along the way:

- After step 2/3 (mesh building): merge all 4 boundary patches and check
  for open/non-manifold edges (should be zero) before tetrahedralizing.
- After step 3 quality optimization: the automatic region re-labeling can
  occasionally split a tiny spurious extra region off; if the printed
  region count isn't what you expect, inspect before trusting the labels.
- After step 6: Glyph filter on `fiber`, colored by `ruleWeight`, to
  confirm the blend near each ring looks smooth (see script docstring and
  the "adapting" section above for tuning `--transition-distance`).
