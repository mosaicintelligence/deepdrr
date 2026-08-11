# Mosaic's changes to DeepDRR

This checkout is a fork of DeepDRR used by [`mosaic`](https://github.com/) `sim/fluoro_sim`.
This file is the record of **every** local change: what it is, why, and the measurement that
justifies it.

**Audience:** the human developer maintaining this fork, and any AI agent editing it.

**Why this file exists.** Each item is written to be **separately contributable upstream** —
one change, one rationale, one piece of evidence, no dependence on the others. Keep it that
way: if a change cannot be described here on its own, it is probably two changes.

Mosaic's own docs stay in the mosaic repo. This file does not restate them; it cites them.
The canonical design and measurement record for the contrast work is
`sim/fluoro_sim/CONTRAST_INTEGRATION.md` in that repo.

## Rules for adding an item

- One numbered section per change, in the order they landed.
- State the **measurement**, not the intent. A change with no number behind it does not go in.
- Name the mosaic-side reproducer, so the evidence can be re-run rather than trusted.
- If a change is inert until some later phase uses it, **say so** — that is a property worth
  knowing when reviewing it.

---

## 1. `project_kernel.cu`: the volume was ray-marched one voxel off every mesh

**Landed** 2026-08-10, phase P0 of mosaic's C5. **Files:**
`deepdrr/projector/project_kernel.cu`, `deepdrr/projector/projector.py`.

### The change

The per-step IJK base index subtracted `1.0f`:

```c
// we offset by -1.0f because we manually calculate the the trilinear filtering value in the surrounding area
// -0.5 offset for cuda texture and additional -0.5 offset to recenter for a standard grid
px[vol_id] = sx_ijk_local[vol_id] + alpha * rx_ijk[vol_id] - 1.0f;
```

It now adds `IJK_SAMPLE_OFFSET`, whose only correct value is `0.0f`. **Nothing else in the
kernel moved** — in particular the density fetch keeps its own `+ 0.5f`.

### Why `0` is the only correct value

Voxel `n` is centred at IJK `n`. That is what the volume's own declared bounds say
(`gVolumeEdgeMinPoint = -0.5`, `gVolumeEdgeMaxPoint = shape - 0.5`) and what
`Volume.ijk_from_world` produces. The two fetches taken from that base then need **different**
corrections, and each already applies its own:

| fetch | filter mode | correction it needs | where it comes from |
|---|---|---|---|
| density, `tex3D<float>(volume_texs[...])` | linear | texel-centre `+0.5` | applied at the call site |
| segmentation label, `tex3D<int>(seg_texs[...])` | **point** | none — `tex3D<int>` returns element `floor(coord)` | n/a; it reads `floor(q)` and `floor(q)+1` directly |

So the correct shared base is plain `q`, and the whole fix is deleting `- 1.0f`. Both the
comment's terms were wrong: the first `-0.5` duplicates the correction the density fetch
already makes, and the second is not a real correction at all.

Net effect of the shipped value: the ray reads voxel 136 where it should read 137, so the
entire volume appears one voxel further along `+i, +j, +k` than every mesh. Meshes are
rasterized by OpenGL straight from world coordinates and never touch this path, so they land
correctly and the volume does not.

### The measurement

`sim/fluoro_sim/qa/kernel_grid_alignment_report.py` in mosaic. An identical solid is rendered
as volume content and as an additive mesh at the same world coordinates; the intensity-weighted
centroids of the two shadows are compared, and pixels-per-voxel is self-calibrated by a known
mesh shift, so no camera arithmetic enters the answer.

Synthetic scene (100³ at 2 mm, a soft-tissue cube filling voxels [40, 60), AP at 384 px):

| `IJK_SAMPLE_OFFSET` | volume − mesh misalignment | predicted (`−offset`) |
|---|---|---|
| **`-1.0f` (upstream `main` and `dev`)** | **+0.996 voxels** | +1.0 |
| `-0.5f` | +0.499 | +0.5 |
| **`0.0f` (this fork)** | **+0.0001 ± 0.0007** | 0.0 |
| `+0.5f` | −0.501 | −0.5 |

On PAT_23 (0.9766 / 0.9766 / 0.625 mm) the shipped value is a **1.51 mm displacement of the
CT relative to every mesh**, ≈ 3.9 px in-plane at a 1024-px aorto-iliac framing.

### Why nobody caught it, and why it had a paper trail

A uniform shift of the whole patient is invisible in an X-ray. It only shows when something
drawn *from the volume* is compared against something drawn *as a mesh*, and no test in this
repo or in mosaic did that before the report above.

Two people reached opposite conclusions with no test between them:

| commit | date | value | what happened |
|---|---|---|---|
| *(pre-existing)* | — | `-0.5f` | Segmentation was one linear-filtered float texture per material. Already **half** a voxel off. |
| `0c21300` | 2025-04-24 | `-1.0f` | Replaced N float textures with one uint8 label texture plus a hand-written trilinear — necessary and correct, since label IDs cannot be linearly interpolated. Changed `-0.5f` → `-1.0f`, doubling the error. |
| `2e2d2e7` "WIP" | 2025-05-13 | `-0.5f` | Reverted it and left a `// TODO offset error`. |
| `ed0c5c3` "Fix Grid Offset" | 2025-05-14 | `-1.0f` | Put `-1.0f` back and deleted the TODO. |

The subtlety both missed is the filter-mode table above: `0c21300` correctly switched the
label fetch to point sampling, which *removed* that fetch's need for a texel-centre
correction, and the base index was moved in the opposite direction instead.

### Why it is a macro and not a literal `0`

`#define IJK_SAMPLE_OFFSET (0.0f)`, overridable through `_get_kernel_projector_module`'s `-D`
list and `Projector(ijk_sample_offset=...)`. The macro is **diagnostic only** and nothing in
the library sets it; it exists so the report can sweep it and *measure* that 0 is right rather
than assert it in a comment. The alternative — a QA report that monkeypatches a private
function to run the sweep — would rot the first time the signature changed.

### Blast radius, and what it does not change

Every rendered frame changes: bone and soft tissue move ~1.5 mm relative to tools and labels.
**No validated number in mosaic moves**, and that is predicted rather than hoped for: every
existing check is either mesh-against-mesh (`project_seg` is mesh-only and never touches this
kernel) or uses a **uniform** volume, in which a shift changes nothing. Mosaic's P0 checkpoint
re-baselines all of them.

---

## 2. A continuous per-voxel additive material-density channel

**Landed** 2026-08-11, phase P2 of mosaic's C5. **Files:** `deepdrr/vol/additive_density_field.py`
(new), `deepdrr/vol/__init__.py`, `deepdrr/projector/project_kernel.cu`,
`deepdrr/projector/projector.py`.

**Inert as shipped.** Nothing in this library and nothing in mosaic's render path constructs
one. `NUM_DENSITY_FIELDS` defaults to 0, and at 0 the kernel is item 1's kernel — measured
below, not assumed. The mosaic caller is phase P3.

### The change

`AdditiveDensityField` — one resident float32 `CUDAarray` of `g/cm³` of **one named
material**, linear-filtered, with its own `shape`, `spacing` and a **full**
`world_from_ijk`; sampled once per ray step inside the field's own box and **added** to that
material's `area_density[]`. Passed as `Projector(..., density_fields=[field])` and updated
on a live `Projector` with `update_contrast(rho)` — one `cudaMemcpy3D`, no re-initialization.

Three lines in the kernel do the work:

```c
float boundary = (0 == t || num_steps - 1 == t) ? 0.5f : 1.0f;   // hoisted, shared by all paths
float weight   = boundary / ((float)n_vols_at_curr_priority);     // 1/n lives HERE, only
area_density[field_material_index[f]] +=
    boundary * tex3D<float>(field_texs[f], fx + 0.5f, fy + 0.5f, fz + 0.5f);
```

| decision | reason |
|---|---|
| **`boundary`, never `weight`** | `1/n` exists because n volumes at one priority are competing descriptions of the *same* matter. A density field carries matter no volume has, so inheriting `1/n` would make an iodine bolus half as dense wherever two volumes happen to overlap — a plausible image, no error. Splitting the two factors at the source makes it structurally impossible rather than a comment |
| **`+ 0.5f`, and no offset on the base index** | The same texel-centre correction the linear volume-density fetch makes. After item 1 there is exactly **one** sampling convention in this kernel, which is why item 1 had to land first |
| **inside `if (!inside_mesh)`** | A subtractive mesh means "this matter is displaced". A catheter body displaces blood *and* the iodine dissolved in it. No mosaic mesh sets `subtractive` today, so this is currently a no-op — decided rather than left to accident |
| **compile-time `-D NUM_DENSITY_FIELDS={n}`, runtime material index** | Mirrors `MESH_ADDITIVE_ENABLED` for the count and `priority[]` for the index, so changing *which* material a field carries does not recompile. Parameters are always in the signature; only the bodies are `#if`-guarded, so there is one arg list rather than two |
| **not a `Renderable`, not in the `volume` list** | It has no priority and is not composited by the priority rule. A separate argument keeps `Projector([ct, mesh, ...])` semantics untouched and stops "just add a second `Volume`" from looking like the same thing |
| **generic material, not hard-coded iodine** | The same code renders a continuous calcium field or an air field with no further fork change. `NUM_MATERIALS` grows by one per new material (4 → 5 on mosaic's cases), not to a bin count |
| **IJK shape in, KJI array out** | `create_cuda_texture` builds `CUDAarray(desc, *shape[::-1])`, so the density array is held in texture order throughout and `update()` is a pure copy. Transposing 128 MiB per frame would cost more than the copy it feeds. `field.empty()` hands out the correctly ordered array and the shape guard makes the mistake loud |

`update()` **raises** on wrong shape (naming the transpose), wrong dtype, a non-contiguous
array, NaN, any negative value, a non-finite value, and a **nonzero boundary shell** — which
means the material is being clipped at the field's own box and the line integral is short.
`Projector.update_contrast` raises if no field was supplied at construction rather than
silently doing nothing, and `Projector.__init__` raises when a density field and a **mesh**
carry the same material, because both paths add into the same `area_density[m]` and the
result is double-counted (`allow_density_field_material_overlap=True` for a deliberate
side-by-side comparison).

### The measurement

`sim/fluoro_sim/qa/density_field_report.py` in mosaic (P2's five checkpoints) and
`sim/tests/test_fluoro_sim_density_field.py` (54 CPU tests: the grid contract, every array
guard, the materials plumbing, and a source guard on the three kernel lines above). Measured on an NVIDIA L4,
2026-08-11.

| # | check | result |
|---|---|---|
| 1 | **Inert at `NUM_DENSITY_FIELDS = 0`** | **bit-identical.** PAT_23 at 512 px, mesh-free (a mesh would measure DeepDRR's OpenGL peel coin-flip instead): 0 of 262 144 px move, max \|diff\| **0.000e+00**, sha256 `0bbdb880…` either side. Render 0.081 s against the 0.086 s bare-volume baseline |
| 2 | **Analytic slab vs a polychromatic hand calculation** | **worst relative error 0.01%** over 10–200 mg I/mL. A 20 mm iodine slab behind 20 cm of soft tissue: 100 mg/mL measures **1.3163** against **1.3162** predicted. The background is 100.03% of its own prediction, i.e. the quadrature lands within one 0.1 mm step of the 200 mm chord |
| 2b | **`boundary` and not `weight`** | With **two** identical volumes at one priority (`n_vols_at_curr_priority = 2`) every row is **identical to the one-volume run**. Under `weight` the iodine would have halved. This is the check the design is written around |
| 3 | **`update_contrast` on a live `Projector`** | monotone over 6 states; `Projector.initialize` calls **1**, from a counter, not an assertion; the texture pointer never changes. 0.9–1.4 ms per update, 0.017 s per render |
| 4 | **The guards** | **20 of 20** fire, each with a message naming the fix |
| 5 | **`enabled = False`** | **bit-identical** to a zero-filled field, to itself after a toggle, and to a `Projector` built with **no field at all** — across a `NUM_MATERIALS` change, because the extra `area_density` term is 0 and sorts first. The field was really on: 1.3163 units at 100 mg I/mL |
| + | **the additive-mesh path is untouched** | `qa/additive_iodine_smoke.py` C1–C3 reproduce mosaic's P1 record digit for digit: deltas 0.0000 → 2.3166, and delta att **1.1102** at 4/10/20/40 segments |

**Per-frame `update()` cost, and one correction to expectations.** The raw copy is measured
by the harness that already owns that number (`qa/render_cost_report.py::measure_texture_copy`),
so the split is measured rather than inferred:

| grid | MiB | `update()` | raw D2D | of which guards | raw H2D |
|---|---|---|---|---|---|
| 20 × 12 × 28 | 0.03 | 0.71–0.97 ms | 0.02 ms | 0.70–0.95 ms | 0.03 ms |
| 181 × 142 × 1307 (PAT_23's lumen box) | 128 | **2.83 / 2.84 / 2.85 ms** | 1.14 / 1.15 / 1.14 ms | **1.98 / 1.91 / 1.93 ms** | 25.0 / 25.9 / 25.9 ms |

Three runs, so the spread is visible rather than implied: the split is stable to ±0.04 ms.

**The guards cost more than the copy.** They are 8 device reductions, so below ~1 MiB the
cost is launch latency alone. A per-frame update is therefore **~2.84 ms, not the 1.12 ms**
the plan projected from the copy alone — still 2% of a 130 ms render, and they are not made
optional: a NaN or a negative density reads as a physics result rather than as an error,
which is the whole reason they exist.

### What it does not change

Every existing caller compiles the field sampling out and gets the same image, byte for byte
(checkpoint 1) — the extra kernel parameters and the hoisted `boundary` are free. The
`boundary`/`weight` split is exact rather than merely equivalent: `1.0f/n` then `* 0.5f`
equals `0.5f/n` in IEEE single precision for every n, because halving is exact and division
rounding is scale-invariant across powers of two. Mosaic renders with one volume, so
`n = 1` anyway.

The new materials block sits in the middle of `Projector.__init__`'s `all_mats` assembly, and
the two upstream behaviours on either side of it — `attenuate_outside_volume` registering
`"air"`, and the name-keyed `all_materials.sort()` — are now pinned by tests, because the
first version of this change deleted the former by accident and no render could have shown it.

### Two things left alone on purpose

1. `voxels[][][]` and `previous_coordinates[]` are declared **outside** the `vol_id` loop and
   written **inside** it, so with `NUM_VOLUMES > 1` volume 1 can reuse volume 0's labels
   wherever their base coordinates coincide. This change makes checkpoint 2b run at
   `NUM_VOLUMES = 2` — with two **identical** grids, so the latent bug is inert there and
   the check is unaffected. It is a separate fork change with its own test; it is not made
   worse and it is not fixed here.
2. `gVoxelElementSize{X,Y,Z}` is passed to the kernel and never used. The field parameters
   deliberately do **not** imitate it: a field needs no voxel size in the kernel, because
   `alpha` is a world-unit ray and the box test is in IJK.
