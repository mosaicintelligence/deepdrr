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
