"""A continuous per-voxel **additive material-density channel** for the ray march.

An :class:`AdditiveDensityField` is one float32 3-D grid of ``g/cm^3`` of **one named
material**, resident on the GPU as a linear-filtered texture, sampled once per ray step and
added to that material's area density. It is the mechanism for rendering a *continuous*
concentration field -- an iodine bolus, a calcium distribution, a pocket of bowel gas --
which the volume path cannot express (one material label plus one density per voxel, so a
continuous concentration has to be quantized into material bins) and the additive-mesh path
can only express piecewise-constant per mesh.

**Why this is not a ``Volume``, and not a ``Renderable``.**

A ``Volume`` answers "what matter is at this point": its materials partition the voxel, and
where several volumes overlap at one priority the kernel *averages* them, because they are
competing descriptions of the same matter. A density field answers a different question --
"how much of this one extra material is dissolved here" -- and is therefore **added**, never
averaged. Passing it as ``Projector(..., density_fields=[field])`` rather than in the
renderable list keeps that distinction in the type system instead of in a comment: it has no
priority, it is not composited by the priority rule, and ``Projector([ct, mesh, ...])``
semantics are untouched.

**Grid.** Its own ``shape``, ``spacing`` and a **full** ``world_from_ijk`` transform -- not
"volume 0's grid plus an integer offset" -- so it can be finer than the volume, or cover
only the region that actually carries the material. Shape and spacing are in **IJK**, the
same order as ``Volume.shape``.

**Axis order, which is the one thing to get right.** ``create_cuda_texture`` builds
``CUDAarray(desc, *shape[::-1])`` and ``Projector.initialize`` pre-transposes each volume
with ``moveaxis([0, 1, 2], [2, 1, 0])``. Transposing this array every frame would cost more
than the copy it feeds, so the density array is held in **texture order** (KJI) throughout:
:meth:`empty` hands out a correctly shaped, correctly ordered device array,
:meth:`update` is then a pure ``cudaMemcpy3D``, and a shape guard makes an IJK-ordered
array a hard error rather than a silently transposed image.

**Zero boundary shell.** The kernel tests each field's box explicitly and never samples
outside it, so the texture's address mode is unreachable and irrelevant; ``clamp`` is kept
because something must be chosen. What *does* matter is that a field whose material is
nonzero **at** its own boundary is a field that has been clipped by its own box -- the
matter continues and the render stops it -- so :meth:`update` refuses one. One mechanism
(the guard), not two (a guard plus a border colour).

Mosaic fork change; see ``MOSAIC_FORK_CHANGES.md`` item 2 in this checkout, and
``sim/fluoro_sim/CONTRAST_INTEGRATION.md`` sec 7j in mosaic for the design and the numbers.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Sequence, Tuple

import numpy as np

from .. import geo

log = logging.getLogger(__name__)

#: The only dtype a density field carries. float32 + linear filtering is what makes the
#: concentration continuous; the material-label texture is uint8/nearest by necessity
#: (labels cannot be interpolated) and would be blocky by construction.
DENSITY_DTYPE = np.float32


def _array_module(rho: Any):
    """``cupy`` for a device array, ``numpy`` for a host one, without importing cupy eagerly."""
    import cupy

    return cupy.get_array_module(rho)


def validate_density_array(
    rho: Any,
    expected_shape: Tuple[int, int, int],
    *,
    material: str,
    check_boundary_shell: bool = True,
) -> None:
    """Raise unless ``rho`` is a valid density array for a field of ``expected_shape``.

    Pure, and it works on a ``numpy`` host array or a ``cupy`` device array, so the guards
    are testable without a GPU. Every check corresponds to a failure that would otherwise
    read as a *physics* result rather than as a bug:

    ==============================  ===========================================================
    check                           what it would look like if it passed
    ==============================  ===========================================================
    shape                           a transposed field, i.e. matter in the wrong place
    dtype                           a silent cast, or a refused ``cudaMemcpy3D``
    C-contiguity                    a hidden full-size copy on every frame
    no NaN                          ``exp(-NaN)`` -> a NaN pixel, which survives to the export
    no negative values              *less* attenuation -- contrast that makes tissue thinner
    finite                          an all-black frame
    zero boundary shell             the material clipped at the box, so the total is short
    ==============================  ===========================================================

    Args:
        rho: candidate density array, in **texture (KJI) order** -- see
            :meth:`AdditiveDensityField.empty`.
        expected_shape: the field's :attr:`AdditiveDensityField.texture_shape`.
        material: the field's material name, for the error messages.
        check_boundary_shell: whether to require a zero outer shell. Always ``True`` from
            :meth:`AdditiveDensityField.update`; the parameter exists for the unit tests
            that check the other guards in isolation.
    """
    if not hasattr(rho, "shape") or not hasattr(rho, "dtype"):
        raise TypeError(
            f"density field {material!r} update needs a cupy or numpy array, got "
            f"{type(rho).__name__}. Get a correctly shaped one from field.empty() and fill "
            f"it in place."
        )

    expected = tuple(int(v) for v in expected_shape)
    got = tuple(int(v) for v in rho.shape)
    if got != expected:
        hint = ""
        if got == expected[::-1]:
            hint = (
                " That is exactly the reverse, so this array is in IJK order and the "
                "texture wants KJI: use field.empty() and fill it, or pass "
                "rho.transpose(2, 1, 0) made contiguous. Copying it as-is would put the "
                "material in the wrong place with no error."
            )
        raise ValueError(
            f"density field {material!r} expects shape {expected} (texture KJI order, i.e. "
            f"reversed field.shape), got {got}.{hint}"
        )

    if np.dtype(rho.dtype) != np.dtype(DENSITY_DTYPE):
        raise ValueError(
            f"density field {material!r} expects dtype {np.dtype(DENSITY_DTYPE)}, got "
            f"{np.dtype(rho.dtype)}. The CUDAarray is float32 and cudaMemcpy3D does not "
            f"convert; build the array with field.empty() (or "
            f"astype(numpy.float32, copy=False)) rather than relying on a cast."
        )

    if not rho.flags.c_contiguous:
        raise ValueError(
            f"density field {material!r} update needs a C-contiguous array; got a view or a "
            f"transpose. cudaMemcpy3D copies a contiguous block, and silently calling "
            f"ascontiguousarray here would hide a full-size allocation on every frame. Wrap "
            f"it yourself: cupy.ascontiguousarray(rho)."
        )

    xp = _array_module(rho)
    # One reduction on the happy path. min() propagates NaN, so `not (lo >= 0)` catches NaN
    # and negatives together; only the failure branch pays for telling them apart.
    lo = float(rho.min()) if rho.size else 0.0
    if not (lo >= 0.0):
        if bool(xp.isnan(rho).any()):
            raise ValueError(
                f"density field {material!r} contains NaN "
                f"({int(xp.isnan(rho).sum())} of {rho.size} voxels). A NaN area density makes "
                f"exp(-tau) NaN, so the pixel is NaN all the way to the exported image. Fix "
                f"the producer -- most often a 0/0 in a concentration-to-density conversion, "
                f"or an uninitialized scatter target; field.empty() starts at zero."
            )
        raise ValueError(
            f"density field {material!r} contains negative densities (min {lo:g}). A negative "
            f"area density *reduces* attenuation, so it would render as thinner tissue rather "
            f"than as an error. Clip the producer at 0 (numpy/cupy .clip(0, None)) if small "
            f"negatives are interpolation undershoot, or fix its sign."
        )
    hi = float(rho.max()) if rho.size else 0.0
    if not np.isfinite(hi):
        raise ValueError(
            f"density field {material!r} contains a non-finite density (max {hi:g}). exp(-inf) "
            f"is 0, so the affected rays render as a black hole rather than as an error."
        )

    if check_boundary_shell and rho.size:
        worst = 0.0
        for axis in range(3):
            for index in (0, -1):
                face = rho[(slice(None),) * axis + (index,)]
                if face.size:
                    worst = max(worst, float(abs(face).max()))
        if worst > 0.0:
            raise ValueError(
                f"density field {material!r} is nonzero on its own boundary shell (max "
                f"{worst:g} g/cm^3 on a face). The kernel never samples outside the field's "
                f"box, so that material is being clipped at the box and the line integral is "
                f"short by however much continues past it. Grow the field's shape (or move "
                f"its origin) until the support has at least one zero voxel on every face."
            )


class AdditiveDensityField:
    """One float32 ``g/cm^3`` grid of one material, added into the ray march.

    Usage -- the field is built once, uploaded once, and updated per frame::

        field = AdditiveDensityField(shape=(181, 142, 1307), spacing=(0.98, 0.98, 0.63),
                                     origin=lumen_box_origin_mm, material="I")
        rho = field.empty()                      # device, texture order, zeros
        with Projector(volume, device=device, density_fields=[field]) as projector:
            for state in states:
                fill(rho, state)                 # your kernel, in place
                projector.update_contrast(rho)   # one cudaMemcpy3D, no re-init
                image = projector.project()

    Args:
        shape: grid size in **IJK**, matching ``Volume.shape``.
        spacing: mm per voxel in IJK. Ignored if ``anatomical_from_ijk`` is given.
        origin: anatomical position of voxel ``(0, 0, 0)``. Ignored if
            ``anatomical_from_ijk`` is given.
        material: a DeepDRR material **name**, e.g. ``"I"`` for iodine or ``"calcium"``.
            Resolved by name everywhere downstream (the absorption table, the segmentation
            remap), so nothing else has to know what this field carries.
        anatomical_from_ijk: the full grid transform, if ``spacing``/``origin`` are not
            enough (a rotated grid, say). Defaults to ``diag(spacing)`` with translation
            ``origin``. **There is deliberately no**
            ``anatomical_coordinate_system="LPS"`` **shortcut like**
            ``Volume.from_parameters`` **has**: that convention folds the grid origin into a
            permuted anatomical frame, so two grids with different origins end up in two
            *different* anatomical frames -- which is a silent misregistration when a field
            is meant to sit inside a volume. Pass ``world_from_anatomical`` instead.
        world_from_anatomical: anatomical -> world, defaulting to identity.
        enabled: whether the kernel samples it. Re-read on **every** ``project()``, so a
            DSA mask frame is "the same scene with contrast off" -- one int, rather than
            zeroing and re-uploading the whole grid.
    """

    def __init__(
        self,
        shape: Sequence[int],
        spacing: Sequence[float] = (1.0, 1.0, 1.0),
        origin: Sequence[float] = (0.0, 0.0, 0.0),
        material: str = "I",
        anatomical_from_ijk: Optional[geo.FrameTransform] = None,
        world_from_anatomical: Optional[geo.FrameTransform] = None,
        enabled: bool = True,
    ) -> None:
        shape = tuple(int(v) for v in np.asarray(shape).reshape(-1))
        if len(shape) != 3 or min(shape) < 2:
            raise ValueError(
                f"a density field's shape must be 3 IJK dimensions of at least 2, got "
                f"{shape}. A dimension of 1 cannot carry the zero boundary shell that keeps "
                f"the material from being clipped at the box (see validate_density_array)."
            )
        self.shape = shape

        spacing_arr = np.asarray(spacing, dtype=np.float64).reshape(-1)
        if spacing_arr.size != 3 or not np.all(spacing_arr > 0):
            raise ValueError(
                f"a density field's spacing must be 3 positive mm values, got {spacing}."
            )
        self.spacing = tuple(float(v) for v in spacing_arr)

        if not isinstance(material, str) or not material:
            raise ValueError(
                f"a density field carries one DeepDRR material NAME, got {material!r}. "
                f"Pass e.g. \"I\" for iodine; the name is what the absorption table and the "
                f"kernel's material index are keyed on."
            )
        self.material = material

        if anatomical_from_ijk is None:
            anatomical_from_ijk = geo.FrameTransform.from_rt(
                rotation=np.diag(spacing_arr),
                translation=np.asarray(origin, dtype=np.float64).reshape(3),
            )
        self.anatomical_from_ijk = geo.frame_transform(anatomical_from_ijk)
        self.world_from_anatomical = (
            geo.FrameTransform.identity(3)
            if world_from_anatomical is None
            else geo.frame_transform(world_from_anatomical)
        )

        self.enabled = bool(enabled)

        # Set by initialize(), cleared by free(). Owned by the Projector's lifecycle, like
        # the volume textures: a field outlives one Projector, its GPU memory does not.
        self._texobj = None
        self._texarr = None

    # ------------------------------------------------------------------ geometry #
    @property
    def texture_shape(self) -> Tuple[int, int, int]:
        """The density array's shape: :attr:`shape` reversed, i.e. KJI.

        This is what :meth:`empty` returns and what :meth:`update` requires. See the module
        docstring for why the array is held this way round rather than transposed per frame.
        """
        return self.shape[::-1]

    @property
    def world_from_ijk(self) -> geo.FrameTransform:
        return self.world_from_anatomical @ self.anatomical_from_ijk

    @property
    def ijk_from_world(self) -> geo.FrameTransform:
        return self.world_from_ijk.inverse()

    #: Aliases matching ``Renderable``'s capitalized spellings, so call sites that already
    #: say ``IJK_from_world`` for a volume read the same for a field.
    @property
    def world_from_IJK(self) -> geo.FrameTransform:
        return self.world_from_ijk

    @property
    def IJK_from_world(self) -> geo.FrameTransform:
        return self.ijk_from_world

    @property
    def nbytes(self) -> int:
        return int(np.prod(self.shape)) * np.dtype(DENSITY_DTYPE).itemsize

    def get_bounding_box_in_ijk(self) -> Tuple[Tuple[float, float, float], ...]:
        """The field's box in its own IJK, cell-centred: ``(-0.5, shape - 0.5)``.

        The same convention as ``gVolumeEdgeMin/MaxPoint*``, so the kernel's slab test is
        literally the volumes' slab test with different inputs.
        """
        return (
            (-0.5, -0.5, -0.5),
            tuple(float(s) - 0.5 for s in self.shape),
        )

    # ------------------------------------------------------------------- lifecycle #
    @property
    def initialized(self) -> bool:
        return self._texarr is not None

    def empty(self):
        """A zeroed **device** array of :attr:`texture_shape`, ready to fill and pass to
        :meth:`update`.

        The whole reason this exists rather than a documented convention: a caller who
        allocates their own array has to get the axis order right from prose, and getting it
        wrong produces a plausible image with the material in the wrong place. Filling the
        array this returns cannot be wrong that way.
        """
        import cupy as cp

        return cp.zeros(self.texture_shape, dtype=DENSITY_DTYPE)

    def initialize(self) -> None:
        """Allocate the resident ``CUDAarray`` and its texture. Called by ``Projector``."""
        from ..projector.projector import create_cuda_texture

        if self.initialized:
            raise RuntimeError(
                f"density field {self.material!r} already holds a texture, so it is attached "
                f"to a live Projector. Two initialized Projectors cannot share one field: "
                f"free the first Projector before building the second. (If the first "
                f"Projector's initialize() failed part-way, call field.free() -- Projector.free "
                f"only cleans up a Projector that finished initializing.)"
            )
        import cupy as cp

        zeros = cp.zeros(self.texture_shape, dtype=DENSITY_DTYPE)
        self._texobj, self._texarr = create_cuda_texture(
            zeros, sampling_mode="linear", address_mode="clamp"
        )
        del zeros
        cp.get_default_memory_pool().free_all_blocks()
        log.debug(
            f"density field {self.material!r}: {self.shape} IJK, "
            f"{self.nbytes / 1024 ** 2:.1f} MiB resident"
        )

    def free(self) -> None:
        """Release the texture. Idempotent, so ``Projector.free`` need not check."""
        self._texobj = None
        self._texarr = None

    @property
    def texture_pointer(self) -> int:
        if not self.initialized:
            raise RuntimeError(
                f"density field {self.material!r} has no texture yet. It is allocated by "
                f"Projector.initialize; render inside `with Projector(...)`."
            )
        return int(self._texobj.ptr)

    # ----------------------------------------------------------------------- update #
    def update(self, rho: Any) -> None:
        """Copy ``rho`` into the resident texture: one ``cudaMemcpy3D``, no re-allocation.

        ``rho`` is in **texture (KJI) order** -- use :meth:`empty` to get one -- and is
        validated in full by :func:`validate_density_array` first. A device array is the
        fast path (measured D2D 1.15 ms for 128 MiB on an L4); a host array works and costs
        an order of magnitude more (H2D 25.8 ms).

        Nothing about the texture, the kernel or the ``Projector`` is rebuilt, which is the
        entire point: a bolus sequence is one volume upload plus N of these.
        """
        if not self.initialized:
            raise RuntimeError(
                f"density field {self.material!r} has no texture to update yet. Pass it as "
                f"Projector(..., density_fields=[field]) and update inside the Projector's "
                f"`with` block -- the texture is allocated by Projector.initialize and freed "
                f"by Projector.free."
            )
        validate_density_array(rho, self.texture_shape, material=self.material)
        self._texarr.copy_from(rho)

    def __repr__(self) -> str:
        return (
            f"AdditiveDensityField(shape={self.shape}, spacing={self.spacing}, "
            f"material={self.material!r}, enabled={self.enabled}, "
            f"initialized={self.initialized})"
        )
