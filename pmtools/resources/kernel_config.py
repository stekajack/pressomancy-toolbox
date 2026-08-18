from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from string import Template as StrTemplate

from typing import Optional, Any, Protocol, TypeAlias, Tuple
from collections.abc import Callable, Sequence
import numpy as np
from numpy.typing import NDArray

# ---- Minimal protocol describing what your "subset" exposes to predicates ----
class HasType(Protocol):
    @property
    def type(self) -> NDArray[np.integer] | np.integer: ...

# Predicates may return a single bool (e.g., ".any()") or a boolean mask array.
PredicateReturn: TypeAlias = bool | NDArray[np.bool_]
Predicate: TypeAlias = Callable[[HasType], PredicateReturn]

# If you need stricter separation at call sites, you can also use:
BoolPredicate: TypeAlias = Callable[[HasType], bool]
MaskPredicate: TypeAlias = Callable[[HasType], NDArray[np.bool_]]

# ---- Config “struct” ----
@dataclass(frozen=True)
class AnalysisConfig:
    data_path: Path | str
    template_hndl: StrTemplate | None
    particle_group: str
    box_dim: np.ndarray | Sequence[float]

    # Time selection as a pythonic (start, end, step). Negative indices allowed.
    chunk: Tuple[Optional[int], Optional[int], int] = (-5, None, 1)

    # Common knobs most of your functions share:
    norm: float = 1.0
    crit: float = 1.47
    extra_flag: Any = None

    # Optional context for functions that need it
    particle_group_alt: str = ''
    sq_params: dict = field(default_factory=lambda: {
        'order': 10,
        'orientations_per_wavevector': 100,
        'subsample_wavevectors': 1,
        'axis_mask': [True, True, True],
        'nthreads': 1,
    })
    volume_cutoff: float | None = None
    probability_floor: float | None = None
    min_cluster_size: int = 20
    z_bin_width: float = 1.0
    min_slice_population: int = 3

    # Predicates (optional). Accept either a bool or a mask return.
    object_predicate: Predicate | None = None
    particle_predicate: Predicate | None = None

    path_to_output: Path | str = ''
