# Pressomancy Toolbox

**Pressomancy Toolbox (`pmtools`) provides analysis kernels for Pressomancy simulation trajectories.** It builds on Pressomancy's HDF5 data access, object connectivity, and particle selection API to analyse filament conformation, particle clusters, magnetic observables, and local structure.

Kernels share an `AnalysisConfig` interface and can be called directly for one trajectory or dispatched across a parameter grid with the process-based `Engine`. Most return results keyed by the input path, preserving their association with the source simulation.

This package is under development. The distribution name is `pressomancy-toolbox`; the Python import is `pmtools`.

## Installation

Use **Python 3.10 or newer** in an environment with a compatible Pressomancy installation.

From this repository's root:

```bash
python -m pip install -e .
```

Installation includes the core dependencies: `numpy`, `h5py`, `igraph`, `vg`, and `pressomancy`. To use a local Pressomancy checkout, install it first with `python -m pip install -e ../pressomancy`. Pressomancy must provide `pressomancy.analysis.H5DataSelector` and the neighbour-search helpers imported by `pmtools.kernels`.

Additional dependencies are loaded by specific kernels:

| Analysis | Required module/API |
| --- | --- |
| Static structure factor (`pmtools.kernels.calculate_sf`) | `sq_avx` |
| RDF and Voronoi analysis, including cylindrical radial sampling | `pyscal` with `System`, `Atom`, and the methods used in the kernels |

`pmtools.kernels.calculate_sf(cfg)` is the toolbox entry point for static structure-factor analysis. It calls `calculate_structure_factor` from the external `sq_avx` module. These are the analysis kernel and its backend, respectively, rather than alternative implementations.

The repository does not pin compatible dependency versions. Check these APIs in your analysis environment before using the corresponding kernels.

## Data and selection model

The input is a Pressomancy HDF5 trajectory readable by `H5DataSelector`. Kernels use its timestep slicing, particle properties, and, where needed, `get_connectivity_values` and `select_particles_by_object` methods.

- **Particle group:** `particle_group` identifies the stored group to analyse. Use the group name from your simulation output.
- **Object selection:** `object_predicate` filters objects through Pressomancy's connectivity API.
- **Particle selection:** `particle_predicate` selects particles within the relevant data view, for example by particle type.
- **Time selection:** `chunk=(start, stop, step)` slices saved frames using Python indexing. The default `(-5, None, 1)` selects the final five frames; `(0, None, 1)` selects all frames.
- **Geometry:** provide the simulation box as `[Lx, Ly, Lz]`, in the same units as the coordinates and distance cutoff. Kernels do not infer it automatically from the trajectory.

Required particle properties depend on the kernel: positions for geometric analysis, IDs and connectivity for object-based analysis, and `dip` for magnetic observables. The shared `get_cluster_iterator` routine reconstructs missing `pos_folded` values from `pos` using the supplied box dimensions when attaching graph attributes. This fallback is local to graph construction; it does not update the selector or the HDF5 file. Some modern kernels read `pos_folded` before calling that routine (for example, `cluster_size`), while others read it without constructing a graph (for example, `calculate_sf` and `calculate_rdf`). Those accesses currently require stored `pos_folded` data.

Filament kernels connect consecutive selected particle IDs within each object. Their results therefore depend on particle ordering representing the intended chain topology. Distance-based kernels instead construct neighbour graphs using `crit`. The graph helpers reconstruct components across periodic boundaries before computing shape observables.

## Analyse one trajectory

This example computes a gyration tensor for each selected filament in the final five frames. Replace the file path, group name, particle type, and box dimensions with those of your simulation.

```python
import numpy as np

from pmtools import AnalysisConfig
from pmtools.kernels import per_fil_gyr


def select_objects(subset):
    # Keep objects containing at least one particle of the chosen type.
    return (subset.type == 0).any()


def select_particles(subset):
    return subset.type == 0


cfg = AnalysisConfig(
    data_path="/data/trajectory.h5",
    template_hndl=None,
    particle_group="Filament",  # Example group name; use your stored group.
    box_dim=np.array([50.0, 50.0, 50.0]),
    chunk=(-5, None, 1),
    object_predicate=select_objects,
    particle_predicate=select_particles,
)

results = per_fil_gyr(cfg)
tensors = results[cfg.data_path]
rg_squared = np.array([tensor.get_R2() for tensor in tensors])
shape_anisotropy = np.array([tensor.get_k2() for tensor in tensors])
```

`per_fil_gyr` returns a flat list of `GyrationTensor` objects accumulated over the selected frames and components. `get_R2()` returns the **squared** radius of gyration; use `np.sqrt(rg_squared)` for the radius. Tensors also expose `array`, `eigenvalues`, `eigenvectors`, `get_b()` (asphericity), and `get_c()` (acylindricity).

Use array-valued boolean masks for particle predicates. Some kernels call the predicate directly and require `.flatten()`, even though the configuration type also permits scalar booleans.

## Run a parameter sweep

`Engine` expands a `string.Template` over parameter values and submits one task per registered kernel and input file. Save the following as a Python script, replacing the example paths and simulation settings. All generated input files must already exist.

```python
from string import Template

from pmtools.kernels import per_fil_gyr
from pmtools.runner import Engine


def select_objects(subset):
    return (subset.type == 0).any()


def select_particles(subset):
    return subset.type == 0


def main():
    parameters = {
        "field": ["0.0", "1.0"],
        "run": ["0", "1", "2"],
    }
    template = Template("field_${field}/run_${run}/trajectory.h5")

    # Engine concatenates this string with each generated relative path.
    with Engine("/data/simulations/") as engine:
        engine.register_kernel(
            per_fil_gyr,
            particle_group="Filament",
            box_dim=[50.0, 50.0, 50.0],
            chunk=(-5, None, 1),
            object_predicate=select_objects,
            particle_predicate=select_particles,
        )
        engine.assemble_paths(parameters, template, parallel_param_id="run")
        engine.run(max_workers=4)
        results = engine.collect_results().copy()
        engine.save_results("gyration")

    return results


if __name__ == "__main__":
    results = main()
```

This schedules six trajectory analyses and writes `/data/simulations/gyration.p.gz`. The `run` parameter is expanded within each field assignment, so its results are grouped together. Use strings for this parameter's values: the current expansion uses string replacement. Omitting `parallel_param_id` gives each generated path its own result key; tasks still execute in the process pool.

Collected results have the structure:

```text
kernel_name
└── assignment_key
    └── list of kernel results, in parameter expansion order
        └── input_path → kernel-specific payload
```

For this example, an assignment key is `field_0.0/run_parallel_param_placeholer/trajectory.h5`. The placeholder spelling reflects the current implementation. Grouping preserves individual run results; it does not average them.

Define kernels and predicates at module scope so they can be pickled by the process pool, and use the `__main__` guard. The default engine uses 10 workers; the class-wide worker limit is 16. Include the trailing separator in `world_path`, since input paths are assembled by string concatenation.

Call `collect_results()` once, then `save_results()` before leaving the context. Shutdown clears the engine's result dictionary; the shallow `.copy()` above retains the collected mapping for later use. Saved results are gzip-compressed Python pickles and may contain `GyrationTensor` instances.

## Available analyses

The current kernels are in [`pmtools/kernels.py`](pmtools/kernels.py).

| Area | Kernels and outputs |
| --- | --- |
| Filament conformation | `per_fil_gyr`: gyration tensors; `Ree_segments`: end-to-end distances and segment lengths; `lp_projection`: segment projections; `mlp_projection`: dipole projections onto the end-to-end vector |
| Cluster geometry | `per_cluster_gyr_tensor`: gyration tensors; `cluster_size`: sizes grouped by frame; `pair_distances`: neighbour-distance samples |
| Magnetism | `magnetisation`, `cluster_magnetisation`, `magnetisation_culster_size`, `degree_magnetisation`, `magn_princip_angle_dist` |
| Structure | `calculate_sf`: static structure factor; `calculate_rdf`: radial distribution function |
| Voronoi geometry | `calculate_volume_voronoi`, `calculate_voronoi_vects`, `calculate_voronoi_face_perimeters`, `calculate_voronoi_no_of_edges`, `calculate_voronoi_vertex_vectors` |
| Local environments | `tag_voronoi_particles`: polyhedron-based tags; `cylindrical_radial_density`: tagged radial-distance samples in z slices |
| Contacts and orientation | `calculate_stacking_fraction`: ligand–stacking-site neighbour mappings; `easy_dipmom_angle`: dipole/director cosine samples from a specific anchor layout |
| Export and inspection | `write_vtk_frame`, `write_cluster_to_vtk`, `get_data_timestep_len` |

Names above preserve the public API's current spelling. Most analysis kernels return `{cfg.data_path: payload}`. Payloads may retain frame boundaries, flatten samples across frames, or contain averaged arrays; inspect the selected kernel before combining results. VTK exporters write files and return `0`.

## Configuration and analysis conventions

[`AnalysisConfig`](pmtools/resources/kernel_config.py) is a frozen dataclass. Its four required fields are `data_path`, `template_hndl`, `particle_group`, and `box_dim`. Set `template_hndl=None` for kernels that do not parse filename parameters. With `Engine`, the runner supplies `data_path` and `template_hndl`; pass the remaining fields to `register_kernel`.

| Field | Default | Purpose |
| --- | --- | --- |
| `chunk` | `(-5, None, 1)` | Saved-frame slice |
| `crit` | `1.47` | Neighbour/contact cutoff where used |
| `norm` | `1.0` | Normalisation factor where used |
| `object_predicate`, `particle_predicate` | `None` | Selection functions; requirements depend on the kernel |
| `particle_group_alt` | `''` | Second group for stacking contacts |
| `sq_params` | See configuration source | Structure-factor sampling and threading parameters |
| `volume_cutoff`, `probability_floor` | `None` | Required explicitly for Voronoi tagging and cylindrical radial sampling |
| `min_cluster_size` | `20` | Minimum cluster size for cylindrical radial sampling |
| `z_bin_width`, `min_slice_population` | `1.0`, `3` | Cylindrical sampling slice width and minimum population |
| `path_to_output` | `''` | Output directory for file-writing kernels |

Choose kernels with their implemented conventions in mind:

- Several neighbour searches and the structure-factor calculation pass only `box_dim[0]` to their backend. Check the cubic-box assumption before using these kernels with a non-cubic box.
- Several distance-based cluster kernels hard-code a minimum of 20 particles. `min_cluster_size` does not override that threshold globally.
- `magnetisation` and `cluster_magnetisation` report the mean **z component** of the dipoles divided by `norm`.
- `Ree_segments`, `lp_projection`, `mlp_projection`, and `magn_princip_angle_dist` parse `what_monomer_number` from the input filename using `template_hndl`. Supply a template matching the full input path and containing that placeholder. Monomer-based reshaping requires compatible particle ordering and counts.
- `calculate_stacking_fraction` uses type-4 stacking sites and type-5 ligands and returns neighbour mappings rather than a normalised fraction. `easy_dipmom_angle` assumes exactly 40 type-2 anchors per filament, ordered as 20 bottom anchors followed by 20 top anchors.
- `cylindrical_radial_density` returns radial samples rather than a normalised density histogram. Voronoi tagging requires `volume_cutoff > 0` and `0 < probability_floor <= 1`.
- Create export directories before calling file-writing kernels. Use separate directories for different trajectories: cluster and Voronoi filenames contain local frame/component indices and can collide. `write_vtk_frame` additionally assumes a specific input-path depth and a string path.

## Extending the toolbox

A custom kernel accepts one `AnalysisConfig` and returns a picklable result. Following the input-path mapping convention makes its output consistent with the built-in analyses:

```python
import h5py
from pressomancy.analysis import H5DataSelector

from pmtools import AnalysisConfig


def count_selected_frames(cfg: AnalysisConfig):
    with h5py.File(cfg.data_path, "r") as trajectory:
        data = H5DataSelector(trajectory, particle_group=cfg.particle_group)
        start, stop, step = cfg.chunk
        count = len(data.timestep[start:stop:step].timestep)
    return {cfg.data_path: count}
```

Call it directly or register it with `Engine` using the same workflow. Keep configuration and callable arguments picklable when using parallel execution.

## Supporting modules

- [`pmtools/refractored_toolbox.py`](pmtools/refractored_toolbox.py): parameter-template expansion, filename parsing, periodic geometry, graph reconstruction, and particle-ID-to-edge mapping.
- [`pmtools/data_helpers.py`](pmtools/data_helpers.py): copy and reduce trajectories to a single frame, and bin structure-factor data for plotting. `reduce_h5_file` modifies its target in place and requires resizable `value`, `step`, and `time` datasets; use a copy when retaining the original trajectory.
- [`pmtools/resources/gryation_tensor.py`](pmtools/resources/gryation_tensor.py): gyration tensor and shape observables.

## Legacy code

`kernels_legacy.py` and `refractored_toolbox_legacy.py` are unmaintained. They are retained temporarily while their remaining useful functionality is reviewed and are intended for removal once that review is complete. Use `pmtools.kernels` and the `AnalysisConfig` workflow for new analyses.

The current kernels still import `pair_potential` from `refractored_toolbox_legacy.py`; that helper must be preserved or migrated before the legacy module is removed.
