from __future__ import annotations

import shutil
from pathlib import Path
from typing import Tuple, List
from pmtools.refractored_toolbox import assemble_paths
import h5py

def copy_and_reduce_data_to_src(
    master_dict_data: dict,
    master_dict_src: dict,
    data_root: str | Path,
    src_root: str | Path,
    template_data,
    template_src
) -> List[Tuple[Path, Path]]:
    """
    Build source and destination paths from templates and copy each file.
    After each copy, collapse all property datasets to a single timestep.

    Returns a list of (src, dst) pairs that were processed.
    """
    paths_data, _ = assemble_paths(master_dict_data, template_data)  # SOURCE (no '/src')
    paths_src, _ = assemble_paths(master_dict_src, template_src)     # DEST   (with '/src')

    if len(paths_data) != len(paths_src):
        raise ValueError(f"Mismatched counts: {len(paths_data)} sources vs {len(paths_src)} dests")

    data_root = Path(data_root)
    src_root = Path(src_root)

    processed: List[Tuple[Path, Path]] = []

    for rel_src, rel_dst in zip(paths_data, paths_src):
        src = data_root / rel_src
        dst = src_root / rel_dst

        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        reduce_h5_file(dst, t_index=-1)
        processed.append((src, dst))

    return processed


def reduce_h5_file(
    dst_path: str | Path,
    t_index: int = -1,
) -> None:
    """
    Open an HDF5 file in-place and keep only a single timestep across
    all datasets shaped like (T, N, D...):
      - /particles/<Group>/<prop>/value -> resized to (1, N, D...)
      - /particles/<Group>/<prop>/step  -> length 1 (if present)
      - /particles/<Group>/<prop>/time  -> length 1 (if present)
    """
    dst_path = Path(dst_path)

    with h5py.File(dst_path, "r+") as f:
        grp_particles = f["particles"]
        for group_name in grp_particles:
            g = grp_particles[group_name]
            for _, prop_grp in g.items():
                val = prop_grp["value"]

                slice_data = val[t_index, ...]  # shape (1, N, D...)
                val.resize((1,) + val.shape[1:])
                val[0, ...] = slice_data

                ds = prop_grp["step"]
                step_val = ds[t_index]
                ds.resize((1,))
                ds[0] = step_val

                ds = prop_grp["time"]
                time_val = ds[t_index]
                ds.resize((1,))
                ds[0] = time_val

    print(f"✔ Shrunk to single timestep at: {dst_path}")