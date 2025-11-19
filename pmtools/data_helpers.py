from __future__ import annotations

import shutil
from pathlib import Path
from typing import Tuple, List
from pmtools.refractored_toolbox import assemble_paths
import h5py
import numpy as np

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


def bin_and_arrange_Sq_for_plot(wave, intens, num_bins=50, 
                                         subsampling_steps=[1, 2, 3, 4, 5], 
                                         segments=[(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)], 
                                         aggregate=True):
    """
    Bin the wave and intens data and subsample the resulting bins.

    Parameters
    ----------
    wave : array-like
        The x-values.
    intens : array-like
        The y-values (same length as `wave`).
    num_bins : int, optional
        Total number of bins (default: 50).
    subsampling_steps : list of int, optional
        Step size for subsampling in each segment (default: [1, 2, 3, 4, 5]).
    segments : list of tuple, optional
        Start (inclusive) and end (exclusive) bin indices for each segment (default:
        [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]).
    aggregate : bool, optional
        If True, compute the average of intensities in each bin and return a 1-D array.
        If False, return a list of numpy arrays with the raw intensities for each bin.

    Returns
    -------
    subsampled_centers : np.ndarray
        The bin centers after subsampling.
    subsampled_data : np.ndarray or list of np.ndarray
        If `aggregate` is True, a numpy array of the averaged intensities for each
        subsampled bin. If False, a list of numpy arrays (raw intensities per bin).

    Notes
    -----
    - The function always creates `num_bins` equal-width bins across [min(wave), max(wave)].
      However, the number of returned points is determined by the `segments` and
      `subsampling_steps` parameters. The defaults split 50 bins into five segments of
      10 bins each and apply subsampling steps [1,2,3,4,5], so the typical returned
      count with defaults is 10 + 5 + 4 + 3 + 2 = 24 (less if bins are empty or edge
      values fall outside the counted bins).
    - `np.digitize` without `right=True` will assign values equal to `max(wave)` to
      bin index `num_bins + 1`, which this function ignores in the loop (bins 1..num_bins).
      To include right-edge values, use `np.digitize(..., right=True)` (this is not
      enabled here to preserve original behaviour).
    - Empty bins lead `np.mean([])` to produce `nan` when `aggregate=True`.
 
    """
    # Ensure that subsampling_steps and segments are compatible
    assert len(subsampling_steps) == len(segments), 'Must have compatible subsampling_steps and segments dimensions'
    
    # Create bins over the range of wave data
    bins = np.linspace(min(wave), max(wave), num_bins + 1)
    bin_indices = np.digitize(wave, bins)
    
    # Prepare lists to hold bin centers and bin data
    bin_centers = []
    bin_data = []
    
    # Loop through each bin to compute center and corresponding data
    for i in range(1, num_bins + 1):
        indices_in_bin = np.where(bin_indices == i)[0]
        y_in_bin = intens[indices_in_bin]
        center = (bins[i - 1] + bins[i]) / 2
        bin_centers.append(center)
        
        # Depending on 'aggregate', either compute the mean or keep raw data
        if aggregate:
            bin_data.append(np.mean(y_in_bin))
        else:
            bin_data.append(y_in_bin)
    
    bin_centers = np.array(bin_centers)
    
    # Subsample the bin centers based on provided segments and steps
    subsampled_centers = []
    for (start, end), step in zip(segments, subsampling_steps):
        if step == 1:
            subsampled_centers.extend(bin_centers[start:end])
        else:
            subsampled_centers.extend(bin_centers[start:end:step])
    subsampled_centers = np.array(subsampled_centers)
    
    # Filter the bin_data based on the subsampled bin centers
    # Note: Since bin_centers are unique, we can use np.isin
    filter_indices = np.isin(bin_centers, subsampled_centers)
    if aggregate:
        subsampled_data = np.array(bin_data)[filter_indices]
    else:
        # For raw data, we use list comprehension to keep the original array structure
        subsampled_data = [bin_data[i] for i, center in enumerate(bin_centers) if center in subsampled_centers]
    
    return subsampled_centers, subsampled_data