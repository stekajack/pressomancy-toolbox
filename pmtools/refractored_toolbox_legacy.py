import gzip
from itertools import product
import pickle
import pandas as pd
import os
import numpy as np
import random

def fetch_data(*args, base_dir, extension='.p.gz'):
    # check if any argument is a string and wrap it in a tuple if it is
    args = [(arg,) if isinstance(arg, str) else arg for arg in args]
    # generate all combinations of parameter values
    combinations_params = list(product(*args))
    # wrap the base_dir strings in a tuple to prevent decomposition
    base_dir = [(bd,) if isinstance(bd, str) else bd for bd in base_dir]
    # generate all combinations of base directory elements and join them with '/'
    combinations_base_dir = ['/'.join(map(str, x)) for x in product(*base_dir)]
    # create a list of all file paths
    paths_to_calc_with = [f"{base_dir}/{cdw}_{params_str}{extension}"
                          for base_dir in combinations_base_dir
                          for params_str in ['_'.join(map(str, x)) for x in combinations_params]
                          for cdw in ['custom_data_wip']]
    # iterate over all file paths and yield dip_3d and file path
    for file_path in paths_to_calc_with:
        try:
            with gzip.open(file_path, 'rb') as file_handle:
                favorite_color = pickle.load(file_handle)
                df = pd.DataFrame(favorite_color)
            yield df, file_path
        except BaseException as exception:
            print(f"An error occurred: {str(exception)}")
            print(file_path)
            yield pd.DataFrame(), file_path

def run_task(kernel, master_dict, parallel_param_id, parallel_param_val, kernel_kwargs={}):
    back_keys = ['what_am_I_looking_at', 'which_sim']
    master_dict[parallel_param_id] = (parallel_param_val,)

    dict_back = {k: v for k, v in master_dict.items() if k in back_keys}
    dict_front = {k: v for k, v in master_dict.items() if k not in back_keys}
    res = []
    for fr in product(*dict_front.values()):
        for ba in product(*dict_back.values()):
            interate_sim_data = fetch_data(*fr, base_dir=ba)
            res.append(kernel(interate_sim_data, **kernel_kwargs))
    return res

def check_data(master_dict, parallel_param_id, parallel_param_val):
    back_keys = ['what_am_I_looking_at', 'which_sim']
    master_dict[parallel_param_id] = (parallel_param_val,)

    dict_back = {k: v for k, v in master_dict.items() if k in back_keys}
    dict_front = {k: v for k, v in master_dict.items() if k not in back_keys}
    total_files, safe_files = 0, 0
    problems, unfinished = [], []
    for fr in product(*dict_front.values()):
        for ba in product(*dict_back.values()):
            data_iterator = fetch_data(*fr, base_dir=ba)
            while True:
                try:
                    df, data_path = next(data_iterator)
                    data_path_striped = data_path.replace(
                        'custom_data_wip', 'checkpoint')
                    data_path_striped = data_path_striped.replace('.p.gz', '')
                    total_files += 1
                    if len(df.columns.values) == 0 or len(os.listdir(data_path_striped)) == 0:
                        problems.append(data_path)
                    elif len(df.columns.values) != 1001:
                        unfinished.append(data_path)
                    else:
                        safe_files += 1
                except StopIteration:
                    break
    print('total_files ', total_files)
    print('safe_files ', safe_files)
    print('unfinished ', unfinished)
    print('problems ', problems)
    return unfinished

def convert_dict_to_list(array):

    def dict_to_list(elem):
        if isinstance(elem, dict):
            # if element is a dictionary, return a list of its values
            return list(elem.values())
        else:
            # if element is not a dictionary, return it as it is
            return elem

    # create a nested list of elements in the input array
    nested_list = array.tolist()

    # recursively traverse the nested list and apply the dict_to_list function to each element
    def traverse_and_apply(nested_list):
        new_list = []
        for elem in nested_list:
            if isinstance(elem, list):
                new_elem = traverse_and_apply(elem)
                new_list.append(new_elem)
            else:
                new_elem = dict_to_list(elem)
                new_list.append(new_elem)
        return new_list

    # apply the traverse_and_apply function to the nested list to get the new list
    new_list = traverse_and_apply(nested_list)

    # convert the new list back to a numpy array with the same shape and dtype as the input array
    new_array = np.array(new_list, dtype=array.dtype).reshape(array.shape)

    return new_array

def nodal_eff(g):

    weights = g.es["weight"][:]
    sp = (1.0 / np.array(g.shortest_paths_dijkstra(weights=weights)))
    np.fill_diagonal(sp, 0)
    N = sp.shape[0]
    ne = (1.0/(N-1)) * np.apply_along_axis(sum, 0, sp)

    return ne

def fold_coordinates(data_3d, box_dim=(47.13493067400575, 47.13493067400575, 94.2698613480115)):
    shape = np.shape(data_3d)
    dip_3d_folded = data_3d.copy()
    dip_3d_folded = [np.array(x) for x in dip_3d_folded]
    for ii, jj in product(range(shape[0]), range(shape[1])):
        while data_3d[ii][jj] >= box_dim[jj]:
            dip_3d_folded[ii][jj] -= box_dim[jj]
        while data_3d[ii][jj] < 0:
            dip_3d_folded[ii][jj] += box_dim[jj]
    return dip_3d_folded

def fold_coordinates_pp(data_pp, box_dim):
    """
    Fold value into primary interval.

    Parameters:
    -----------
    data_pp : 1D array consiting of (x,y,z) coords
    box_dim : np.array(1,3,dtype=float), optional
        Box dimensions are required to be able to fold PBC coordinates correctly.

    Returns:
    --------
    1D array consiting of (x,y,z) coords folded into [0, l)
    """
    data_pp = np.array(data_pp.copy())
    for ii, elem in enumerate(data_pp):
        while data_pp[ii] >= box_dim[ii]:
            data_pp[ii] -= box_dim[ii]
        while data_pp[ii] < 0:
            data_pp[ii] += box_dim[ii]
    return data_pp

def lj_pair_energy(epsilon, x, sigma=1., r_cut=2.5, c_shift=0.0):
    pot = 4.0 * epsilon * ((sigma / x)**12 - (sigma / x)**6) + c_shift
    return pot

def pair_potential(d, m1, m2):
    r2 = np.dot(d, d)
    r = np.sqrt(r2)
    r3 = r2 * r
    r5 = r3 * r2

    pe1 = np.dot(m1, m2)
    pe2 = np.dot(m1, d)
    pe3 = np.dot(m2, d)

    return pe1 / r3 - 3.0 * pe2 * pe3 / r5

def fene_potential(dist):
    k = 40.0
    eq_dist = 0.
    drmax = 3.0
    drmax2 = drmax**2
    drmax2i = 1/drmax2
    dr = dist - eq_dist
    return -0.5 * k * drmax2 * np.log(1.0 - dr * dr * drmax2i)

def contextual_shuffle(pivot_param, context_manager, pool_gather):
    # figure out what are the shape constranints that need to be adhered to.
    shape_axis = [x for x, y, in zip(
        context_manager.values(), context_manager.keys()) if y != pivot_param]
    spacing = np.prod(shape_axis)
    master_return, master_return_lbl = [], []
    # organise data per pivot parameter, meaning shuffle to a list of lists with the length of the pivot parameter.
    top_most = [[] for x in np.empty(context_manager[pivot_param], list)]
    for ii, top_handle in enumerate(top_most):
        for pool in pool_gather:
            top_most[ii].extend(pool[ii*spacing:spacing*(ii+1)])
    # traverse the data per pivot parameter. The point is to figure out what is the parallel parameter and shuffle so that output matches shape axis contraint.
    for pool in top_most:
        # goup data per parallel parameter and separate in key/value lists. tested only of simulation is the parallel parameter.
        _y = np.array([[pool[x].copy() for x in range(0+bb, len(pool), spacing)]
                       for bb in range(spacing)]).flatten()
        pool_reshaped = np.reshape(_y, shape_axis+[len(pool_gather)])
        pool_reshaped_lbl = np.empty(np.shape(pool_reshaped), object)
        for id_sim, sim_el in enumerate(pool_reshaped):
            shape = np.shape(sim_el)
            for ii, jj in product(range(shape[0]), range(shape[1])):
                pool_reshaped_lbl[id_sim][ii][jj] = sim_el[ii][jj].keys()
                pool_reshaped[id_sim][ii][jj] = sim_el[ii][jj].values()

        storage = np.empty(shape_axis, dtype=list)
        storage_lbl = np.empty(shape_axis, dtype=list)
        for ii, jj in product(range(shape_axis[0]), range(shape_axis[1])):
            storage[ii][jj] = []
            storage_lbl[ii][jj] = []

        for ii, jj in product(range(shape_axis[0]), range(shape_axis[1])):
            for id_sim in range(len(pool_gather)):
                storage[ii][jj].extend(pool_reshaped[ii][jj][id_sim])
                storage_lbl[ii][jj].extend(pool_reshaped_lbl[ii][jj][id_sim])

        master_return.append(storage)
        master_return_lbl.append(storage_lbl)
    return master_return, master_return_lbl

def calc_gyr_rad(positions):
    r_cm = np.mean(positions, axis=0)
    r_sub = positions - r_cm

    # Compute the gyration tensor elements
    xx = np.mean(r_sub[:, 0]**2)
    yy = np.mean(r_sub[:, 1]**2)
    zz = np.mean(r_sub[:, 2]**2)
    xy = np.mean(r_sub[:, 0]*r_sub[:, 1])
    xz = np.mean(r_sub[:, 0]*r_sub[:, 2])
    yz = np.mean(r_sub[:, 1]*r_sub[:, 2])

    # Construct the gyration tensor
    gyration_tensor_element = np.array(
        [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])

    res, egiv = np.linalg.eig(gyration_tensor_element)
    P_inv = np.linalg.inv(egiv)
    X = np.dot(P_inv, gyration_tensor_element)
    B = np.dot(X, egiv)
    return sum(np.diag(B))

def check_overlapp(cluster, cluster_list, crit):
    """
    Check for overlap between a cluster and a list of clusters.

    Args:
    - cluster: np.array, The cluster to check for overlap.
    - cluster_list: list of np.array, List of clusters to check against.
    - crit: float, The criterion for overlap.

    Returns:
    - bool, np.array: Boolean indicating if overlap is detected, and the original cluster.
    """
    for cluster_ext in cluster_list:
        indices_int = np.arange(cluster.shape[0])
        indices_ext = np.arange(cluster_ext.shape[0])
        index_combinations = np.array(list(product(indices_int, indices_ext)))
        distances = np.linalg.norm(
            cluster[index_combinations[:, 0]] - cluster_ext[index_combinations[:, 1]], axis=-1)
        if any(distances < crit):
            return True, cluster
    return False, cluster

def fill_box(list_of_clusters, boxvecs, max_clust_vect):
    """
    Fill a box with clusters while avoiding overlap.

    Args:
    - list_of_clusters: list of np.array, List of clusters to fit into the box.
    - boxvecs: np.array, Vector defining the box dimensions.

    Returns:
    - list of np.array: List of clusters shifted to fit into the box without overlap.
    """
    total_len = len(list_of_clusters)
    random_points = np.random.rand(len(list_of_clusters), 3) * boxvecs
    list_of_clusters_shifted = []

    for iid, cluster in enumerate(list_of_clusters):
        shifted_cluster = []
        while True:
            random_point = random_points[iid]
            filter_array = np.all(cluster + random_point <= boxvecs, axis=1)
            if all(filter_array):
                if iid > 0:
                    distances_rp = np.abs(random_points[:iid] - random_point)
                    filter_list = np.all(distances_rp < max_clust_vect, axis=1)
                    list_filtered = [list_of_clusters_shifted[i]
                                     for i in range(iid) if filter_list[i]]
                else:
                    list_filtered = []

                if list_filtered:
                    overlap_detected, shifted_cluster = check_overlapp(
                        cluster + random_point, list_filtered, 1.)
                    if overlap_detected:
                        random_points[iid] = np.random.rand(3) * boxvecs
                    else:
                        break
                else:
                    shifted_cluster = cluster + random_point
                    break
            else:
                random_points[iid] = np.random.rand(3) * boxvecs
        list_of_clusters_shifted.append(shifted_cluster)
    return list_of_clusters_shifted

def generate_unique_sublists(original_list, M, P):
    """
    Generate unique sublists from the original list.

    Args:
    - original_list: list, Original list to generate sublists from.
    - M: int, Number of elements per sublist.
    - P: int, Number of sublists.

    Returns:
    - list of lists: List of unique sublists.
    """
    sublists = []

    # Generate indices for the outer list
    indices = list(range(len(original_list)))

    for _ in range(P):
        # Randomly select M unique indices
        selected_indices = random.sample(indices, M)

        # Extract sublists using the selected indices
        selected_elements = [original_list[i] for i in selected_indices]
        sublists.append(selected_elements)

        # Remove selected indices
        indices = [i for i in indices if i not in selected_indices]

    return sublists
