import random
from itertools import product
import numpy as np
import igraph as ig
from itertools import product, combinations
import refractored_toolbox as context
# import vg
import os
from pressomancy.analysis import H5DataSelector
import h5py
# import pyscal as pc


def per_fil_mag(data_iterator, box_dim, chunk=(-50, -1), norm=1., crit=1.47):
    """
    Compute the per filament magnetization of a suspension dataset.

    Parameters:
    -----------
    interate_sim_data : iterator
        An iterator that yields pairs of pandas dataframes and data file paths.
        The dataframes should contain simulation data in a specific format.
    chunk : tuple, optional
        A tuple specifying the start and end indices of the columns to be used in the
        calculation of the magnetization. Default is (-50, -1).
    norm : float, optional
        A normalization factor for the magnetisation. Default is 1.

    Returns:
    --------
    per_sim_data : dict
        A dictionary that maps data file paths to the corresponding computed magnetizations.
    """
    data_with_context = {}
    while True:
        try:
            df, data_path = next(data_iterator)
            if len(df) == 0:
                data_with_context[data_path] = float('nan')

            else:
                magnetisation_per_frame = []
                start, end = chunk
                relevant_columns = df.columns.values[start:end]
                for elem in relevant_columns:
                    magnetisation_data = df[elem].apply(lambda x: x['dip'][-1])
                    mag_pf = np.mean(magnetisation_data)/float(norm)
                    magnetisation_per_frame.append(np.mean(mag_pf))
                data_with_context[data_path] = np.mean(
                    magnetisation_per_frame, axis=0)
        except StopIteration:
            break
    return data_with_context


def per_fil_gyr(data_iterator, box_dim, chunk=(-50, -1), norm=1., crit=1.47):
    """
    Compute the per filament gyration radius of a suspension dataset.

    Parameters:
    -----------
    interate_sim_data : iterator
        An iterator that yields pairs of pandas dataframes and data file paths.
        The dataframes should contain simulation data in a specific format.
    chunk : tuple, optional
        A tuple specifying the start and end indices of the columns to be used in the
        calculation of the magnetization. Default is (-50, -1).
    norm : float, optional
        A normalization factor for the gyration tensor. Default is 1.
    box_dim : np.array(1,3,dtype=float), optional
        Box dimensions are required to be able to handle PBC coordinates correctly per object,
        so that objects arent broken accross the box boundary!

    Returns:
    --------
    per_sim_data : dict
        A dictionary that maps data file paths to the corresponding computed gyration radii.
    """
    per_sim_data = {}
    while True:
        try:
            df, data_path = next(data_iterator)

            if len(df) == 0:
                per_sim_data[data_path] = float('nan')

            else:
                accumulated_per_timestep = []
                start, end = chunk
                relevant_columns = df.columns.values[start:end]
                for elem in relevant_columns:
                    posss = df[elem].apply(
                        lambda x: context.fold_coordinates_pp(x['pos'], box_dim=box_dim))
                    pf_indices = [list(range(x, x + 20))
                                  for x in range(0, len(posss), 20)]
                    edges = [(x, y) for pf_el in pf_indices for x,
                             y in zip(pf_el[:-1], pf_el[1:])]
                    g2 = ig.Graph(n=len(posss), edges=edges)
                    g2.vs["pos"] = posss
                    g2.simplify()

                    decomposition = g2.decompose()
                    master_list = []
                    for subgraph in decomposition:
                        flag, pass_graph = context.check_breakage(
                            subgraph, box_dim)
                        if not flag:
                            positions = context.unbreak_graph(
                                pass_graph, box_dim)
                        else:
                            positions = np.array(subgraph.vs['pos'])
                        master_list.append(context.calc_gyr_rad(positions))
                    accumulated_per_timestep.append(np.mean(master_list))
                per_sim_data[data_path] = accumulated_per_timestep
        except StopIteration:
            break
    return per_sim_data


def per_fil_gyr_dist(data_iterator, box_dim, chunk=(-50, -1), norm=1.):
    """
    Compute the per filament gyration radius of a suspension dataset.

    Parameters:
    -----------
    interate_sim_data : iterator
        An iterator that yields pairs of pandas dataframes and data file paths.
        The dataframes should contain simulation data in a specific format.
    chunk : tuple, optional
        A tuple specifying the start and end indices of the columns to be used in the
        calculation of the magnetization. Default is (-50, -1).
    norm : float, optional
        A normalization factor for the gyration tensor. Default is 1.
    box_dim : np.array(1,3,dtype=float), optional
        Box dimensions are required to be able to handle PBC coordinates correctly per object,
        so that objects arent broken accross the box boundary!

    Returns:
    --------
    per_sim_data : dict
        A dictionary that maps data file paths to the corresponding computed gyration radii.
    """
    per_sim_data = {}
    while True:
        try:
            df, data_path = next(data_iterator)

            if len(df) == 0:
                per_sim_data[data_path] = float('nan')

            else:

                gyration_tensor_full = []
                start, end = chunk
                for elem in df.columns.values[start:end]:
                    position_data = np.array(
                        [df[elem][x]['pos'] for x in df.index.values])
                    gyration_tensor_pf = []
                    for per_fil_pos in [context.unbreak_pbc_object(position_data[x: x + 20], box_dim)for x in range(0, len(position_data), 20)]:
                        r_cm = np.mean(per_fil_pos, axis=0)
                        r_sub = per_fil_pos - r_cm

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
                        gyration_tensor_pf.append(B)
                    giros_ptmpsim = [sum(np.diag(x))/norm**2
                                     for x in gyration_tensor_pf]
                    gyration_tensor_full.extend(giros_ptmpsim)
                per_sim_data[data_path] = gyration_tensor_full
        except StopIteration:
            break
    return per_sim_data


def transitivity_undirected(data_iterator, box_dim, chunk=(-50, -1), crit=1.3, en_crit=-0.1):
    per_sim_data = {}
    while True:
        try:
            df, data_path = next(data_iterator)
            if len(df) == 0:
                per_sim_data[data_path] = float('nan')

            else:
                accumulated_per_timestep = []
                start, end = chunk
                for elem in df.columns.values[start:end]:
                    dip_3d = [(idx, df[elem][x]['pos'], df[elem][x]['dip'])
                              for idx, x in enumerate(df.index.values)]
                    edges = [(x[0], y[0]) for x, y in combinations(dip_3d, 2) if np.linalg.norm(context.min_img_dist(
                        x[1], y[1], box_dim)) <= crit and context.pair_potential(context.min_img_dist(
                            x[1], y[1], box_dim), x[2], y[2]) < en_crit]
                    pf_ids = [dip_3d[x:x+20].copy()
                              for x in range(0, len(dip_3d), 20)]
                    for pf_ids_el in pf_ids:
                        edges.extend([(x[0], y[0]) for x, y in zip(
                            pf_ids_el[:-1], pf_ids_el[1:])])
                    g = ig.Graph(n=len(dip_3d), edges=edges)
                    g2 = g.copy()
                    g2.simplify()
                    accumulated_per_timestep.append(
                        g2.transitivity_avglocal_undirected(mode='nan'))
                    # there could be another mean missings
                # per_sim_data[data_path] = np.mean(accumulated_per_timestep, axis=0)
                per_sim_data[data_path] = accumulated_per_timestep

        except StopIteration:
            break
    return per_sim_data


def cluster_no(data_iterator, box_dim, chunk=(-50, -1), crit=1.3, en_crit=-0.1):
    per_sim_data = {}
    while True:
        try:
            df, data_path = next(data_iterator)
            if len(df) == 0:
                per_sim_data[data_path] = float('nan')

            else:
                accumulated_per_timestep = []
                start, end = chunk
                relevant_columns = df.columns.values[start:end]
                for elem in relevant_columns:
                    posss = df[elem].apply(
                        lambda x: context.fold_coordinates_pp(x['pos'], box_dim=box_dim))
                    sys = pc.System()
                    box_x, box_y, box_z = box_dim
                    sys.box = [
                        [box_x, 0.0, 0.0],
                        [0.0, box_y, 0.0],
                        [0.0, 0.0, box_z]]
                    sys.atoms = np.vectorize(lambda pos_el, id_el: pc.Atom(
                        pos=pos_el, id=id_el))(posss, np.arange(len(posss)))
                    sys.find_neighbors(method='cutoff', cutoff=crit)
                    edges = [(at.id, ng)
                             for at in sys.atoms for ng in at.neighbors]
                    g2 = ig.Graph(n=len(posss), edges=edges)
                    g2.vs["pos"] = posss
                    g2.simplify()
                    accumulated_per_timestep.append(len(g2.decompose()))
                per_sim_data[data_path] = accumulated_per_timestep

        except StopIteration:
            break
    return per_sim_data


def cluster_no_fancy(data_iterator, box_dim, chunk=(-50, -1), lj_eps=1.0):
    per_sim_data = {}
    while True:
        try:
            df, data_path = next(data_iterator)
            if len(df) == 0:
                per_sim_data[data_path] = float('nan')

            else:
                accumulated_per_timestep = []
                start, end = chunk
                for elem in df.columns.values[start:end]:
                    dip_3d = [(idx, df[elem][x]['pos'], df[elem][x]['dip'])
                              for idx, x in enumerate(df.index.values)]
                    edges, weights = [], []
                    for x, y in combinations(dip_3d, 2):
                        energy = 0
                        dist_vec = context.min_img_dist(x[1], y[1], box_dim)
                        energy += context.pair_potential(dist_vec, x[2], y[2])
                        dist = np.linalg.norm(dist_vec)
                        if dist <= 2.5:
                            lj_check = context.lj_pair_energy(lj_eps, dist)
                            energy += lj_check
                        if abs(energy) > 0.05:
                            edges.append((x[0], y[0]))
                            weights.append(abs(energy))

                    g2 = ig.Graph(n=len(dip_3d), edges=edges)
                    g2.es['weight'] = weights/max(weights)
                    g2.simplify(combine_edges=sum)
                    accumulated_per_timestep.append(len(g2.decompose()))
                per_sim_data[data_path] = accumulated_per_timestep

        except StopIteration:
            break
    return per_sim_data


def magn_cluster_core(data_iterator, box_dim, chunk=(-50, -1), crit=1.3, en_crit=-0.1, norm=1.0):
    per_sim_data = {}
    while True:
        try:
            df, data_path = next(data_iterator)
            if len(df) == 0:
                per_sim_data[data_path] = float('nan')

            else:
                submagn_per_tmstp = []
                start, end = chunk
                relevant_columns = df.columns.values[start:end]
                for elem in relevant_columns:
                    posss = df[elem].apply(
                        lambda x: context.fold_coordinates_pp(x['pos'], box_dim=box_dim))
                    momss = df[elem].apply(lambda x: x['dip'])
                    sys = pc.System()
                    box_x, box_y, box_z = box_dim
                    sys.box = [
                        [box_x, 0.0, 0.0],
                        [0.0, box_y, 0.0],
                        [0.0, 0.0, box_z]]
                    sys.atoms = np.vectorize(lambda pos_el, id_el: pc.Atom(
                        pos=pos_el, id=id_el))(posss, np.arange(len(posss)))
                    sys.find_neighbors(method='cutoff', cutoff=crit)
                    edges = [(at.id, ng)
                             for at in sys.atoms for ng in at.neighbors]
                    g2 = ig.Graph(n=len(posss), edges=edges)
                    g2.vs["dip"] = momss
                    g2.vs["pos"] = posss
                    g2.simplify()
                    submagn_per_tmstp.append(np.mean(
                        [np.mean(np.array(sub_graph.vs['dip'])[:, -1]) for sub_graph in g2.decompose()]))
                per_sim_data[data_path] = submagn_per_tmstp
        except StopIteration:
            break
    return per_sim_data


def efficiency_global(data_iterator, box_dim, chunk=(-50, -1), crit=1.3, en_crit=-0.1):
    per_sim_data = {}
    while True:
        try:
            df, data_path = next(data_iterator)
            if len(df) == 0:
                per_sim_data[data_path] = float('nan')

            else:
                accumulated_per_timestep = []
                start, end = chunk
                for elem in df.columns.values[start:end]:
                    dip_3d = [(idx, df[elem][x]['pos'], df[elem][x]['dip'])
                              for idx, x in enumerate(df.index.values)]
                    edges = [(x[0], y[0]) for x, y in combinations(dip_3d, 2) if np.linalg.norm(context.min_img_dist(
                        x[1], y[1], box_dim)) <= crit and context.pair_potential(context.min_img_dist(
                            x[1], y[1], box_dim), x[2], y[2]) < en_crit]
                    pf_ids = [dip_3d[x:x+20].copy()
                              for x in range(0, len(dip_3d), 20)]
                    for pf_ids_el in pf_ids:
                        edges.extend([(x[0], y[0]) for x, y in zip(
                            pf_ids_el[:-1], pf_ids_el[1:])])
                    g2 = ig.Graph(n=len(dip_3d), edges=edges)
                    g2.simplify()
                    g2.es['weight'] = np.ones(g2.ecount())
                    accumulated_per_timestep.append(np.mean(
                        [np.nanmean(context.nodal_eff(sub_graph))
                         for sub_graph in g2.decompose()]))
                    # there could be another mean missings
                per_sim_data[data_path] = np.mean(
                    accumulated_per_timestep, axis=0)
        except StopIteration:
            break
    return per_sim_data


def network_degree(data_iterator, box_dim, chunk=(-50, -1), crit=1.3, en_crit=-0.1):
    per_sim_data = {}
    while True:
        try:
            df, data_path = next(data_iterator)
            if len(df) == 0:
                per_sim_data[data_path] = float('nan')

            else:
                accumulated_per_timestep = []
                start, end = chunk
                for elem in df.columns.values[start:end]:
                    dip_3d = [(idx, df[elem][x]['pos'], df[elem][x]['dip'])
                              for idx, x in enumerate(df.index.values)]
                    edges = [(x[0], y[0]) for x, y in combinations(dip_3d, 2) if np.linalg.norm(context.min_img_dist(
                        x[1], y[1], box_dim)) <= crit and context.pair_potential(context.min_img_dist(
                            x[1], y[1], box_dim), x[2], y[2]) < en_crit]
                    pf_ids = [dip_3d[x:x+20].copy()
                              for x in range(0, len(dip_3d), 20)]
                    for pf_ids_el in pf_ids:
                        edges.extend([(x[0], y[0]) for x, y in zip(
                            pf_ids_el[:-1], pf_ids_el[1:])])
                    g2 = ig.Graph(n=len(dip_3d), edges=edges)
                    g2.simplify()
                    accumulated_per_timestep.extend(g2.degree())
                    # there could be another mean missings
                per_sim_data[data_path] = accumulated_per_timestep
        except StopIteration:
            break
    return per_sim_data


def network_degree_fancy(data_iterator, box_dim, chunk=(-50, -1), crit=1.3, en_crit=-0.1):
    per_sim_data = {}
    while True:
        try:
            df, data_path = next(data_iterator)
            if len(df) == 0:
                per_sim_data[data_path] = float('nan')

            else:
                accumulated_per_timestep = []
                start, end = chunk
                relevant_columns = df.columns.values[start:end]
                for elem in relevant_columns:
                    posss = df[elem].apply(
                        lambda x: context.fold_coordinates_pp(x['pos'], box_dim=box_dim))
                    sys = pc.System()
                    box_x, box_y, box_z = box_dim
                    sys.box = [
                        [box_x, 0.0, 0.0],
                        [0.0, box_y, 0.0],
                        [0.0, 0.0, box_z]]
                    sys.atoms = np.vectorize(lambda pos_el, id_el: pc.Atom(
                        pos=pos_el, id=id_el))(posss, np.arange(len(posss)))
                    sys.find_neighbors(method='cutoff', cutoff='sann')
                    edges = [(at.id, ng)
                             for at in sys.atoms for ng in at.neighbors]
                    g2 = ig.Graph(n=len(posss), edges=edges)
                    g2.simplify()
                    accumulated_per_timestep.extend(g2.degree())
                    # there could be another mean missings
                per_sim_data[data_path] = accumulated_per_timestep
        except StopIteration:
            break
    return per_sim_data


def density_profle(data_iterator, box_dim, chunk=(-50, -1), crit=1.47, en_crit=-0.1):
    """
    Compute the per cluster density profile of a suspension dataset.

    Parameters:
    -----------
    interate_sim_data : iterator
        An iterator that yields pairs of pandas dataframes and data file paths.
        The dataframes should contain simulation data in a specific format.
    chunk : tuple, optional
        A tuple specifying the start and end indices of the columns to be used in the
        calculation of the magnetization. Default is (-50, -1).
    norm : float, optional
        A normalization factor for the gyration tensor. Default is 1.
    box_dim : np.array(1,3,dtype=float), optional
        Box dimensions are required to be able to handle PBC coordinates correctly per object,
        so that objects arent broken accross the box boundary!

    Returns:
    --------
    per_sim_data : dict
        A dictionary that maps data file paths to the corresponding computed density profiled in a list of lists for each timestep. Not that the first element ofeach sublist is the number of clusters, which is to be used to normalise the data in postprocessing.
    """
    per_sim_data = {}
    while True:
        try:
            df, data_path = next(data_iterator)
            if len(df) == 0:
                per_sim_data[data_path] = float('nan')

            else:
                accumulated_per_timestep = []
                start, end = chunk
                relevant_columns = df.columns.values[start:end]
                for elem in relevant_columns:
                    posss = df[elem].apply(
                        lambda x: context.fold_coordinates_pp(x['pos'], box_dim=box_dim))
                    momss = df[elem].apply(lambda x: x['dip'])
                    sys = pc.System()
                    box_x, box_y, box_z = box_dim
                    sys.box = [
                        [box_x, 0.0, 0.0],
                        [0.0, box_y, 0.0],
                        [0.0, 0.0, box_z]]
                    sys.atoms = np.vectorize(lambda pos_el, id_el: pc.Atom(
                        pos=pos_el, id=id_el))(posss, np.arange(len(posss)))
                    sys.find_neighbors(method='cutoff', cutoff='sann')
                    edges = [(at.id, ng)
                             for at in sys.atoms for ng in at.neighbors]
                    g2 = ig.Graph(n=len(posss), edges=edges)
                    g2.vs["dip"] = momss
                    g2.vs["pos"] = posss
                    g2.simplify()

                    decomposition = g2.decompose()
                    master_list = [len(decomposition),]
                    for subgraph in decomposition:
                        flag, pass_graph = context.check_breakage(
                            subgraph, box_dim)
                        if not flag:
                            positions = context.unbreak_graph(
                                pass_graph, box_dim)
                        else:
                            positions = np.array(subgraph.vs['pos'])
                        r_cm = np.mean(positions, axis=0)
                        distances = positions-r_cm
                        master_list.extend(distances)
                    accumulated_per_timestep.append(master_list)
                    # there could be another mean missings
                per_sim_data[data_path] = accumulated_per_timestep
        except StopIteration:
            break
    return per_sim_data


def get_clusters(data_iterator, box_dim, chunk=(-50, -1), crit=1.47):

    per_sim_data = {}
    while True:
        try:
            df, data_path = next(data_iterator)
            if len(df) == 0:
                per_sim_data[data_path] = float('nan')

            else:
                accumulated_per_timestep = []
                start, end = chunk
                relevant_columns = df.columns.values[start:end]
                for elem in relevant_columns:
                    posss = df[elem].apply(
                        lambda x: context.fold_coordinates_pp(x['pos'], box_dim=box_dim))
                    sys = pc.System()
                    box_x, box_y, box_z = box_dim
                    sys.box = [
                        [box_x, 0.0, 0.0],
                        [0.0, box_y, 0.0],
                        [0.0, 0.0, box_z]]
                    sys.atoms = np.vectorize(lambda pos_el, id_el: pc.Atom(
                        pos=pos_el, id=id_el))(posss, np.arange(len(posss)))
                    sys.find_neighbors(method='cutoff', cutoff='sann')
                    edges = [(at.id, ng)
                             for at in sys.atoms for ng in at.neighbors]

                    g2 = ig.Graph(n=len(posss), edges=edges)
                    g2.vs["pos"] = posss
                    g2.simplify()

                    decomposition = g2.decompose()
                    master_list = []
                    for subgraph in decomposition:
                        flag, pass_graph = context.check_breakage(
                            subgraph, box_dim)
                        if not flag:
                            positions = context.unbreak_graph(
                                pass_graph, box_dim)
                        else:
                            positions = np.array(subgraph.vs['pos'])
                        master_list.append(
                            positions-np.mean(positions, axis=0))
                    accumulated_per_timestep.append(master_list)
                per_sim_data[data_path] = accumulated_per_timestep
        except StopIteration:
            break
    return per_sim_data


def density_visual(data_iterator, box_dim, chunk=(-50, -1), crit=1.47, en_crit=-0.1):

    vol = 4/3*.5**3*np.pi
    per_sim_data = {}
    while True:
        try:
            df, data_path = next(data_iterator)
            if len(df) == 0:
                per_sim_data[data_path] = float('nan')
            else:
                accumulated_per_timestep = []
                start, end = chunk
                relevant_columns = df.columns.values[start:end]
                for elem in relevant_columns:
                    posss = df[elem].apply(
                        lambda x: context.fold_coordinates_pp(x['pos'], box_dim=box_dim))
                    momss = df[elem].apply(lambda x: x['dip'])

                    sys = pc.System()
                    box_x, box_y, box_z = box_dim
                    sys.box = [
                        [box_x, 0.0, 0.0],
                        [0.0, box_y, 0.0],
                        [0.0, 0.0, box_z]]
                    sys.atoms = np.vectorize(lambda pos_el, id_el: pc.Atom(
                        pos=pos_el, id=id_el))(posss, np.arange(len(posss)))
                    sys.find_neighbors(method='cutoff', cutoff=crit)
                    edges = [(at.id, ng)
                             for at in sys.atoms for ng in at.neighbors]
                    g2 = ig.Graph(n=len(posss), edges=edges)
                    g2.vs["dip"] = momss
                    g2.vs["pos"] = posss
                    g2.vs["ng"] = [at.neighbors for at in sys.atoms]

                    g2.simplify()
                    decomposition = g2.decompose(minelements=2)
                    for subgraph in decomposition:
                        flag, pass_graph = context.check_breakage(
                            subgraph, crit)
                        if not flag:
                            positions = context.unbreak_graph(pass_graph)
                        else:
                            positions = np.array(subgraph.vs['pos'])
                        r_cm = np.mean(positions, axis=0)
                        distances = positions-r_cm
                        komsije = np.array(
                            [len(vrs['ng'])/vol for vrs in subgraph.vs]).reshape(-1, 1)
                        returner = np.hstack((distances, komsije))
                        accumulated_per_timestep.extend(returner)

                per_sim_data[data_path] = accumulated_per_timestep
        except StopIteration:
            break
    return per_sim_data


def density_visual_onfoot(data_iterator, box_dim, chunk=(-50, -1), crit=1.47, en_crit=-0.1):
    """
    Compute the per cluster density profile of a suspension dataset.

    Parameters:
    -----------
    interate_sim_data : iterator
        An iterator that yields pairs of pandas dataframes and data file paths.
        The dataframes should contain simulation data in a specific format.
    chunk : tuple, optional
        A tuple specifying the start and end indices of the columns to be used in the
        calculation of the magnetization. Default is (-50, -1).
    norm : float, optional
        A normalization factor for the gyration tensor. Default is 1.
    box_dim : np.array(1,3,dtype=float), optional
        Box dimensions are required to be able to handle PBC coordinates correctly per object,
        so that objects arent broken accross the box boundary!

    Returns:
    --------
    per_sim_data : dict
        A dictionary that maps data file paths to the corresponding computed density profiled in a list of lists for each timestep. Not that the first element ofeach sublist is the number of clusters, which is to be used to normalise the data in postprocessing.
    """
    vol = 4/3*crit**3*np.pi
    per_sim_data = {}
    while True:
        try:
            df, data_path = next(data_iterator)
            if len(df) == 0:
                per_sim_data[data_path] = float('nan')

            else:
                accumulated_per_timestep = []
                start, end = chunk
                for elem in df.columns.values[start:end]:
                    dip_3d = np.array([(idx, df[elem][x]['pos'], df[elem][x]['dip'])
                                       for idx, x in enumerate(df.index.values)], dtype=tuple)

                    idld, posss, momss = dip_3d.T

                    index_combinations = np.array(list(combinations(idld, 2)))
                    x, y = np.array(list(combinations(posss, 2))
                                    ).transpose((1, 0, 2))
                    distances = np.linalg.norm(
                        context.min_img_dist(x, y, box_dim), axis=-1)
                    filter_crit = distances < crit
                    edges = index_combinations[filter_crit]

                    g2 = ig.Graph(n=len(dip_3d), edges=edges)
                    g2.vs["pos"] = dip_3d[:, 1]
                    g2.vs["dip"] = dip_3d[:, 2]

                    g2.simplify()

                    decomposition = g2.decompose()
                    master_list = [len(decomposition),]
                    for subgraph in decomposition:
                        positions = context.unbreak_pbc_object(
                            subgraph.vs['pos'], box_dim)
                        r_cm = np.mean(positions, axis=0)
                        distances = positions-r_cm
                        komsije = np.array(
                            [len(vrs.neighbors())/vol for vrs in subgraph.vs]).reshape(-1, 1)
                        returner = np.hstack((distances, komsije))
                        accumulated_per_timestep.extend(returner)

                    # there could be another mean missings
                per_sim_data[data_path] = accumulated_per_timestep
        except StopIteration:
            break
    return per_sim_data


def density_profle_alt(data_iterator, box_dim, chunk=(-50, -1), crit=1.47, en_crit=-0.1):

    vol_pp = 4/3*0.5**3*np.pi
    per_sim_data = {}
    while True:
        try:
            df, data_path = next(data_iterator)
            if len(df) == 0:
                per_sim_data[data_path] = float('nan')

            else:
                accumulated_per_timestep = [[], []]
                start, end = chunk
                for elem in df.columns.values[start:end]:
                    dip_3d = [(idx, context.fold_coordinates_pp(df[elem][x]['pos'], box_dim=box_dim), df[elem][x]['dip'])
                              for idx, x in enumerate(df.index.values)]
                    idld, posss, momss = np.array(dip_3d, dtype=tuple).T

                    sys = pc.System()
                    box_x, box_y, box_z = box_dim
                    sys.box = [
                        [box_x, 0.0, 0.0],
                        [0.0, box_y, 0.0],
                        [0.0, 0.0, box_z]]
                    sys.atoms = [pc.Atom(pos=pos_el, id=id_el)
                                 for id_el, pos_el, dip_el in dip_3d]
                    sys.find_neighbors(method='cutoff', cutoff=crit)
                    edges = [(at.id, ng)
                             for at in sys.atoms for ng in at.neighbors]
                    g2 = ig.Graph(n=len(dip_3d), edges=edges)
                    g2.vs["dip"] = momss
                    g2.vs["pos"] = posss
                    g2.simplify()

                    decomposition = g2.decompose()
                    for subgraph in decomposition:
                        positions = context.unbreak_pbc_object(
                            subgraph.vs['pos'], box_dim)
                        r_cm = np.mean(positions, axis=0)
                        hh = np.linalg.norm(positions-r_cm, axis=-1)
                        hist, bin_edges = np.histogram(hh, bins=int(
                            (box_dim[0]*0.5)/0.25), range=(0, box_dim[0]*0.5), density=False)
                        rho_norm = 0.64/vol_pp
                        edgewidth = np.abs(bin_edges[1]-bin_edges[0])
                        r = bin_edges[:-1]
                        shell_vols = (4./3.)*np.pi*((r+edgewidth)**3 - r**3)
                        shell_rho = hist/shell_vols
                        rho = shell_rho/sum(shell_rho)
                        accumulated_per_timestep[0].append(rho)
                        accumulated_per_timestep[1].append(r)
                    # there could be another mean missings
                per_sim_data[data_path] = accumulated_per_timestep
        except StopIteration:
            break
    return per_sim_data


def calc_rdf(data_iterator, box_dim, chunk=(-50, -1), crit=1.3, en_crit=-0.1):
    per_sim_data = {}
    while True:
        try:
            df, data_path = next(data_iterator)
            if len(df) == 0:
                per_sim_data[data_path] = float('nan')

            else:
                accumulated_per_timestep = []
                start, end = chunk
                for elem in df.columns.values[start:end]:
                    dip_3d = [(idx, context.fold_coordinates_pp(df[elem][x]['pos'], box_dim=box_dim), df[elem][x]['dip'])
                              for idx, x in enumerate(df.index.values)]
                    sys = pc.System()
                    box_x, box_y, box_z = box_dim
                    sys.box = [
                        [box_x, 0.0, 0.0],
                        [0.0, box_y, 0.0],
                        [0.0, 0.0, box_z]]
                    sys.atoms = [pc.Atom(pos=pos_el, id=id_el)
                                 for id_el, pos_el, dip_el in dip_3d]
                    rdf = sys.calculate_rdf(
                        histobins=int((box_dim[0]*0.5)/0.25), histomax=box_dim[0]*0.5)
                    accumulated_per_timestep.append(rdf)
                per_sim_data[data_path] = accumulated_per_timestep
        except StopIteration:
            break
    return per_sim_data


def isomorph_core(data_iterator, chunk=(-50, -1), crit=1.3, en_crit=-0.1):
    per_sim_data = {}
    while True:
        try:
            df, data_path = next(data_iterator)
            if len(df) == 0:
                per_sim_data[data_path] = float('nan')
            else:
                print('len(df) ', len(df.columns.values))
                submagn_per_tmstp = []
                start, end = chunk
                for elem in df.columns.values[start:end]:
                    dip_3d = [(idx, context.fold_coordinates_pp(df[elem][x]['pos']), df[elem][x]['dip'])
                              for idx, x in enumerate(df.index.values)]
                    edges = [(x[0], y[0]) for x, y in combinations(dip_3d, 2) if np.linalg.norm(x[1]-y[1])
                             <= crit and context.pair_potential(x[1]-y[1], x[2], y[2]) < en_crit]
                    pf_ids = [dip_3d[x:x+20].copy()
                              for x in range(0, len(dip_3d), 20)]
                    for pf_ids_el in pf_ids:
                        edges.extend([(x[0], y[0]) for x, y in zip(
                            pf_ids_el[:-1], pf_ids_el[1:])])
                    idld, posss, momss = np.array(dip_3d, dtype=tuple).T
                    g2 = ig.Graph(n=len(dip_3d), edges=edges)
                    g2.vs["dip"] = momss
                    g2.simplify()
                    master_list = []
                    for subgraph in g2.decompose():
                        is_new_list = True
                        for sublist in master_list:
                            # if subgraph.isomorphic(sublist[0]):
                            if len(subgraph.vs) == len(sublist[0].vs):
                                sublist.append(subgraph)
                                is_new_list = False
                                break
                        if is_new_list:
                            master_list.append([subgraph])
                    submagn_per_tmstp.append(master_list)
                per_sim_data[data_path] = submagn_per_tmstp
        except StopIteration:
            break
    return per_sim_data


def angle_dist(data_iterator, box_dim, chunk=(-50, -1), crit=1.3, en_crit=-0.1):
    per_sim_data = {}
    reference = np.array((0, 0, 1))
    while True:
        try:
            df, data_path = next(data_iterator)
            if len(df) == 0:
                per_sim_data[data_path] = float('nan')

            else:
                accumulated_per_timestep = []
                start, end = chunk
                relevant_columns = df.columns.values[start:end]
                for elem in relevant_columns:

                    posss = df[elem].apply(
                        lambda x: context.fold_coordinates_pp(x['pos'], box_dim=box_dim))
                    momss = df[elem].apply(lambda x: x['dip'])
                    idld = np.arange(len(posss))

                    # create connections for a network based on genral criteria
                    sys = pc.System()
                    box_x, box_y, box_z = box_dim
                    sys.box = [
                        [box_x, 0.0, 0.0],
                        [0.0, box_y, 0.0],
                        [0.0, 0.0, box_z]]
                    sys.atoms = np.vectorize(lambda pos_el, id_el: pc.Atom(
                        pos=pos_el, id=id_el))(posss, idld)
                    sys.find_neighbors(method='cutoff', cutoff='sann')

                    edges = [(at.id, ng)
                             for at in sys.atoms for ng in at.neighbors]
                    graph_clusters = ig.Graph(n=len(posss), edges=edges)
                    graph_clusters.vs["pos"] = posss
                    graph_clusters.vs["name"] = idld
                    graph_clusters.simplify()

                    angles_per_cluster = []
                    for cluster in graph_clusters.decompose():
                        flag, pass_graph = context.check_breakage(
                            cluster, box_dim)
                        if not flag:
                            positions = context.unbreak_graph(
                                pass_graph, box_dim)
                        else:
                            positions = np.array(cluster.vs['pos'])

                        r_cm_cluster = np.mean(positions, axis=0)

                        for chain_idx in idld.reshape(-1, 20):
                            if any(idx in cluster.vs['name'] for idx in chain_idx):
                                relevant_vertices_pos = cluster.vs.select(
                                    lambda vertex: vertex['name'] in chain_idx)['pos']
                                r_cm = np.mean(relevant_vertices_pos, axis=0)
                                axis_1 = r_cm-r_cm_cluster
                                r_sub = relevant_vertices_pos - r_cm
                                xx = np.mean(r_sub[:, 0]**2)
                                yy = np.mean(r_sub[:, 1]**2)
                                zz = np.mean(r_sub[:, 2]**2)
                                xy = np.mean(r_sub[:, 0]*r_sub[:, 1])
                                xz = np.mean(r_sub[:, 0]*r_sub[:, 2])
                                yz = np.mean(r_sub[:, 1]*r_sub[:, 2])

                                # Construct the gyration tensor
                                gyration_tensor_element = np.array(
                                    [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
                                res, egiv = np.linalg.eig(
                                    gyration_tensor_element)
                                angle_deg = vg.angle(
                                    vg.aligned_with(
                                        axis_1, reference, reverse=False),
                                    vg.aligned_with(
                                        egiv[:, 0],
                                        reference, reverse=False),
                                    units='deg')
                                angles_per_cluster.append(
                                    np.minimum(angle_deg, 180 - angle_deg))

                    accumulated_per_timestep.append(angles_per_cluster)

                per_sim_data[data_path] = accumulated_per_timestep
        except StopIteration:
            break
    return per_sim_data


def cluster_gyr(data_iterator, box_dim, chunk=(-50, -1), crit=1.3, en_crit=-0.1):
    per_sim_data = {}
    while True:
        try:
            df, data_path = next(data_iterator)
            if len(df) == 0:
                per_sim_data[data_path] = float('nan')

            else:
                accumulated_per_timestep = []
                start, end = chunk
                relevant_columns = df.columns.values[start:end]
                for elem in relevant_columns:
                    posss = df[elem].apply(
                        lambda x: context.fold_coordinates_pp(x['pos'], box_dim=box_dim))

                    sys = pc.System()
                    box_x, box_y, box_z = box_dim
                    sys.box = [
                        [box_x, 0.0, 0.0],
                        [0.0, box_y, 0.0],
                        [0.0, 0.0, box_z]]
                    sys.atoms = np.vectorize(lambda pos_el, id_el: pc.Atom(
                        pos=pos_el, id=id_el))(posss, np.arange(len(posss)))
                    sys.find_neighbors(method='cutoff', cutoff='sann')
                    edges = [(at.id, ng)
                             for at in sys.atoms for ng in at.neighbors]
                    g2 = ig.Graph(n=len(posss), edges=edges)
                    g2.vs["pos"] = posss
                    g2.simplify()

                    decomposition = g2.decompose()
                    master_list = []
                    for subgraph in decomposition:
                        flag, pass_graph = context.check_breakage(
                            subgraph, box_dim)

                        if not flag:
                            positions = context.unbreak_graph(
                                pass_graph, box_dim)

                        else:
                            positions = np.array(subgraph.vs['pos'])

                        master_list.append(context.calc_gyr_rad(positions))
                    accumulated_per_timestep.append(np.mean(master_list))
                per_sim_data[data_path] = accumulated_per_timestep
        except StopIteration:
            break
    return per_sim_data


def magn_angle_dist(data_iterator, box_dim, chunk=(-50, -1), crit=1.3, en_crit=-0.1):
    per_sim_data = {}
    reference = np.array((0, 0, 1))
    while True:
        try:
            df, data_path = next(data_iterator)
            if len(df) == 0:
                per_sim_data[data_path] = float('nan')

            else:
                accumulated_per_timestep = []
                start, end = chunk
                relevant_columns = df.columns.values[start:end]
                for elem in relevant_columns:

                    posss = df[elem].apply(
                        lambda x: context.fold_coordinates_pp(x['pos'], box_dim=box_dim))
                    momss = df[elem].apply(lambda x: x['dip'])
                    idld = np.arange(len(posss))

                    # create connections for a network based on genral criteria
                    sys = pc.System()
                    box_x, box_y, box_z = box_dim
                    sys.box = [
                        [box_x, 0.0, 0.0],
                        [0.0, box_y, 0.0],
                        [0.0, 0.0, box_z]]
                    sys.atoms = np.vectorize(lambda pos_el, id_el: pc.Atom(
                        pos=pos_el, id=id_el))(posss, idld)
                    sys.find_neighbors(method='cutoff', cutoff='sann')

                    edges = [(at.id, ng)
                             for at in sys.atoms for ng in at.neighbors]
                    graph_clusters = ig.Graph(n=len(posss), edges=edges)
                    graph_clusters.vs["pos"] = posss
                    graph_clusters.vs["name"] = idld
                    graph_clusters.simplify()
                    # make a graph where filaments are considered as independant clusters!
                    edges_fils = [(x, y) for pf_el in idld.reshape(-1, 20)
                                  for x, y in zip(pf_el[:-1], pf_el[1:])]
                    graph_filamets = ig.Graph(n=len(posss), edges=edges_fils)
                    graph_filamets.vs["pos"] = posss
                    graph_filamets.vs["dip"] = momss
                    graph_filamets.vs["name"] = idld
                    graph_filamets.simplify()

                    angles_per_cluster = []
                    for cluster in graph_clusters.decompose():
                        for chain in graph_filamets.decompose():
                            if any(v['name'] in cluster.vs['name'] for v in chain.vs):
                                flag, pass_graph = context.check_breakage(
                                    chain, box_dim)

                                if not flag:
                                    positions = context.unbreak_graph(
                                        pass_graph, box_dim)

                                else:
                                    positions = np.array(chain.vs['pos'])

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
                                res, egiv = np.linalg.eig(
                                    gyration_tensor_element)
                                data_mag = np.array([vs['dip']
                                                    for vs in chain.vs])
                                magn_chain_tmstp = np.mean(data_mag, axis=0)
                                angle_deg = vg.angle(
                                    magn_chain_tmstp,
                                    vg.aligned_with(
                                        egiv[:, 0],
                                        reference, reverse=False),
                                    units='deg')
                                # angle_deg = vg.angle(magn_chain_tmstp, egiv[:, 0], units='deg')
                                angles_per_cluster.append(
                                    np.minimum(angle_deg, 180 - angle_deg))
                                # angles_per_cluster.append(vg.angle(magn_chain_tmstp, egiv[:, 0], units='deg'))
                    accumulated_per_timestep.extend(angles_per_cluster)

                per_sim_data[data_path] = accumulated_per_timestep
        except StopIteration:
            break
    return per_sim_data


def fully_connected_undirected_network(data_iterator, box_dim, chunk=(-50, -1), lj_eps=1.0):
    per_sim_data = {}
    while True:
        try:
            df, data_path = next(data_iterator)
            if len(df) == 0:
                per_sim_data[data_path] = float('nan')

            else:
                accumulated_per_timestep = []
                start, end = chunk
                for elem in df.columns.values[start:end]:
                    dip_3d = [(idx, df[elem][x]['pos'], df[elem][x]['dip'])
                              for idx, x in enumerate(df.index.values)]
                    # custom graph construction argument, zeeman is excluded, themal is deterrent, so only central and dip-dip globaly. Bonded parts have a weight adjusted by the fene contribution, so highly complex intormaiton probagation with connections that ahve different importance and "bandwidt"
                    edges, weights = [], []
                    for x, y in combinations(dip_3d, 2):
                        energy = 0
                        dist_vec = context.min_img_dist(x[1], y[1], box_dim)
                        energy += context.pair_potential(dist_vec, x[2], y[2])
                        dist = np.linalg.norm(dist_vec)
                        if dist <= 2.5:
                            lj_check = context.lj_pair_energy(lj_eps, dist)
                            energy += lj_check
                        if energy < -0.01:
                            edges.append((x[0], y[0]))
                            weights.append(abs(energy))

                    g2 = ig.Graph(n=len(dip_3d), edges=edges)
                    g2.es['weight'] = weights/max(weights)
                    g2.simplify(combine_edges=sum)
                    haj = context.nodal_eff(g2)
                    accumulated_per_timestep.append(np.mean(haj))
                    # there could be another mean missings
                per_sim_data[data_path] = accumulated_per_timestep
        except StopIteration:
            break
    return per_sim_data


def writevtk(data_iterator, box_dim, _path=None, chunk=(-50, -1),):
    pos_tmstp, dip_tmstp = [], []
    while True:
        counter = 0
        try:
            df, data_path = next(data_iterator)
            if len(df) == 0:
                pos_tmstp, dip_tmstp = [], []

            else:
                pos_tmstp, dip_tmstp = [], []
                start, end = chunk
                for elem in df.columns.values[start:end]:
                    pos_tmstp.append(
                        [context.fold_coordinates_pp(
                            df[elem][x]['pos'],
                            box_dim=box_dim)
                         for x in df.index.values])
                    dip_tmstp.append([df[elem][x]['dip']
                                     for x in df.index.values])
                os.mkdir(path=_path+'/'+data_path.split('/')[-1][:-5])
                for pos, directors in zip(pos_tmstp, dip_tmstp):
                    with open(_path+'/'+data_path.split('/')[-1][:-5]+'/'+str(counter)+'.vtk', 'w') as vtk:
                        vtk.write("# vtk DataFile Version 2.0\n")
                        vtk.write("particles\n")
                        vtk.write("ASCII\n")
                        vtk.write("DATASET UNSTRUCTURED_GRID\n")
                        vtk.write("POINTS {} floats\n".format(len(pos)))
                        for i in range(len(pos)):
                            vtk.write("%f %f %f\n" %
                                      (pos[i][0], pos[i][1], pos[i][2]))

                        vtk.write("POINT_DATA {}\n".format(len(pos)))
                        vtk.write("SCALARS dipoles float 3\n")
                        vtk.write("LOOKUP_TABLE default\n")
                        for i in range(len(directors)):
                            vtk.write("%f %f %f\n" % (
                                directors[i][0], directors[i][1], directors[i][2]))
                    counter += 1

        except StopIteration:
            print('done')
            break
    return 0

def write_vtk_frame_modern(data_path, path_target='path', frame=0):
    data_file=h5py.File(data_path, "r")
    data=H5DataSelector(data_file ,particle_group="Filament")
    data_per_fram=data.timestep[frame]
    positions=data_per_fram.pos_folded
    dipoles=data_per_fram.dip
    with open(path_target, 'w') as vtk:
        vtk.write("# vtk DataFile Version 2.0\n")
        vtk.write("particles\n")
        vtk.write("ASCII\n")
        vtk.write("DATASET UNSTRUCTURED_GRID\n")
        vtk.write("POINTS {} floats\n".format(len(positions)))
        for i in range(len(positions)):
            vtk.write("%f %f %f\n" %
                        (positions[i][0], positions[i][1], positions[i][2]))

        vtk.write("POINT_DATA {}\n".format(len(positions)))
        vtk.write("SCALARS dipoles float 3\n")
        vtk.write("LOOKUP_TABLE default\n")
        for i in range(len(dipoles)):
            vtk.write("%f %f %f\n" % (
                dipoles[i][0], dipoles[i][1], dipoles[i][2]))
    return 0


def energies_pc_with_time(data_iterator, box_dim, chunk=(-50, -1), crit=1.3, en_crit=-0.1, lj_eps=1.0, ext_fld=1.0):
    per_sim_data = {}
    while True:
        try:
            df, data_path = next(data_iterator)
            if len(df) == 0:
                per_sim_data[data_path] = float('nan')

            else:
                lj_energy_pt, dip_energy_pt, zeeman_energy_pt = [], [], []
                start, end = chunk
                for elem in df.columns.values[start:end]:
                    dip_3d = [(idx, df[elem][x]['pos'], df[elem][x]['dip'])
                              for idx, x in enumerate(df.index.values)]
                    idld, posss, momss = np.array(dip_3d, dtype=tuple).T
                    edges = [(x[0], y[0]) for x, y in combinations(dip_3d, 2) if np.linalg.norm(context.min_img_dist(
                        x[1], y[1], box_dim)) <= crit and context.pair_potential(context.min_img_dist(
                            x[1], y[1], box_dim), x[2], y[2]) < en_crit]
                    pf_ids = [dip_3d[x:x+20].copy()
                              for x in range(0, len(dip_3d), 20)]
                    for pf_ids_el in pf_ids:
                        edges.extend([(x[0], y[0]) for x, y in zip(
                            pf_ids_el[:-1], pf_ids_el[1:])])
                    g2 = ig.Graph(n=len(dip_3d), edges=edges)
                    g2.vs["name"] = idld
                    g2.vs["pos"] = posss
                    g2.vs["dip"] = momss
                    g2.simplify()

                    lj_energy_pc, dip_energy_pc, zeeman_energy_pc = [], [], []
                    for cluster in g2.decompose():
                        dip_energy_pc.append(np.mean([context.pair_potential(context.min_img_dist(
                            ii_hndl['pos'], jj_hndl['pos'], box_dim), ii_hndl['dip'], jj_hndl['dip']) for ii_hndl, jj_hndl in combinations(cluster.vs, 2)]))
                        lj_energy_pc.append(np.mean([context.lj_pair_energy(lj_eps, np.linalg.norm(context.min_img_dist(
                            ii_hndl['pos'], jj_hndl['pos'], box_dim))) for ii_hndl, jj_hndl in combinations(cluster.vs, 2)]))
                        zeeman_energy_pc.append(
                            np.mean([-1.0 * np.dot(np.array([0, 0, ext_fld]), ii_hndl['dip']) for ii_hndl in cluster.vs]))
                        # -1.0 * m_field * p.calc_dip()

                    lj_energy_pt.append(np.mean(lj_energy_pc))
                    dip_energy_pt.append(np.mean(dip_energy_pc))
                    zeeman_energy_pt.append(np.mean(zeeman_energy_pc))
                per_sim_data[data_path] = (
                    lj_energy_pt, dip_energy_pt, zeeman_energy_pt)

        except StopIteration:
            break
    return per_sim_data


def energies_tot_with_time(data_iterator, box_dim, chunk=(-50, -1), crit=1.3, en_crit=-0.1, lj_eps=1.0, ext_fld=1.0):
    per_sim_data = {}
    while True:
        try:
            df, data_path = next(data_iterator)
            if len(df) == 0:
                per_sim_data[data_path] = float('nan')

            else:
                lj_energy_pt, dip_energy_pt, zeeman_energy_pt = [], [], []
                start, end = chunk
                print(len(df.columns.values))

                for elem in df.columns.values[start:end]:
                    dip_3d = [(idx, df[elem][x]['pos'], df[elem][x]['dip'])
                              for idx, x in enumerate(df.index.values)]
                    idld, posss, momss = np.array(dip_3d, dtype=tuple).T
                    edges = [(x[0], y[0]) for x, y in combinations(dip_3d, 2) if np.linalg.norm(context.min_img_dist(
                        x[1], y[1], box_dim)) <= crit and context.pair_potential(context.min_img_dist(
                            x[1], y[1], box_dim), x[2], y[2]) < en_crit]
                    pf_ids = [dip_3d[x:x+20].copy()
                              for x in range(0, len(dip_3d), 20)]
                    for pf_ids_el in pf_ids:
                        edges.extend([(x[0], y[0]) for x, y in zip(
                            pf_ids_el[:-1], pf_ids_el[1:])])
                    g2 = ig.Graph(n=len(dip_3d), edges=edges)
                    g2.vs["name"] = idld
                    g2.vs["pos"] = posss
                    g2.vs["dip"] = momss
                    g2.simplify()

                    dip_energy_pt.append(np.sum([context.pair_potential(
                        context.min_img_dist(
                            ii_hndl['pos'], jj_hndl['pos'], box_dim),
                        ii_hndl['dip'],
                        jj_hndl['dip']) for ii_hndl,
                        jj_hndl in combinations(g2.vs, 2)])/len(g2.vs))
                    lj_energy_pt.append(
                        np.sum(
                            [context.lj_pair_energy(
                                lj_eps, np.linalg.norm(context.min_img_dist(ii_hndl['pos'], jj_hndl['pos'], box_dim))) for ii_hndl,
                                jj_hndl in combinations(g2.vs, 2)])/len(g2.vs))
                    zeeman_energy_pt.append(
                        np.sum([-1.0 * np.dot(np.array([0, 0, ext_fld]), ii_hndl['dip']) for ii_hndl in g2.vs])/len(g2.vs))
                per_sim_data[data_path] = (
                    lj_energy_pt, dip_energy_pt, zeeman_energy_pt)

        except StopIteration:
            break
    return per_sim_data


def bond_order_param(data_iterator, box_dim, chunk=(-50, -1), crit=1.3):
    per_sim_data = {}
    while True:
        try:
            df, data_path = next(data_iterator)
            if len(df) == 0:
                per_sim_data[data_path] = float('nan')

            else:
                steinhardt_avg_pt = []
                start, end = chunk
                relevant_columns = df.columns.values[start:end]
                for elem in relevant_columns:
                    posss = df[elem].apply(
                        lambda x: context.fold_coordinates_pp(x['pos'], box_dim=box_dim))
                    sys = pc.System()
                    box_x, box_y, box_z = box_dim
                    sys.box = [
                        [box_x, 0.0, 0.0],
                        [0.0, box_y, 0.0],
                        [0.0, 0.0, box_z]]
                    sys.atoms = np.vectorize(lambda pos_el, id_el: pc.Atom(
                        pos=pos_el, id=id_el))(posss, np.arange(len(posss)))
                    sys.find_neighbors(method='cutoff', cutoff='sann')
                    q_axis = [2, 3, 4, 5, 6, 7, 8, 9, 10]
                    sys.calculate_q(q_axis, averaged=True)
                    q_vals = sys.get_qvals(q_axis, averaged=True)
                    steinhardt_avg_pt.append(q_vals)
                print(np.shape(steinhardt_avg_pt))
                per_sim_data[data_path] = np.mean(steinhardt_avg_pt, axis=0)

        except StopIteration:
            break
    return per_sim_data


def stainhardt_param(data_iterator, box_dim, chunk=(-50, -1), crit=1.3, q_val=0):
    per_sim_data = {}
    while True:
        try:
            df, data_path = next(data_iterator)
            if len(df) == 0:
                per_sim_data[data_path] = float('nan')

            else:
                steinhardt_avg_pt = []
                start, end = chunk
                for elem in df.columns.values[start:end]:
                    dip_3d = [(idx, df[elem][x]['pos'], df[elem][x]['dip'])
                              for idx, x in enumerate(df.index.values)]
                    sys = pc.System()
                    box_x, box_y, box_z = box_dim
                    sys.box = [
                        [box_x, 0.0, 0.0],
                        [0.0, box_y, 0.0],
                        [0.0, 0.0, box_z]]
                    sys.atoms = [pc.Atom(pos=pos_el, id=id_el)
                                 for id_el, pos_el, dip_el in dip_3d]
                    sys.find_neighbors(method='cutoff', cutoff=crit)
                    sys.calculate_q(q_val, averaged=True)
                    steinhardt_avg_pt.append(
                        sys.get_qvals(q_val, averaged=True))
                per_sim_data[data_path] = np.mean(steinhardt_avg_pt, axis=0)

        except StopIteration:
            break
    return per_sim_data


def voronoi_tessellation(data_iterator, box_dim, chunk=(-50, -1), crit=1.3):
    per_sim_data = {}
    while True:
        try:
            df, data_path = next(data_iterator)
            if len(df) == 0:
                per_sim_data[data_path] = float('nan')

            else:
                steinhardt_avg_pt = []
                start, end = chunk
                print(len(df.columns.values))

                for elem in df.columns.values[start:end]:
                    dip_3d = [(idx, df[elem][x]['pos'], df[elem][x]['dip'])
                              for idx, x in enumerate(df.index.values)]
                    sys = pc.System()
                    box_x, box_y, box_z = box_dim
                    sys.box = [
                        [box_x, 0.0, 0.0],
                        [0.0, box_y, 0.0],
                        [0.0, 0.0, box_z]]
                    sys.atoms = [pc.Atom(pos=pos_el, id=id_el)
                                 for id_el, pos_el, dip_el in dip_3d]
                    sys.find_neighbors(method='voronoi')
                    sys.calculate_vorovector()
                    vorvor_vals = [atom.vorovector for atom in sys.atoms]
                    steinhardt_avg_pt.append(np.mean(vorvor_vals, axis=0))
                print('vorvor_vals mean ', np.mean(steinhardt_avg_pt, axis=0))
                per_sim_data[data_path] = np.mean(steinhardt_avg_pt, axis=0)

        except StopIteration:
            break
    return per_sim_data


def voronoi_polyhedra_volume(data_iterator, box_dim, chunk=(-50, -1), crit=1.3):
    per_sim_data = {}
    while True:
        try:
            df, data_path = next(data_iterator)
            if len(df) == 0:
                per_sim_data[data_path] = float('nan')

            else:
                steinhardt_avg_pt = []
                start, end = chunk
                print(len(df.columns.values))

                for elem in df.columns.values[start:end]:
                    dip_3d = [(idx, df[elem][x]['pos'], df[elem][x]['dip'])
                              for idx, x in enumerate(df.index.values)]
                    sys = pc.System()
                    box_x, box_y, box_z = box_dim
                    sys.box = [
                        [box_x, 0.0, 0.0],
                        [0.0, box_y, 0.0],
                        [0.0, 0.0, box_z]]
                    sys.atoms = [pc.Atom(pos=pos_el, id=id_el)
                                 for id_el, pos_el, dip_el in dip_3d]
                    sys.find_neighbors(method='voronoi')
                    vorvor_vol = [atom.volume for atom in sys.atoms]
                    steinhardt_avg_pt.extend(vorvor_vol)
                # print('vorvor_vals mean ', np.mean(steinhardt_avg_pt, axis=0))
                per_sim_data[data_path] = steinhardt_avg_pt

        except StopIteration:
            break
    return per_sim_data


def voronoi_polyhedra_area_pf(data_iterator, box_dim, chunk=(-50, -1), crit=1.3):
    per_sim_data = {}
    while True:
        try:
            df, data_path = next(data_iterator)
            if len(df) == 0:
                per_sim_data[data_path] = float('nan')

            else:
                steinhardt_avg_pt = []
                start, end = chunk
                print(len(df.columns.values))

                for elem in df.columns.values[start:end]:
                    dip_3d = [(idx, df[elem][x]['pos'], df[elem][x]['dip'])
                              for idx, x in enumerate(df.index.values)]
                    sys = pc.System()
                    box_x, box_y, box_z = box_dim
                    sys.box = [
                        [box_x, 0.0, 0.0],
                        [0.0, box_y, 0.0],
                        [0.0, 0.0, box_z]]
                    sys.atoms = [pc.Atom(pos=pos_el, id=id_el)
                                 for id_el, pos_el, dip_el in dip_3d]
                    sys.find_neighbors(method='voronoi')
                    vorvor_vol = [atom.volume for atom in sys.atoms]
                    vorvor_area_pf = []
                    for atom in sys.atoms:
                        if atom.volume < 20:
                            vorvor_area_pf.extend(
                                atom.polyhedra_face_area_list)
                    steinhardt_avg_pt.extend(vorvor_area_pf)
                # print('vorvor_vals mean ', np.mean(steinhardt_avg_pt, axis=0))
                per_sim_data[data_path] = steinhardt_avg_pt

        except StopIteration:
            break
    return per_sim_data


def voronoi_polyhedra_vol_mag(data_iterator, box_dim, chunk=(-50, -1), crit=1.3, norm=1.0):
    per_sim_data = {}
    while True:
        try:
            df, data_path = next(data_iterator)
            if len(df) == 0:
                per_sim_data[data_path] = float('nan')

            else:
                steinhardt_avg_pt = []
                start, end = chunk
                print(len(df.columns.values))

                for elem in df.columns.values[start:end]:
                    dip_3d = [(idx, df[elem][x]['pos'], df[elem][x]['dip'])
                              for idx, x in enumerate(df.index.values)]
                    sys = pc.System()
                    box_x, box_y, box_z = box_dim
                    sys.box = [
                        [box_x, 0.0, 0.0],
                        [0.0, box_y, 0.0],
                        [0.0, 0.0, box_z]]
                    sys.atoms = [pc.Atom(pos=pos_el, id=id_el)
                                 for id_el, pos_el, dip_el in dip_3d]
                    sys.find_neighbors(method='voronoi')
                    vorvor_vol = [atom.volume for atom in sys.atoms]
                    magic_ax = [vg.dot(np.array([0, 0, 1]), vg.normalize(dip_el))
                                for id_el, pos_el, dip_el in dip_3d]
                    # magic_ax = [dip_el[-1] / norm
                    #             for id_el, pos_el, dip_el in dip_3d]
                    return_organised = [(x, y) for (x, y) in zip(
                        vorvor_vol, magic_ax) if x < 20]

                    steinhardt_avg_pt.extend(return_organised)
                # print('vorvor_vals mean ', np.mean(steinhardt_avg_pt, axis=0))
                per_sim_data[data_path] = steinhardt_avg_pt

        except StopIteration:
            break
    return per_sim_data


def voronoi_polyhedra_asph_mag(data_iterator, box_dim, chunk=(-50, -1), crit=1.3):
    per_sim_data = {}
    while True:
        try:
            df, data_path = next(data_iterator)
            if len(df) == 0:
                per_sim_data[data_path] = float('nan')

            else:
                steinhardt_avg_pt = []
                start, end = chunk
                print(len(df.columns.values))

                for elem in df.columns.values[start:end]:
                    dip_3d = [(idx, df[elem][x]['pos'], df[elem][x]['dip'])
                              for idx, x in enumerate(df.index.values)]
                    sys = pc.System()
                    box_x, box_y, box_z = box_dim
                    sys.box = [
                        [box_x, 0.0, 0.0],
                        [0.0, box_y, 0.0],
                        [0.0, 0.0, box_z]]
                    sys.atoms = [pc.Atom(pos=pos_el, id=id_el)
                                 for id_el, pos_el, dip_el in dip_3d]
                    sys.find_neighbors(method='voronoi')
                    vorvor_vol = [atom.volume for atom in sys.atoms]
                    vorvor_area = [np.sum(atom.polyhedra_face_area_list)
                                   for atom in sys.atoms]
                    vorvor_asphericity = [x**3/(36*np.pi*y**2)
                                          for x, y in zip(vorvor_area, vorvor_vol)]
                    magic_ax = [vg.dot(np.array([0, 0, 1]), vg.normalize(dip_el))
                                for id_el, pos_el, dip_el in dip_3d]
                    return_organised = [(x, y) for (x, y, z) in zip(
                        vorvor_asphericity, magic_ax, vorvor_vol) if z < 20]

                    steinhardt_avg_pt.extend(return_organised)
                # print('vorvor_vals mean ', np.mean(steinhardt_avg_pt, axis=0))
                per_sim_data[data_path] = steinhardt_avg_pt

        except StopIteration:
            break
    return per_sim_data


def make_pca_data_classic(data_iterator, box_dim, chunk=(-50, -1), crit=1.3, en_crit=-0.1):
    per_sim_data = {}
    while True:
        try:
            df, data_path = next(data_iterator)
            if len(df) == 0:
                per_sim_data[data_path] = float('nan')

            else:
                accumulated_per_timestep = []
                start, end = chunk
                master_dict = {}
                for elemid, elem in enumerate(df.columns.values[start:end]):
                    composit_data = [(idx, df[elem][x]['pos'], df[elem][x]['dip'])
                                     for idx, x in enumerate(df.index.values)]
                    idld, posss, momss = np.array(composit_data, dtype=tuple).T
                    pf_composit_data = [composit_data[x: x + 20].copy()
                                        for x in range(0, len(composit_data),
                                                       20)]
                    # create connections for a network based on genral criteria
                    edges = [(x[0], y[0]) for x, y in combinations(dip_3d, 2) if np.linalg.norm(context.min_img_dist(
                        x[1], y[1], box_dim)) <= crit and context.pair_potential(context.min_img_dist(
                            x[1], y[1], box_dim), x[2], y[2]) < en_crit]
                    for pf_el in pf_composit_data:
                        edges.extend([(x[0], y[0])
                                     for x, y in zip(pf_el[:-1], pf_el[1:])])
                    g2 = ig.Graph(n=len(composit_data), edges=edges)
                    g2.vs["name"] = idld
                    g2.vs["pos"] = posss
                    g2.vs["mom"] = momss
                    g2.simplify()
                    for vid in g2.vs.indices:
                        neighbors = g2.neighbors(vid)
                        # number of neighbours
                        master_dict[elemid *
                                    len(g2.vs.indices)+vid] = [len(neighbors),]
                        # manetisation which is essentially the same as angle w.r.t mag fld
                        master_dict[elemid*len(g2.vs.indices) +
                                    vid].append(g2.vs[vid]["mom"][-1])
                        distances, combo_hild = [], []
                        combo_hild.extend(neighbors)
                        combo_hild.append(vid)
                        for ii, jj in combinations(combo_hild, 2):
                            distances.append(context.min_img_dist(
                                g2.vs[ii]["pos"],
                                g2.vs[jj]["pos"],
                                box_dim))
                        master_dict[elemid*len(g2.vs.indices) +
                                    vid].append(np.mean(distances))
                        master_dict[elemid*len(g2.vs.indices) +
                                    vid].append(np.std(distances))

                per_sim_data[data_path] = master_dict

        except StopIteration:
            break
    return per_sim_data


def make_pca_data_experimental(data_iterator, box_dim, chunk=(-50, -1), crit=1.3, en_crit=-0.1, method='vor'):
    per_sim_data = {}
    reference = np.array((0, 0, 1))
    while True:
        try:
            df, data_path = next(data_iterator)
            if len(df) == 0:
                per_sim_data[data_path] = float('nan')
            else:
                start, end = chunk
                master_dict = {}
                for elemid, elem in enumerate(df.columns.values[start:end]):
                    composit_data = [(idx, df[elem][x]['pos'], df[elem][x]['dip'])
                                     for idx, x in enumerate(df.index.values)]
                    idld, posss, momss = np.array(composit_data, dtype=tuple).T
                    sys = pc.System()
                    box_x, box_y, box_z = box_dim
                    sys.box = [
                        [box_x, 0.0, 0.0],
                        [0.0, box_y, 0.0],
                        [0.0, 0.0, box_z]]
                    sys.atoms = [pc.Atom(pos=pos_el, id=id_el)
                                 for id_el, pos_el, dip_el in composit_data]
                    if method == 'vor':
                        sys.find_neighbors(method='voronoi')
                    elif method == 'sann':
                        sys.find_neighbors(method='cutoff', cutoff='sann')
                    elif method == 'classic':
                        sys.find_neighbors(method='cutoff', cutoff=crit)
                    else:
                        print('you defined a method that isnt implemented')
                        return 0
                    edges = []
                    for atom in sys.atoms:
                        if atom.neighbors:
                            for x, y in product([atom.id,], atom.neighbors):
                                edges.append((x, y))

                    g2 = ig.Graph(n=len(sys.atoms), edges=edges)
                    g2.vs["name"] = idld
                    g2.vs["pos"] = posss
                    g2.vs["mom"] = momss
                    g2.simplify()
                    for vid in g2.vs.indices:
                        neighbors = g2.neighbors(vid)
                        # number of neighbours
                        master_dict[elemid *
                                    len(g2.vs.indices)+vid] = [len(neighbors),]
                        # magnetisation which is essentially the same as angle w.r.t mag fld
                        master_dict[elemid*len(g2.vs.indices) +
                                    vid].append(g2.vs[vid]["mom"][-1])
                        distances, combo_hild = [], []
                        combo_hild.extend(neighbors)
                        combo_hild.append(vid)
                        if len(combo_hild) > 1:
                            for ii, jj in combinations(combo_hild, 2):
                                distances.append(np.linalg.norm(context.min_img_dist(
                                    g2.vs[ii]["pos"],
                                    g2.vs[jj]["pos"],
                                    box_dim)))
                        else:
                            distances.append(0)
                        # avg inter neighbour distance and its std()
                        master_dict[elemid*len(g2.vs.indices) +
                                    vid].append(np.mean(distances))
                        master_dict[elemid*len(g2.vs.indices) +
                                    vid].append(np.std(distances))

                        # if voronoi is used for clustering save cell volume and area to the pca order parameter list
                        atom_el = sys.get_atom(vid)
                        if method == 'vor':
                            vorvor_asphericity = np.sum(
                                atom_el.polyhedra_face_area_list) ** 3 / (36 * np.pi * atom_el.volume ** 2)
                            # print(vorvor_asphericity)
                            # master_dict[elemid*len(g2.vs.indices)+vid].append(atom_el.avg_volume)
                            # master_dict[elemid*len(g2.vs.indices)+vid].append(atom_el.volume)
                            master_dict[elemid*len(g2.vs.indices) +
                                        vid].append(vorvor_asphericity)
                        neig_vecs = atom_el.neighbor_vector
                        ang_lst = []
                        if len(neig_vecs) > 1:
                            for x, y in combinations(neig_vecs, 2):
                                angle_deg = vg.angle(
                                    vg.aligned_with(
                                        np.array(x), reference, reverse=False),
                                    vg.aligned_with(
                                        np.array(y),
                                        reference, reverse=False),
                                    units='deg')
                                ang_lst.append(np.minimum(
                                    angle_deg, 180 - angle_deg))
                        else:
                            ang_lst.append(180)
                        # avg inter neighbour angle and its std()
                        master_dict[elemid*len(g2.vs.indices) +
                                    vid].append(np.mean(ang_lst))
                        master_dict[elemid*len(g2.vs.indices) +
                                    vid].append(np.std(ang_lst))
                per_sim_data[data_path] = master_dict

        except StopIteration:
            break
    return per_sim_data


def cherry_pick_vert(data_iterator, box_dim, chunk=(-50, -1), crit=1.3, en_crit=-0.1):
    per_sim_data = {}
    while True:
        try:
            df, data_path = next(data_iterator)
            if len(df) == 0:
                per_sim_data[data_path] = float('nan')

            else:
                accumulated_per_timestep = []
                start, end = chunk
                for elem in df.columns.values[start:end]:
                    dip_3d = [(idx, df[elem][x]['pos'], df[elem][x]['dip'])
                              for idx, x in enumerate(df.index.values)]
                    edges = [(x[0], y[0]) for x, y in combinations(dip_3d, 2) if np.linalg.norm(context.min_img_dist(
                        x[1], y[1], box_dim)) <= crit and context.pair_potential(context.min_img_dist(
                            x[1], y[1], box_dim), x[2], y[2]) < en_crit]
                    # pf_ids = [dip_3d[x:x+20].copy()
                    #           for x in range(0, len(dip_3d), 20)]
                    # for pf_ids_el in pf_ids:
                    #     edges.extend([(x[0], y[0]) for x, y in zip(
                    #         pf_ids_el[:-1], pf_ids_el[1:])])
                    g2 = ig.Graph(n=len(dip_3d), edges=edges)
                    g2.simplify()
                    for idx, ver in enumerate(g2.vs):
                        if ver.degree() >= 3:
                            print(idx, ver.neighbors())
                        # accumulated_per_timestep.append(
                        #     (idx, ver.degree(), ver.neighbors()))
                    # there could be another mean missings
                per_sim_data[data_path] = accumulated_per_timestep
        except StopIteration:
            break
    return per_sim_data
