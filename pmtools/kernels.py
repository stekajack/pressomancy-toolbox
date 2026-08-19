from pathlib import Path
from pmtools.resources.gryation_tensor import GyrationTensor
import numpy as np
from itertools import pairwise
import pmtools.refractored_toolbox as context
from pmtools.resources.kernel_config import AnalysisConfig
from pressomancy.analysis import H5DataSelector
from pressomancy.helper_functions import get_neighbours, get_neighbours_cross_lattice, min_img_dist
import h5py
import igraph as ig
from pmtools.refractored_toolbox_legacy import pair_potential
import vg
import warnings

def per_fil_gyr(cfg: AnalysisConfig):
    """
    Compute per-filament gyration tensors over a trajectory.

    For each timestep (batched by ``cfg.chunk``), identifies filament objects
    using the provided predicates, builds linear edges along each filament,
    iterates over connected clusters via ``context.get_cluster_iterator``,
    and accumulates :class:`~pmtools.resources.gryation_tensor.GyrationTensor`
    objects from particle positions.

    Parameters
    ----------
    cfg : AnalysisConfig
        Analysis configuration with at least:
        - ``data_path`` : path to HDF5 file.
        - ``particle_group`` : HDF5 group name.
        - ``object_predicate`` : callable taking a subset and returning a mask
          (used to select filament objects).
        - ``particle_predicate`` : callable producing a boolean mask for particles.
        - ``box_dim`` : array-like box dimensions.
        - ``chunk`` : tuple ``(start, end, step)`` for timestep slicing.

    Returns
    -------
    dict
        Mapping ``{cfg.data_path: List[GyrationTensor]}`` containing one
        gyration tensor per yielded cluster across processed timesteps.
    """

    data_with_context = {}
    data_file=h5py.File(cfg.data_path, "r")
    data=H5DataSelector(data_file, particle_group=cfg.particle_group)
    accumulated_gts = []
    start, end, step = cfg.chunk
    for col in data.timestep[start:end:step].timestep:

        fitered_fil_ids=col.get_connectivity_values(cfg.particle_group, predicate=cfg.object_predicate)
        part_sel=col.select_particles_by_object(cfg.particle_group, fitered_fil_ids,predicate=cfg.particle_predicate)
        pf_indices = [col.select_particles_by_object(cfg.particle_group, myed,predicate=cfg.particle_predicate).id.flatten() for myed in fitered_fil_ids]
        edges = context.edges_from_ordered_particle_id_groups(part_sel, pf_indices)

        graph_iterator=context.get_cluster_iterator(part_sel, edges, cfg.box_dim)
        for subgraph in graph_iterator:
            accumulated_gts.append(GyrationTensor(np.array(subgraph.vs['pos_folded_unbroken'])) )
       
    data_with_context[cfg.data_path] = accumulated_gts
    return data_with_context

def per_cluster_gyr_tensor(cfg: AnalysisConfig):
    """
    Compute per-cluster gyration tensors over a trajectory.

    For each timestep (batched by ``cfg.chunk``), builds cluster edges based on cuttoff, iterates over connected clusters via ``context.get_cluster_iterator``, and accumulates :class:`~pmtools.resources.gryation_tensor.GyrationTensor` objects from particle positions.

    Parameters
    ----------
    cfg : AnalysisConfig
        Analysis configuration with at least:
        - ``data_path`` : path to HDF5 file.
        - ``particle_group`` : HDF5 group name.
        - ``particle_predicate`` : callable producing a boolean mask for particles.
        - ``box_dim`` : array-like box dimensions.
        - ``chunk`` : tuple ``(start, end, step)`` for timestep slicing.

    Returns
    -------
    dict
        Mapping ``{cfg.data_path: List[GyrationTensor]}`` containing one
        gyration tensor per yielded cluster across processed timesteps.
    """

    data_with_context = {}
    data_file=h5py.File(cfg.data_path, "r")
    data=H5DataSelector(data_file, particle_group=cfg.particle_group)
    accumulated_gts = []
    start, end, step = cfg.chunk
    for col in data.timestep[start:end:step].timestep:
        connectivity_values = col.get_connectivity_values(cfg.particle_group)
        sel_dataview=col.select_particles_by_object(cfg.particle_group, connectivity_values, predicate=cfg.particle_predicate)
        posss = sel_dataview.pos_folded
        connectivity_list=get_neighbours(posss,cfg.box_dim[0],cfg.crit)
        edges=[]
        for part,niegh_parts in connectivity_list.items():
            for niegh in niegh_parts:
                edges.append((part,niegh))
        graph_iterator=context.get_cluster_iterator(sel_dataview, edges, cfg.box_dim, min_part=20)
        for subgraph in graph_iterator:
            accumulated_gts.append(GyrationTensor(np.array(subgraph.vs['pos_folded_unbroken'])) )
       
    data_with_context[cfg.data_path] = accumulated_gts
    return data_with_context

def cluster_size(cfg: AnalysisConfig):
    """
    Compute per-cluster gyration tensors over a trajectory.

    For each timestep (batched by ``cfg.chunk``), builds cluster edges based on cuttoff, iterates over connected clusters via ``context.get_cluster_iterator``, and accumulates :class:`~pmtools.resources.gryation_tensor.GyrationTensor` objects from particle positions.

    Parameters
    ----------
    cfg : AnalysisConfig
        Analysis configuration with at least:
        - ``data_path`` : path to HDF5 file.
        - ``particle_group`` : HDF5 group name.
        - ``particle_predicate`` : callable producing a boolean mask for particles.
        - ``box_dim`` : array-like box dimensions.
        - ``chunk`` : tuple ``(start, end, step)`` for timestep slicing.

    Returns
    -------
    dict
        Mapping ``{cfg.data_path: List[GyrationTensor]}`` containing one
        gyration tensor per yielded cluster across processed timesteps.
    """

    data_with_context = {}
    data_file=h5py.File(cfg.data_path, "r")
    data=H5DataSelector(data_file, particle_group=cfg.particle_group)
    accumulated_gts = []
    start, end, step = cfg.chunk
    for col in data.timestep[start:end:step].timestep:
        connectivity_values = col.get_connectivity_values(cfg.particle_group)
        sel_dataview=col.select_particles_by_object(cfg.particle_group, connectivity_values, predicate=cfg.particle_predicate)
        posss = sel_dataview.pos_folded
        connectivity_list=get_neighbours(posss,cfg.box_dim[0],cfg.crit)
        edges=[]
        for part,niegh_parts in connectivity_list.items():
            for niegh in niegh_parts:
                edges.append((part,niegh))
        graph_iterator=context.get_cluster_iterator(sel_dataview, edges, cfg.box_dim, min_part=20)
        tmpss=[]
        for subgraph in graph_iterator:
            tmpss.append(len(subgraph.vs))
        accumulated_gts.append(tmpss)
       
    data_with_context[cfg.data_path] = accumulated_gts
    return data_with_context

def cluster_magnetisation(cfg: AnalysisConfig):
    """
    Compute mean dipole magnetisation per particle cluster over a trajectory.

    For each timestep (batched by ``cfg.chunk``), selected particles are
    clustered by neighbour distance ``cfg.crit``. For each cluster, records
    the mean z-component of the particle dipoles, divided by ``cfg.norm``.

    Parameters
    ----------
    cfg : AnalysisConfig
        Analysis configuration with at least:
        - ``data_path`` : path to HDF5 file.
        - ``particle_group`` : HDF5 group name.
        - ``particle_predicate`` : callable producing a boolean mask for particles.
        - ``box_dim`` : array-like box dimensions.
        - ``crit`` : neighbour cutoff for cluster detection.
        - ``norm`` : magnetisation normalization factor.
        - ``chunk`` : tuple ``(start, end, step)`` for timestep slicing.

    Returns
    -------
    dict
        Mapping ``{cfg.data_path: List[List[float]]}`` containing one list of
        cluster magnetisations per processed timestep.
    """

    data_with_context = {}
    data_file=h5py.File(cfg.data_path, "r")
    data=H5DataSelector(data_file, particle_group=cfg.particle_group)
    accumulated_gts = []
    start, end, step = cfg.chunk
    for col in data.timestep[start:end:step].timestep:
        connectivity_values = col.get_connectivity_values(cfg.particle_group)
        sel_dataview=col.select_particles_by_object(cfg.particle_group, connectivity_values, predicate=cfg.particle_predicate)
        posss = sel_dataview.pos_folded
        connectivity_list=get_neighbours(posss,cfg.box_dim[0],cfg.crit)
        edges=[]
        for part,niegh_parts in connectivity_list.items():
            for niegh in niegh_parts:
                edges.append((part,niegh))
        graph_iterator=context.get_cluster_iterator(
            sel_dataview,
            edges,
            cfg.box_dim,
            min_part=20,
            attibutes=['pos_folded', 'dip'],
        )
        accumulated_magnetisation=[]
        for subgraph in graph_iterator:
            dipoles=np.mean(subgraph.vs['dip'],axis=0)[-1]/float(cfg.norm)
            accumulated_magnetisation.append(dipoles)      
        accumulated_gts.append(accumulated_magnetisation)
       
    data_with_context[cfg.data_path] = accumulated_gts
    return data_with_context

def pair_distances(cfg: AnalysisConfig):
    """
    Measure neighbour-pair distances within selected particle clusters.

    Particles are selected with ``cfg.particle_predicate`` and linked by
    ``cfg.crit`` using periodic boundaries. The returned list contains one
    minimum-image distance for each graph edge inside each detected cluster.
    """

    data_with_context = {}
    data_file=h5py.File(cfg.data_path, "r")
    data=H5DataSelector(data_file, particle_group=cfg.particle_group)
    accumulated_gts = []
    start, end, step = cfg.chunk
    for col in data.timestep[start:end:step].timestep:
        connectivity_values = col.get_connectivity_values(cfg.particle_group)
        sel_dataview=col.select_particles_by_object(cfg.particle_group, connectivity_values, predicate=cfg.particle_predicate)
        posss = sel_dataview.pos_folded
        connectivity_list=get_neighbours(posss,cfg.box_dim[0],cfg.crit)
        edges=[]
        for part,niegh_parts in connectivity_list.items():
            for niegh in niegh_parts:
                edges.append((part,niegh))
        graph_iterator=context.get_cluster_iterator(sel_dataview, edges, cfg.box_dim, min_part=20)
        for subgraph in graph_iterator:
            for vertex in subgraph.vs:
                for neighbor in subgraph.neighbors(vertex):
                    accumulated_gts.append(np.linalg.norm(min_img_dist(vertex['pos_folded'],subgraph.vs[neighbor]['pos_folded'], cfg.box_dim)))
       
    data_with_context[cfg.data_path] = accumulated_gts
    return data_with_context

def mlp_projection(cfg: AnalysisConfig):
    """
    Compute the segment-wise projection onto the end-to-end vector (``ℓ_p`` proxy).

    For each timestep, selects filaments, builds linear edges, iterates over
    connected clusters, and computes the projection of consecutive monomer
    center-of-mass segments onto the filament end-to-end vector.

    Parameters
    ----------
    cfg : AnalysisConfig
        Analysis configuration with fields used:
        - ``data_path``, ``particle_group``, ``object_predicate``,
          ``particle_predicate``, ``template_hndl``, ``box_dim``, ``chunk``.

    Returns
    -------
    dict
        ``{cfg.data_path: (mean_segment_projections, x_axis)}`` where
        ``mean_segment_projections`` is the average projection over all
        processed clusters/timesteps and ``x_axis`` is ``np.arange(monomer_no-1)+1``.
    """

    data_with_context = {}
    data_file=h5py.File(cfg.data_path, "r")
    data=H5DataSelector(data_file,particle_group=cfg.particle_group)
    monomer_no = int(context.determine_key_val_from_filename(cfg.template_hndl,cfg.data_path,'what_monomer_number'))
    accumulated_lp_seg = []
    start, end, step = cfg.chunk
    for col in data.timestep[start:end:step].timestep:
        fitered_fil_ids=col.get_connectivity_values(cfg.particle_group, predicate=cfg.object_predicate)
        part_sel=col.select_particles_by_object(cfg.particle_group, fitered_fil_ids,predicate=cfg.particle_predicate)
        
        pf_indices = [col.select_particles_by_object(cfg.particle_group, myed,predicate=cfg.particle_predicate).id.flatten() for myed in fitered_fil_ids]
        edges = context.edges_from_ordered_particle_id_groups(part_sel, pf_indices)
        graph_iterator=context.get_cluster_iterator(part_sel, edges, cfg.box_dim, attibutes=['pos_folded','dip'])
        for subgraph in graph_iterator:
            positions=np.array(subgraph.vs['pos_folded_unbroken'])
            dipoles=np.array(subgraph.vs['dip'])
            com_pos = np.mean(positions.reshape(monomer_no, -1, 3), axis=1)
            ete_vec = com_pos[-1]-com_pos[0]
            dipole_norms = np.mean(np.linalg.norm(dipoles, axis=1))
            res_dip = np.dot(dipoles, ete_vec)/pow(dipole_norms,2)
            accumulated_lp_seg.append(res_dip)            
    xax = np.arange(monomer_no)+1
    data_with_context[cfg.data_path] = np.mean(
        accumulated_lp_seg, axis=0), xax
    return data_with_context

def easy_dipmom_angle(cfg: AnalysisConfig):
    """
    Calculate the angle between magnetic dipoles and directors reconstructed from type-2 anchors.

    Warning
    -------
    This is a dataset-specific recovery kernel for simulations where particle
    directors were not saved. It assumes that each filament has exactly 40
    type-2 anchor particles ordered as all bottom anchors first, followed by
    all top anchors, so anchors ``[:20]`` pair with anchors ``[20:]``. Do not
    use this kernel for datasets with interleaved anchors or stored director
    vectors without revisiting the pairing logic.
    """
    warnings.warn(
        "easy_dipmom_angle assumes type-2 anchors are ordered as 20 bottom "
        "anchors followed by 20 top anchors per filament; this is a "
        "dataset-specific director reconstruction hack.",
        RuntimeWarning,
        stacklevel=2,
    )

    data_with_context = {}
    data_file=h5py.File(cfg.data_path, "r")
    data=H5DataSelector(data_file,particle_group=cfg.particle_group)
    accumulated_lp_seg = []
    start, end, step = cfg.chunk
    for col in data.timestep[start:end:step].timestep:
        fitered_fil_ids=col.get_connectivity_values(cfg.particle_group, predicate=cfg.object_predicate)
        for filament_id in fitered_fil_ids:

            part_sel_mag=col.select_particles_by_object(cfg.particle_group, filament_id,predicate=cfg.particle_predicate)

            def select_easy(subset):
                return subset.type == 2
            
            part_sel_easy=col.select_particles_by_object(cfg.particle_group, filament_id, predicate=select_easy)
            positions=np.array(part_sel_easy.pos_folded)
            assert np.shape(positions)==(40,3), "Expected 40 particles per easy filament"
            p1 = positions[:20]       # indices 0–19
            p2 = positions[20:]       # indices 20–39
            segments = min_img_dist(p2,p1, cfg.box_dim)
            seg_norms = np.linalg.norm(segments, axis=1)
            assert all(seg_norms<2.), f"Segment norm too large, something is wrong! seg_norms={seg_norms}"
            segments = segments / seg_norms[:, None]
            dipoles=np.array(part_sel_mag.dip)
            dipole_norms = np.linalg.norm(dipoles, axis=1)
            dipoles=dipoles/dipole_norms[:, None]
            res_cors = np.einsum('ij,ij->i', dipoles, segments) 
            accumulated_lp_seg.extend(res_cors)            
    data_with_context[cfg.data_path] = accumulated_lp_seg
    return data_with_context

def lp_projection(cfg: AnalysisConfig):
    """
    Compute the segment-wise projection onto the end-to-end vector (``ℓ_p`` proxy).

    For each timestep, selects filaments, builds linear edges, iterates over
    connected clusters, and computes the projection of consecutive monomer
    center-of-mass segments onto the filament end-to-end vector.

    Parameters
    ----------
    cfg : AnalysisConfig
        Analysis configuration with fields used:
        - ``data_path``, ``particle_group``, ``object_predicate``,
          ``particle_predicate``, ``template_hndl``, ``box_dim``, ``chunk``.

    Returns
    -------
    dict
        ``{cfg.data_path: (mean_segment_projections, x_axis)}`` where
        ``mean_segment_projections`` is the average projection over all
        processed clusters/timesteps and ``x_axis`` is ``np.arange(monomer_no-1)+1``.
    """

    data_with_context = {}
    data_file=h5py.File(cfg.data_path, "r")
    data=H5DataSelector(data_file,particle_group=cfg.particle_group)
    monomer_no = int(context.determine_key_val_from_filename(cfg.template_hndl,cfg.data_path,'what_monomer_number'))
    accumulated_lp_seg = []
    start, end, step = cfg.chunk
    for col in data.timestep[start:end:step].timestep:
        fitered_fil_ids=col.get_connectivity_values(cfg.particle_group, predicate=cfg.object_predicate)
        part_sel=col.select_particles_by_object(cfg.particle_group, fitered_fil_ids,predicate=cfg.particle_predicate)
        
        pf_indices = [col.select_particles_by_object(cfg.particle_group, myed,predicate=cfg.particle_predicate).id.flatten() for myed in fitered_fil_ids]
        edges = context.edges_from_ordered_particle_id_groups(part_sel, pf_indices)
        graph_iterator=context.get_cluster_iterator(part_sel, edges, cfg.box_dim, attibutes=['pos_folded','dip'])
        for subgraph in graph_iterator:
            positions=np.array(subgraph.vs['pos_folded_unbroken'])
            com_pos = np.mean(positions.reshape(monomer_no, -1, 3), axis=1)
            ete_vec = com_pos[-1]-com_pos[0]
            segments = np.diff(com_pos, axis=0)
            seg_norms = np.mean(np.linalg.norm(segments, axis=1))
            res = np.dot(segments, ete_vec)/pow(seg_norms,2)
            accumulated_lp_seg.append(res)            
    xax = np.arange(monomer_no-1)+1
    data_with_context[cfg.data_path] = np.mean(
        accumulated_lp_seg, axis=0), xax
    return data_with_context  

def Ree_segments(cfg: AnalysisConfig):
    """
    Compute filament end-to-end distances and segment lengths.

    For each timestep, selects filaments, builds linear edges, iterates over
    connected clusters, and records the end-to-end distance and consecutive
    monomer center-of-mass segment lengths for each filament.

    Parameters
    ----------
    cfg : AnalysisConfig
        Analysis configuration with fields used:
        - ``data_path``, ``particle_group``, ``object_predicate``,
          ``particle_predicate``, ``template_hndl``, ``box_dim``, ``chunk``.

    Returns
    -------
    dict
        ``{cfg.data_path: (end_to_end_distances, segment_lengths)}``, with one
        entry per processed filament and timestep. Each segment-length entry
        contains ``monomer_no - 1`` consecutive segment lengths.
    """

    data_with_context = {}
    data_file=h5py.File(cfg.data_path, "r")
    data=H5DataSelector(data_file,particle_group=cfg.particle_group)
    monomer_no = int(context.determine_key_val_from_filename(cfg.template_hndl,cfg.data_path,'what_monomer_number'))
    accumulated_ree = []
    accumulated_segments = []
    start, end, step = cfg.chunk
    for col in data.timestep[start:end:step].timestep:
        fitered_fil_ids=col.get_connectivity_values(cfg.particle_group, predicate=cfg.object_predicate)
        part_sel=col.select_particles_by_object(cfg.particle_group, fitered_fil_ids,predicate=cfg.particle_predicate)

        pf_indices = [col.select_particles_by_object(cfg.particle_group, myed,predicate=cfg.particle_predicate).id.flatten() for myed in fitered_fil_ids]

        edges = context.edges_from_ordered_particle_id_groups(part_sel, pf_indices)
        graph_iterator=context.get_cluster_iterator(part_sel, edges, cfg.box_dim, attibutes=['pos_folded','dip'])
        for subgraph in graph_iterator:
            positions=np.array(subgraph.vs['pos_folded_unbroken'])
            com_pos = np.mean(positions.reshape(monomer_no, -1, 3), axis=1)
            ete_vec = com_pos[-1]-com_pos[0]
            segments = np.diff(com_pos, axis=0)
            seg_norms = np.linalg.norm(segments, axis=1)
            accumulated_ree.append(np.linalg.norm(ete_vec))
            accumulated_segments.append(seg_norms)
    data_with_context[cfg.data_path] = accumulated_ree, accumulated_segments
    return data_with_context

def magnetisation(cfg: AnalysisConfig):

    """
    Compute mean dipole magnetisation per cluster per timestep.

    For each timestep, constructs clusters for the selected filaments and
    records the mean dipole vector (last component divided by ``cfg.norm``)
    for each cluster.

    Parameters
    ----------
    cfg : AnalysisConfig
        Configuration with at least:
        - ``data_path``, ``particle_group``, ``object_predicate``,
          ``particle_predicate``, ``box_dim``, ``norm``, ``chunk``.

    Returns
    -------
    dict
        ``{cfg.data_path: List[float]}`` list of magnetisation values
        accumulated over clusters and timesteps.
    """
    data_with_context = {}
    data_file=h5py.File(cfg.data_path, "r")
    data=H5DataSelector(data_file,particle_group=cfg.particle_group)
    accumulated_magnetisation = []
    start, end, step = cfg.chunk
    for col in data.timestep[start:end:step].timestep:
        fitered_fil_ids=col.get_connectivity_values(cfg.particle_group, predicate=cfg.object_predicate)
        part_sel=col.select_particles_by_object(cfg.particle_group, fitered_fil_ids,predicate=cfg.particle_predicate)
        pf_indices = [col.select_particles_by_object(cfg.particle_group, myed,predicate=cfg.particle_predicate).id.flatten() for myed in fitered_fil_ids]
        edges = context.edges_from_ordered_particle_id_groups(part_sel, pf_indices)
        graph_iterator=context.get_cluster_iterator(part_sel, edges, cfg.box_dim, attibutes=['pos_folded','dip'])
        for subgraph in graph_iterator:
            dipoles=np.mean(subgraph.vs['dip'],axis=0)[-1]/float(cfg.norm)
            accumulated_magnetisation.append(dipoles)            
    data_with_context[cfg.data_path] = accumulated_magnetisation
    return data_with_context 

def magn_princip_angle_dist(cfg: AnalysisConfig):
    """
    Compute angles between cluster magnetisation and principal gyration axis.

    Filaments selected by ``cfg.object_predicate`` and ``cfg.particle_predicate``
    are grouped into clusters. For each cluster, the mean dipole vector is
    compared with the largest principal-axis eigenvector.
    """

    data_with_context = {}
    data_file=h5py.File(cfg.data_path, "r")
    data=H5DataSelector(data_file,particle_group=cfg.particle_group)
    monomer_no = int(context.determine_key_val_from_filename(cfg.template_hndl,cfg.data_path,'what_monomer_number'))
    accumulated_magnetisation = []
    start, end, step = cfg.chunk
    for col in data.timestep[start:end:step].timestep:
        fitered_fil_ids=col.get_connectivity_values(cfg.particle_group, predicate=cfg.object_predicate)
        part_sel=col.select_particles_by_object(cfg.particle_group, fitered_fil_ids,predicate=cfg.particle_predicate)
        pf_indices = [col.select_particles_by_object(cfg.particle_group, myed,predicate=cfg.particle_predicate).id.flatten() for myed in fitered_fil_ids]
        edges = context.edges_from_ordered_particle_id_groups(part_sel, pf_indices)
        graph_iterator=context.get_cluster_iterator(part_sel, edges, cfg.box_dim, attibutes=['pos_folded','dip'])
        reference=np.array([0,0,1])
        for subgraph in graph_iterator:
            chain_dip_mom=np.mean(subgraph.vs['dip'],axis=0)/float(cfg.norm)
            gt=GyrationTensor(np.array(subgraph.vs['pos_folded_unbroken'])) 
            res_angle=vg.angle(chain_dip_mom,vg.aligned_with(gt.eigenvectors[:, -1], reference, reverse=False), units='deg')
            accumulated_magnetisation.append(np.minimum(res_angle, 180 - res_angle))            
    data_with_context[cfg.data_path] = accumulated_magnetisation
    return data_with_context 

def magnetisation_culster_size(cfg: AnalysisConfig):
    """
    Return cluster size together with mean dipole magnetisation.

    Selected particles are clustered by neighbour distance ``cfg.crit``. Each
    result entry is ``(cluster_size, mean_dipole_z / cfg.norm)``.
    """

    data_with_context = {}
    data_file=h5py.File(cfg.data_path, "r")
    data=H5DataSelector(data_file,particle_group=cfg.particle_group)
    accumulated_magnetisation = []
    start, end, step = cfg.chunk
    for col in data.timestep[start:end:step].timestep:
        connectivity_values = col.get_connectivity_values(cfg.particle_group)
        sel_dataview=col.select_particles_by_object(cfg.particle_group, connectivity_values, predicate=cfg.particle_predicate)
        posss = sel_dataview.pos_folded
        connectivity_list=get_neighbours(posss,cfg.box_dim[0],cfg.crit)
        edges=[]
        for part,niegh_parts in connectivity_list.items():
            for niegh in niegh_parts:
                edges.append((part,niegh))
        graph_iterator=context.get_cluster_iterator(sel_dataview, edges, cfg.box_dim, min_part=20,attibutes=['pos_folded','dip'])
        for subgraph in graph_iterator:
            dipoles=np.mean(subgraph.vs['dip'],axis=0)[-1]/float(cfg.norm)
            accumulated_magnetisation.append((len(subgraph.vs),dipoles))            
    data_with_context[cfg.data_path] = accumulated_magnetisation
    return data_with_context 

def degree_magnetisation(cfg: AnalysisConfig):
    """
    Compute filtered contact degree and dipole orientation per selected particle.

    Candidate neighbours are found within ``cfg.crit``. Edges are kept when the
    pair potential is attractive enough, then each selected particle contributes
    ``(degree, dipole_z)`` in the filtered graph.
    """

    data_with_context = {}
    data_file=h5py.File(cfg.data_path, "r")
    data=H5DataSelector(data_file,particle_group=cfg.particle_group)
    accumulated_magnetisation = []
    start, end, step = cfg.chunk
    for col in data.timestep[start:end:step].timestep:
        connectivity_values = col.get_connectivity_values(cfg.particle_group)
        sel_dataview=col.select_particles_by_object(cfg.particle_group, connectivity_values, predicate=cfg.particle_predicate)
        posss = sel_dataview.pos_folded
        dipoles = sel_dataview.dip
        connectivity_list=get_neighbours(posss,cfg.box_dim[0],cfg.crit)
        edges=[]
        for part,niegh_parts in connectivity_list.items():
            for niegh in niegh_parts:
                edges.append((part,niegh))

        neighbour_graph = ig.Graph(n=len(posss), edges=edges)
        neighbour_graph.simplify()
        filtered_edges=[]
        for x,y in neighbour_graph.get_edgelist():
            res=pair_potential(
                min_img_dist(posss[x], posss[y], cfg.box_dim),
                dipoles[x],
                dipoles[y],
            )
            if res <= -0.1:
                filtered_edges.append((x,y))

        graph = ig.Graph(n=len(posss), edges=filtered_edges)
        graph.simplify()
        graph.vs['dip'] = list(dipoles)
        for vertex in graph.vs:
            accumulated_magnetisation.append((graph.degree(vertex.index), vertex['dip'][-1]))
    data_with_context[cfg.data_path] = accumulated_magnetisation
    return data_with_context     

def calculate_stacking_fraction(cfg: AnalysisConfig):
    """
    Identify ligand–stacking-site contacts across timesteps.

    Selects stacking sites (type==4) and ligands (type==5) from two particle
    groups, computes cross-lattice nearest neighbours within cutoff
    ``cfg.crit`` using periodic boundary conditions, and returns the grouped
    indices per timestep.

    Parameters
    ----------
    cfg : AnalysisConfig
        Requires:
        - ``data_path``, ``particle_group``, ``particle_group_alt``,
          ``box_dim``, ``crit``, ``chunk``.

    Returns
    -------
    dict
        ``{cfg.data_path: List[dict]}`` where each dict maps a ligand index
        to indices of nearby stacking sites for each processed timestep.
    """
    
    data_with_context = {}
    data_file=h5py.File(cfg.data_path, "r")
    data=H5DataSelector(data_file, particle_group=cfg.particle_group)
    data_other=H5DataSelector(data_file,particle_group=cfg.particle_group_alt)
    
    start, end, step = cfg.chunk
    data_per_timestep=[]
    for col_fil,col_crow in zip(data.timestep[start:end:step].timestep, data_other.timestep[start:end:step].timestep):
        mask_stack=col_fil.particles[:].type.flatten()==4
        mask_ligand=col_crow.particles[:].type.flatten()==5
        
        mask_stack=np.arange(len(mask_stack))[mask_stack]
        mask_ligand=np.arange(len(mask_ligand))[mask_ligand]
        
        stacking_sites=col_fil.particles[list(mask_stack)]
        ligands=col_crow.particles[list(mask_ligand)]
    
        grouped_indices=get_neighbours_cross_lattice(ligands.pos,stacking_sites.pos, cfg.box_dim[0],cfg.crit)
        data_per_timestep.append(grouped_indices)
        
    data_with_context[cfg.data_path] = data_per_timestep
    return data_with_context

def calculate_sf(cfg: AnalysisConfig):
    """
    Calculate the static structure factor ``S(q)`` from selected particles.

    Uses the external ``sq_avx`` library for fast evaluation. For each
    timestep (batched by ``cfg.chunk``), applies ``cfg.particle_predicate`` to
    select particles, then calls ``sq_avx.calculate_structure_factor`` with
    parameters from ``cfg.sq_params``.

    Parameters
    ----------
    cfg : AnalysisConfig
        Required fields:
        - ``data_path``, ``particle_group``, ``box_dim``, ``chunk``,
          ``sq_params`` dict with keys:
          ``'order'``, ``'orientations_per_wavevector'``,
          ``'subsample_wavevectors'``; optional ``'axis_mask'`` and ``'nthreads'``.
        - ``particle_predicate`` callable.

    Returns
    -------
    dict
        ``{cfg.data_path: (wavevectors_container, intensities_container)}``,
        where both entries are lists over processed timesteps.

    Notes
    -----
    See the implementation of ``sq_avx`` at the referenced repository:
    https://github.com/stekajack/espressoSq
    """

    import sq_avx

    data_with_context = {}
    data_file=h5py.File(cfg.data_path, "r")
    data=H5DataSelector(data_file,particle_group=cfg.particle_group)

    sq_params = cfg.sq_params
    subsample_wavevectors = sq_params['subsample_wavevectors']

    wavevectors_container, intensities_container = [], []
    start, end, step = cfg.chunk
    for col in data.timestep[start:end:step].timestep:
        mask=cfg.particle_predicate(col).flatten() # type: ignore
        posss = col.pos_folded[mask]
        wavevectors, intensities = sq_avx.calculate_structure_factor(
            posss,
            sq_params['order'],
            cfg.box_dim[0],
            sq_params['orientations_per_wavevector'],
            subsample_wavevectors,
            sq_params.get('axis_mask', [True, True, True]),
            sq_params.get('nthreads', 1),
        )
        wavevectors_container.append(wavevectors)
        intensities_container.append(intensities)

    data_with_context[cfg.data_path] = wavevectors_container, intensities_container
    return data_with_context

def calculate_rdf(cfg: AnalysisConfig):
    """
    Calculate radial distribution functions for selected particles.

    For each timestep in ``cfg.chunk``, particles passing
    ``cfg.particle_predicate`` are passed to pyscal and the RDF histogram is
    returned as wavevector/bin positions and intensities.
    """

    import pyscal as pc

    data_with_context = {}
    data_file=h5py.File(cfg.data_path, "r")
    data=H5DataSelector(data_file,particle_group=cfg.particle_group)

    wavevectors_container, intensities_container = [], []
    start, end, step = cfg.chunk
    for col in data.timestep[start:end:step].timestep:
        mask=cfg.particle_predicate(col).flatten() # type: ignore
        posss = col.pos_folded[mask]
        sys = pc.System()
        sys.box = [
            [cfg.box_dim[0], 0.0, 0.0],
            [0.0, cfg.box_dim[1], 0.0],
            [0.0, 0.0, cfg.box_dim[2]]]
        sys.atoms = [pc.Atom(pos=pos_el, id=id_el)
                        for id_el, pos_el in enumerate(posss)]
        wavevectors, intensities = sys.calculate_rdf(
            histobins=int((cfg.box_dim[0]*0.5)/0.25), histomax=cfg.box_dim[0]*0.5)
        wavevectors_container.append(wavevectors)
        intensities_container.append(intensities)
        
    data_with_context[cfg.data_path] = wavevectors_container, intensities_container
    return data_with_context

def tag_voronoi_particles(cfg: AnalysisConfig):
    """Tag selected particles by Voronoi ``(V, E, F)`` polyhedra."""
    import pyscal as pc

    if cfg.volume_cutoff is None or cfg.probability_floor is None:
        raise ValueError(
            'tag_voronoi_particles requires explicit volume_cutoff and probability_floor values.'
        )
    if cfg.volume_cutoff <= 0:
        raise ValueError('volume_cutoff must be positive.')
    if not 0 < cfg.probability_floor <= 1:
        raise ValueError('probability_floor must be in the interval (0, 1].')

    data_with_context = {}
    data_file=h5py.File(cfg.data_path, "r")
    data=H5DataSelector(data_file,particle_group=cfg.particle_group)
    res_container=[]
    start, end, step = cfg.chunk

    for frame_id, col in enumerate(data.timestep[start:end:step].timestep):
        connectivity_values = col.get_connectivity_values(cfg.particle_group)
        sel_dataview = col.select_particles_by_object(
            cfg.particle_group, connectivity_values, predicate=cfg.particle_predicate
        )
        posss = sel_dataview.pos_folded
        src_ids = sel_dataview.id.flatten()

        sys = pc.System()
        sys.box = [
            [cfg.box_dim[0], 0.0, 0.0],
            [0.0, cfg.box_dim[1], 0.0],
            [0.0, 0.0, cfg.box_dim[2]]]
        sys.atoms = [pc.Atom(pos=pos_el, id=id_el)
                     for id_el, pos_el in zip(src_ids, posss)]
        sys.find_neighbors(method='voronoi')

        atom_ids=np.array([atom.id for atom in sys.atoms])
        assert np.array_equal(src_ids, atom_ids), 'identities of the particles from the source must match identities from pyscal!'
        vor_vol=np.array([atom.volume for atom in sys.atoms])
        valid_mask=vor_vol < cfg.volume_cutoff
        face_perimeters=[atom.face_perimeters for atom in sys.atoms]
        no_of_edges=[atom.no_of_edges for atom in sys.atoms]
        vertex_vectors=[atom.vertex_vectors for atom in sys.atoms]
        F=np.array([len(x) for iid, x in enumerate(face_perimeters) if valid_mask[iid]])
        E=np.array([x for iid, x in enumerate(no_of_edges) if valid_mask[iid]])
        V=np.array([len(x)/3 for iid, x in enumerate(vertex_vectors) if valid_mask[iid]])
        assert all(V-E+F == 2), 'cant have non-covex voronoi cells!'

        polyhedra=np.stack([V, E, F], axis=1)
        unique_polyhedra, counts = np.unique(polyhedra, axis=0, return_counts=True)
        probs=counts/counts.sum()
        idx=np.argsort(probs)[::-1]
        unique_polyhedra=unique_polyhedra[idx]
        probs=probs[idx]
        polyhedron_probabilities = {
            tuple(int(val) for val in poly): float(prob)
            for poly, prob in zip(unique_polyhedra, probs)}
        cumulative, top_k=0.0, 0
        while top_k < len(unique_polyhedra) and cumulative <= cfg.probability_floor:
            cumulative += probs[top_k]
            top_k += 1

        enumerated_polyhedra=[]
        enumeration_lookup={}
        for offset, poly in enumerate(unique_polyhedra[:top_k]):
            enum_value=347+offset
            poly_key=tuple(int(val) for val in poly)
            enumerated_polyhedra.append((enum_value, poly_key))
            enumeration_lookup[poly_key]=enum_value

        particle_polyhedra_types=[]
        poly_idx=0
        for atom_id, is_valid in zip(atom_ids, valid_mask):
            if is_valid:
                poly_key=tuple(int(val) for val in polyhedra[poly_idx])
                poly_idx += 1
                enum_value=enumeration_lookup.get(poly_key, 1)
            else:
                poly_key=(-1, -1, -1)
                enum_value=765
            particle_polyhedra_types.append((int(atom_id), poly_key, int(enum_value)))

        if cfg.path_to_output:
            local_file=f'{cfg.path_to_output}/voronoi_tags_frame_{frame_id}.vtf'
            with open(local_file, 'w') as vtf:
                vtf.write(f"unitcell {cfg.box_dim[0]} {cfg.box_dim[1]} {cfg.box_dim[2]}\n")
                vtf.write(f"atom 0:{len(posss)-1} radius 0.5 name A\n")
                for particle_index, (_, _, type_id) in enumerate(particle_polyhedra_types):
                    vtf.write(f"atom {particle_index} type {type_id}\n")
                vtf.write("timestep\n")
                for x, y, z in posss:
                    vtf.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

        res_container.append({
            "enumerated_polyhedra": enumerated_polyhedra,
            "particle_polyhedra_types": particle_polyhedra_types,
            "polyhedron_probabilities": polyhedron_probabilities,
        })

    data_with_context[cfg.data_path] = res_container
    return data_with_context

def cylindrical_radial_density(cfg: AnalysisConfig):
    """Collect cylindrical radial-distance samples for Voronoi-tagged particles.

    Voronoi tagging is delegated to :func:`tag_voronoi_particles`. The tagged
    particle IDs are then joined to clusters formed with ``cfg.crit``. For each
    cluster, particles are binned along ``z`` with ``cfg.z_bin_width`` and the
    radial distance is measured from the cluster-wide xy centre.

    Returns
    -------
    dict
        ``{cfg.data_path: {'radial_samples': ..., 'bin_width': ...}}``. Each
        radial sample is ``((V, E, F), type_id, radial_distance, z_centre)``.
    """
    data_with_context={}
    tagged_frames=tag_voronoi_particles(cfg)[cfg.data_path]
    radial_samples=[]
    data_file=h5py.File(cfg.data_path, "r")
    data=H5DataSelector(data_file,particle_group=cfg.particle_group)
    start, end, step = cfg.chunk

    for col, tagged_frame in zip(data.timestep[start:end:step].timestep, tagged_frames, strict=True):
        connectivity_values = col.get_connectivity_values(cfg.particle_group)
        sel_dataview = col.select_particles_by_object(
            cfg.particle_group, connectivity_values, predicate=cfg.particle_predicate
        )
        type_lookup={
            particle_id: type_id
            for particle_id, _, type_id in tagged_frame['particle_polyhedra_types']
        }
        poly_lookup={
            particle_id: polyhedron
            for particle_id, polyhedron, _ in tagged_frame['particle_polyhedra_types']
        }

        posss=sel_dataview.pos_folded
        connectivity_list=get_neighbours(posss,cfg.box_dim[0],cfg.crit)
        edges=[]
        for part, neigh_parts in connectivity_list.items():
            for neigh in neigh_parts:
                edges.append((part, neigh))
        graph_iterator=context.get_cluster_iterator(
            sel_dataview, edges, cfg.box_dim, min_part=cfg.min_cluster_size,
            attibutes=['id','pos_folded','dip'])

        for subgraph in graph_iterator:
            loc_pos=np.asarray(subgraph.vs['pos_folded_unbroken'],dtype=float)
            loc_ids=np.asarray(subgraph.vs['id'],dtype=int).ravel()
            z_coords=loc_pos[:, 2]
            bin_edges=np.arange(z_coords.min(),z_coords.max()+cfg.z_bin_width,cfg.z_bin_width)
            if bin_edges.size == 1 or bin_edges[-1] < z_coords.max():
                bin_edges = np.append(bin_edges, bin_edges[-1] + cfg.z_bin_width)

            xy_coords=loc_pos[:, :2]
            cluster_center_xy=xy_coords.mean(axis=0)
            last_interval=bin_edges.size-2
            for idx in range(bin_edges.size-1):
                lower, upper=bin_edges[idx], bin_edges[idx+1]
                if idx == last_interval:
                    mask_slice=(z_coords >= lower) & (z_coords <= upper)
                else:
                    mask_slice=(z_coords >= lower) & (z_coords < upper)
                slice_indices=np.where(mask_slice)[0]
                if slice_indices.size < cfg.min_slice_population:
                    if slice_indices.size:
                        warnings.warn(
                            f"Slice with {slice_indices.size} particle(s) encountered in cylindrical_radial_density for {cfg.data_path}; skipping.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    continue

                slice_positions=loc_pos[slice_indices]
                z_center=float(slice_positions[:, 2].mean())
                radial_distances=np.linalg.norm(xy_coords[slice_indices]-cluster_center_xy,axis=1)
                for particle_id, radial_distance in zip(loc_ids[slice_indices], radial_distances):
                    particle_id=int(particle_id)
                    radial_samples.append((
                        tuple(poly_lookup[particle_id]),
                        int(type_lookup[particle_id]),
                        float(radial_distance),
                        z_center,
                    ))

    data_with_context[cfg.data_path]={
        'radial_samples': radial_samples,
        'bin_width': cfg.z_bin_width,
    }
    return data_with_context

def calculate_volume_voronoi(cfg: AnalysisConfig):
    """
    Calculate Voronoi cell volumes for selected particles over time.

    Each timestep contributes a list of pyscal Voronoi volumes for particles
    passing ``cfg.particle_predicate``.
    """

    import pyscal as pc

    data_with_context = {}
    data_file=h5py.File(cfg.data_path, "r")
    data=H5DataSelector(data_file,particle_group=cfg.particle_group)

    res_containter=[]
    start, end, step = cfg.chunk
    for col in data.timestep[start:end:step].timestep:
        mask=cfg.particle_predicate(col).flatten() # type: ignore
        posss = col.pos_folded[mask]
        sys = pc.System()
        sys.box = [
            [cfg.box_dim[0], 0.0, 0.0],
            [0.0, cfg.box_dim[1], 0.0],
            [0.0, 0.0, cfg.box_dim[2]]]
        sys.atoms = [pc.Atom(pos=pos_el, id=id_el)
                        for id_el, pos_el in enumerate(posss)]
        sys.find_neighbors(method='voronoi')
        vor_vol = [atom.volume for atom in sys.atoms]

        res_containter.append(vor_vol)
    data_with_context[cfg.data_path] = res_containter
    return data_with_context     

def calculate_voronoi_vects(cfg: AnalysisConfig):
    """
    Calculate pyscal Voronoi vectors for selected particles.

    Results are flattened across all timesteps selected by ``cfg.chunk``.
    """

    import pyscal as pc

    data_with_context = {}
    data_file=h5py.File(cfg.data_path, "r")
    data=H5DataSelector(data_file,particle_group=cfg.particle_group)

    res_containter=[]
    start, end, step = cfg.chunk
    for col in data.timestep[start:end:step].timestep:
        mask=cfg.particle_predicate(col).flatten() # type: ignore
        posss = col.pos_folded[mask]
        sys = pc.System()
        sys.box = [
            [cfg.box_dim[0], 0.0, 0.0],
            [0.0, cfg.box_dim[1], 0.0],
            [0.0, 0.0, cfg.box_dim[2]]]
        sys.atoms = [pc.Atom(pos=pos_el, id=id_el)
                        for id_el, pos_el in enumerate(posss)]
        sys.find_neighbors(method='voronoi')
        sys.calculate_vorovector()
        vor_vec = [atom.vorovector for atom in sys.atoms]
        res_containter.extend(vor_vec)
    data_with_context[cfg.data_path] = res_containter
    return data_with_context 

def calculate_voronoi_face_perimeters(cfg: AnalysisConfig):
    """
    Calculate Voronoi face perimeter data for selected particles.

    Results are flattened across all timesteps selected by ``cfg.chunk``.
    """

    import pyscal as pc

    data_with_context = {}
    data_file=h5py.File(cfg.data_path, "r")
    data=H5DataSelector(data_file,particle_group=cfg.particle_group)

    res_containter=[]
    start, end, step = cfg.chunk
    for col in data.timestep[start:end:step].timestep:
        mask=cfg.particle_predicate(col).flatten() # type: ignore
        posss = col.pos_folded[mask]
        sys = pc.System()
        sys.box = [
            [cfg.box_dim[0], 0.0, 0.0],
            [0.0, cfg.box_dim[1], 0.0],
            [0.0, 0.0, cfg.box_dim[2]]]
        sys.atoms = [pc.Atom(pos=pos_el, id=id_el)
                        for id_el, pos_el in enumerate(posss)]
        sys.find_neighbors(method='voronoi')
        sys.calculate_vorovector()
        face_perimeters = [atom.face_perimeters for atom in sys.atoms]
        res_containter.extend(face_perimeters)
    data_with_context[cfg.data_path] = res_containter
    return data_with_context 

def calculate_voronoi_no_of_edges(cfg: AnalysisConfig):
    """
    Calculate the number of edges in each selected particle's Voronoi cell.

    Results are flattened across all timesteps selected by ``cfg.chunk``.
    """

    import pyscal as pc

    data_with_context = {}
    data_file=h5py.File(cfg.data_path, "r")
    data=H5DataSelector(data_file,particle_group=cfg.particle_group)

    res_containter=[]
    start, end, step = cfg.chunk
    for col in data.timestep[start:end:step].timestep:
        mask=cfg.particle_predicate(col).flatten() # type: ignore
        posss = col.pos_folded[mask]
        sys = pc.System()
        sys.box = [
            [cfg.box_dim[0], 0.0, 0.0],
            [0.0, cfg.box_dim[1], 0.0],
            [0.0, 0.0, cfg.box_dim[2]]]
        sys.atoms = [pc.Atom(pos=pos_el, id=id_el)
                        for id_el, pos_el in enumerate(posss)]
        sys.find_neighbors(method='voronoi')
        sys.calculate_vorovector()
        no_of_edges = [atom.no_of_edges for atom in sys.atoms]
        res_containter.extend(no_of_edges)
    data_with_context[cfg.data_path] = res_containter
    return data_with_context

def calculate_voronoi_vertex_vectors(cfg: AnalysisConfig):
    """
    Calculate Voronoi vertex-vector data for selected particles.

    Results are flattened across all timesteps selected by ``cfg.chunk``.
    """

    import pyscal as pc

    data_with_context = {}
    data_file=h5py.File(cfg.data_path, "r")
    data=H5DataSelector(data_file,particle_group=cfg.particle_group)

    res_containter=[]
    start, end, step = cfg.chunk
    for col in data.timestep[start:end:step].timestep:
        mask=cfg.particle_predicate(col).flatten() # type: ignore
        posss = col.pos_folded[mask]
        sys = pc.System()
        sys.box = [
            [cfg.box_dim[0], 0.0, 0.0],
            [0.0, cfg.box_dim[1], 0.0],
            [0.0, 0.0, cfg.box_dim[2]]]
        sys.atoms = [pc.Atom(pos=pos_el, id=id_el)
                        for id_el, pos_el in enumerate(posss)]
        sys.find_neighbors(method='voronoi')
        sys.calculate_vorovector()
        vertex_vectors = [atom.vertex_vectors for atom in sys.atoms]
        res_containter.extend(vertex_vectors)
    data_with_context[cfg.data_path] = res_containter
    return data_with_context   
    
def write_vtk_frame(cfg: AnalysisConfig, frame=-1):
    """
    Write a single VTK file (ASCII, Unstructured Grid) for a selected frame.

    Selects particles by ``cfg.particle_predicate`` in the given timestep and
    writes positions and dipole vectors to ``cfg.path_to_output``.

    Parameters
    ----------
    cfg : AnalysisConfig
        Must include ``data_path``, ``particle_group``, ``particle_predicate``,
        and ``path_to_output``.
    frame : int, optional
        Timestep index to export (default ``-1`` for the last).

    Returns
    -------
    int
        ``0`` on completion.
    """
    data_file=h5py.File(cfg.data_path, "r")
    data=H5DataSelector(data_file ,particle_group=cfg.particle_group)
    data_per_fram=data.timestep[frame]
    connectivity_values = data_per_fram.get_connectivity_values(cfg.particle_group)
    sel_dataview=data_per_fram.select_particles_by_object(cfg.particle_group, connectivity_values, predicate=cfg.particle_predicate)
    positions=sel_dataview.pos_folded
    dipoles=sel_dataview.dip
    simss=cfg.data_path.split('/')[6]
    frfr=cfg.data_path.split('/')[-1].strip('.h5')
    local_file=f'{cfg.path_to_output}/{frfr}_{simss}.vtk'
    with open(local_file, 'w') as vtk:
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

def write_cluster_to_vtk(cfg: AnalysisConfig):
    """
    Write VTK files for each detected cluster in each processed frame.

    For every selected timestep, builds a graph from selected object
    connectivity when ``cfg.object_predicate`` is provided. Otherwise, it uses
    a neighbour search within ``cfg.crit``. It then writes each connected
    component to an individual VTK file in ``cfg.path_to_output``.

    Parameters
    ----------
    cfg : AnalysisConfig
        Must define ``data_path``, ``particle_group``, ``particle_predicate``,
        ``box_dim``, ``crit``, and ``path_to_output``.

    Returns
    -------
    int
        ``0`` on completion.

    Notes
    -----
    The VTK files are named ``cluster_{cluster_id}_frame_{frame_id}.vtk``.
    """
    data_file=h5py.File(cfg.data_path, "r")
    data=H5DataSelector(data_file,particle_group=cfg.particle_group)
    start, end, step = cfg.chunk
    for frame_id,col in enumerate(data.timestep[start:end:step].timestep):
        connectivity_values = col.get_connectivity_values(
            cfg.particle_group, predicate=cfg.object_predicate)
        sel_dataview=col.select_particles_by_object(
            cfg.particle_group, connectivity_values, predicate=cfg.particle_predicate)
        if cfg.object_predicate is None:
            posss = sel_dataview.pos_folded
            connectivity_list=get_neighbours(posss,cfg.box_dim[0],cfg.crit)
            edges=[(part, neighbour)
                   for part, neighbours in connectivity_list.items()
                   for neighbour in neighbours]
        else:
            selected_ids = np.asarray(sel_dataview.id).flatten()
            id_to_local = {int(particle_id): index
                           for index, particle_id in enumerate(selected_ids)}
            edges=[]
            for object_id in connectivity_values:
                object_ids = col.select_particles_by_object(
                    cfg.particle_group, object_id,
                    predicate=cfg.particle_predicate).id.flatten()
                edges.extend((id_to_local[int(x)], id_to_local[int(y)])
                             for x, y in pairwise(object_ids))
        graph_iterator=context.get_cluster_iterator(
            sel_dataview, edges, cfg.box_dim,
            attibutes=['pos','pos_folded','dip'])
        for cluster_id, subgraph in enumerate(graph_iterator):
            positions=subgraph.vs['pos_folded_unbroken']
            dipoles=subgraph.vs['dip']
            local_file=f'{cfg.path_to_output}/cluster_{cluster_id}_frame_{frame_id}.vtk'
            with open(local_file, 'w') as vtk:
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

def get_data_timestep_len(cfg: AnalysisConfig):
    """
    Get the number of timesteps in the selected particle group.

    Parameters
    ----------
    cfg : AnalysisConfig
        Must include ``data_path`` and ``particle_group``.

    Returns
    -------
    dict
        ``{cfg.data_path: int}`` mapping to the length of ``data.timestep``.
    """
    
    data_with_context = {}
    data_file=h5py.File(cfg.data_path, "r")
    data=H5DataSelector(data_file,particle_group=cfg.particle_group)
    data_with_context[cfg.data_path] = len(data.timestep)
    return data_with_context
