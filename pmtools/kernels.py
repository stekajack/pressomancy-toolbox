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
        
        pf_indices = [col.select_particles_by_object(cfg.particle_group, myed,predicate=cfg.particle_predicate).id.flatten() for myed in fitered_fil_ids]

        edges = [(int(x), int(y)) for pf_el in pf_indices for x,y in pairwise(pf_el)]
        graph_iterator=context.get_cluster_iterator(col.select_particles_by_object(cfg.particle_group, fitered_fil_ids,predicate=cfg.particle_predicate), edges, cfg.box_dim)
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
        sel_dataview=col.select_particles_by_predicate(cfg.particle_group, predicate=cfg.particle_predicate)
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
        sel_dataview=col.select_particles_by_predicate(cfg.particle_group, predicate=cfg.particle_predicate)
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

def pair_distances(cfg: AnalysisConfig):

    data_with_context = {}
    data_file=h5py.File(cfg.data_path, "r")
    data=H5DataSelector(data_file, particle_group=cfg.particle_group)
    accumulated_gts = []
    start, end, step = cfg.chunk
    for col in data.timestep[start:end:step].timestep:
        sel_dataview=col.select_particles_by_predicate(cfg.particle_group, predicate=cfg.particle_predicate)
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
        pf_indices = [list(range(x, x + 20))
                                  for x in range(0, len(part_sel.particles), 20)]
        # print(pf_indices)

        edges = [(int(x), int(y)) for pf_el in pf_indices for x,y in pairwise(pf_el)]
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

# def easy_dipmom_angle(cfg: AnalysisConfig):
    
#     data_with_context = {}
#     data_file=h5py.File(cfg.data_path, "r")
#     data=H5DataSelector(data_file,particle_group=cfg.particle_group)
#     monomer_no = int(context.determine_key_val_from_filename(cfg.template_hndl,cfg.data_path,'what_monomer_number'))
#     accumulated_lp_seg = []
#     start, end, step = cfg.chunk
#     for col in data.timestep[start:end:step].timestep:
#         fitered_fil_ids=col.get_connectivity_values(cfg.particle_group, predicate=cfg.object_predicate)
#         part_sel_mag=col.select_particles_by_object(cfg.particle_group, fitered_fil_ids,predicate=cfg.particle_predicate)

#         def select_easy(subset):
#             return subset.type == 2
        
#         part_sel_easy=col.select_particles_by_object(cfg.particle_group, fitered_fil_ids, predicate=select_easy)
        
#         pf_indices_mag = [list(range(x, x + 20))
#                                   for x in range(0, len(part_sel_mag.particles), 20)]
#         pf_indices_easy = [list(range(x, x + 40))
#                                   for x in range(0, len(part_sel_easy.particles), 40)]
#         edges_mag = [(int(x), int(y)) for pf_el in pf_indices_mag for x,y in pairwise(pf_el)]
#         edges_easy = [(int(x), int(y)) for pf_el in pf_indices_easy for x,y in pairwise(pf_el)]
#         assert np.shape(pf_indices_easy)==(210,40), "Expected 210 easy filaments of 40 particles each"
#         graph_iterator_mag=context.get_cluster_iterator(part_sel_mag, edges_mag, cfg.box_dim, attibutes=['pos_folded','dip'], min_part=20)
#         graph_iterator_easy=context.get_cluster_iterator(part_sel_easy, edges_easy, cfg.box_dim, attibutes=['pos_folded',], min_part=40)
#         for subgraph_mag, subgraph_easy in zip(graph_iterator_mag,graph_iterator_easy):
#             positions=np.array(subgraph_easy.vs['pos_folded_unbroken'])
#             assert np.shape(positions)==(40,3), "Expected 40 particles per easy filament"
#             p1 = positions[0::2]      # indices 0,2,4,...
#             p2 = positions[1::2]      # indices 1,3,5,...
#             segments = p2 - p1      # shape (20, 3)
           
#             seg_norms = np.linalg.norm(segments, axis=1)
#             assert all(seg_norms<3.), f"Segment norm too large, something is wrong! seg_norms={seg_norms}"
#             segments = segments / seg_norms[:, None]
#             dipoles=np.array(subgraph_mag.vs['dip'])
#             dipole_norms = np.linalg.norm(dipoles, axis=1)
#             dipoles=dipoles/dipole_norms[:, None]
#             res_cors = np.einsum('ij,ij->i', dipoles, segments) 
#             accumulated_lp_seg.extend(res_cors)            
#     data_with_context[cfg.data_path] = accumulated_lp_seg
#     return data_with_context


def easy_dipmom_angle(cfg: AnalysisConfig):
    
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
        pf_indices = [list(range(x, x + 20))
                                  for x in range(0, len(part_sel.particles), 20)]
        # print(pf_indices)

        edges = [(int(x), int(y)) for pf_el in pf_indices for x,y in pairwise(pf_el)]
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
        pf_indices = [col.select_particles_by_object(cfg.particle_group, myed,predicate=cfg.particle_predicate).id.flatten() for myed in fitered_fil_ids]
        edges = [(int(x), int(y)) for pf_el in pf_indices for x,y in pairwise(pf_el)]
        graph_iterator=context.get_cluster_iterator(col.select_particles_by_object(cfg.particle_group, fitered_fil_ids,predicate=cfg.particle_predicate), edges, cfg.box_dim, attibutes=['pos_folded','dip'])
        for subgraph in graph_iterator:
            dipoles=np.mean(subgraph.vs['dip'],axis=0)[-1]/float(cfg.norm)
            accumulated_magnetisation.append(dipoles)            
    data_with_context[cfg.data_path] = accumulated_magnetisation
    return data_with_context 

def magn_princip_angle_dist(cfg: AnalysisConfig):

    data_with_context = {}
    data_file=h5py.File(cfg.data_path, "r")
    data=H5DataSelector(data_file,particle_group=cfg.particle_group)
    accumulated_magnetisation = []
    start, end, step = cfg.chunk
    for col in data.timestep[start:end:step].timestep:
        fitered_fil_ids=col.get_connectivity_values(cfg.particle_group, predicate=cfg.object_predicate)
        part_sel=col.select_particles_by_object(cfg.particle_group, fitered_fil_ids,predicate=cfg.particle_predicate)
        pf_indices = [col.select_particles_by_object(cfg.particle_group, myed,predicate=cfg.particle_predicate).id.flatten() for myed in fitered_fil_ids]
        pf_indices = [list(range(x, x + 20))
                                  for x in range(0, len(part_sel.particles), 20)]
        edges = [(int(x), int(y)) for pf_el in pf_indices for x,y in pairwise(pf_el)]
        graph_iterator=context.get_cluster_iterator(part_sel, edges, cfg.box_dim, attibutes=['pos_folded','dip'])
        reference=np.array([0,0,1])
        for subgraph in graph_iterator:
            chain_dip_mom=np.mean(subgraph.vs['dip'],axis=0)/float(cfg.norm)
            gt=GyrationTensor(np.array(subgraph.vs['pos_folded_unbroken'])) 
            res_angle=vg.angle(chain_dip_mom,vg.aligned_with(gt.eigenvectors[-1], reference, reverse=False), units='deg')
            accumulated_magnetisation.append(np.minimum(res_angle, 180 - res_angle))            
    data_with_context[cfg.data_path] = accumulated_magnetisation
    return data_with_context 

def magnetisation_culster_size(cfg: AnalysisConfig):

    data_with_context = {}
    data_file=h5py.File(cfg.data_path, "r")
    data=H5DataSelector(data_file,particle_group=cfg.particle_group)
    accumulated_magnetisation = []
    start, end, step = cfg.chunk
    for col in data.timestep[start:end:step].timestep:
        sel_dataview=col.select_particles_by_predicate(cfg.particle_group, predicate=cfg.particle_predicate)
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

    data_with_context = {}
    data_file=h5py.File(cfg.data_path, "r")
    data=H5DataSelector(data_file,particle_group=cfg.particle_group)
    accumulated_magnetisation = []
    start, end, step = cfg.chunk
    for col in data.timestep[start:end:step].timestep:
        sel_dataview=col.select_particles_by_predicate(cfg.particle_group, predicate=cfg.particle_predicate)
        posss = sel_dataview.pos_folded
        connectivity_list=get_neighbours(posss,cfg.box_dim[0],cfg.crit)
        edges=[]
        for part,niegh_parts in connectivity_list.items():
            for niegh in niegh_parts:
                edges.append((part,niegh))
        g2 = ig.Graph(n=len(data.particles), edges=edges)
        attibutes=['pos_folded','dip']
        for att in attibutes:
            g2.vs[att] = getattr(data,att)
        g2.simplify()
        edges_list=g2.get_edgelist()
        edges_filtered=[]
        for x,y in edges_list:
            print(x,y)
            print(np.shape(g2.vs[x]['pos_folded']))
            res=pair_potential(min_img_dist(g2.vs[x]['pos_folded'], g2.vs[y]['pos_folded'], cfg.box_dim), g2.vs[x]['dip'], g2.vs[y]['dip'])
            if res <=-0.1:
                edges_filtered.append((x,y))
        g3 = ig.Graph(n=len(data.particles), edges=edges_filtered)
        attibutes=['pos_folded','dip']
        for att in attibutes:
            g3.vs[att] = getattr(data,att)
        g3.simplify()
        graph_iterator=context.get_cluster_iterator(sel_dataview, edges, cfg.box_dim, min_part=20,attibutes=['pos_folded','dip'])
        for verterx in g3.vs:
            accumulated_magnetisation.append((len(verterx.neighbors()),verterx['dip'][-1]))            
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
          ``'order'``, ``'orientations_per_wavevector'``, ``'subsample_every'``,
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

    wavevectors_container, intensities_container = [], []
    start, end, step = cfg.chunk
    for col in data.timestep[start:end:step].timestep:
        mask=cfg.particle_predicate(col).flatten() # type: ignore
        posss = col.pos_folded[mask]
        wavevectors, intensities = sq_avx.calculate_structure_factor(
            posss, cfg.sq_params['order'], cfg.box_dim[0], cfg.sq_params['orientations_per_wavevector'], cfg.sq_params['subsample_every'])
        wavevectors_container.append(wavevectors)
        intensities_container.append(intensities)

    data_with_context[cfg.data_path] = wavevectors_container, intensities_container
    return data_with_context

def calculate_rdf(cfg: AnalysisConfig):

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


def write_vtf_from_particles(particles, box_dim, out_path):
    """
    Write a minimal VTF file for VMD from (id, position, type) tuples.
    """
    out_path = Path(out_path)
    particles = list(particles)

    if not particles:
        out_path.write_text("vtf 1.00\n")
        return out_path

    positions = np.array([p[1] for p in particles], dtype=float)
    types = [int(p[2]) for p in particles]

    lines = []
    lines.append(f"unitcell {box_dim[0]} {box_dim[1]} {box_dim[2]}\n")
    lines.append(f"atom 0:{len(particles)-1} radius 0.5 name A\n")
    for idx, t in enumerate(types):
        lines.append(f"atom {idx} type {t}\n")
    lines.append("timestep\n")
    for x, y, z in positions:
        lines.append(f"{x:.6f} {y:.6f} {z:.6f}\n")

    out_path.write_text("".join(lines))
    return out_path


def mega_giga_analysis(cfg: AnalysisConfig):
    import pyscal as pc

    data_with_context = {}
    data_file=h5py.File(cfg.data_path, "r")
    data=H5DataSelector(data_file,particle_group=cfg.particle_group)

    start, end, step = cfg.chunk
    for col in data.timestep[start:end:step].timestep:
        predicate_mask=cfg.particle_predicate(col).flatten() # type: ignore
        posss = col.pos_folded[predicate_mask]
        sys = pc.System()
        sys.box = [
            [cfg.box_dim[0], 0.0, 0.0],
            [0.0, cfg.box_dim[1], 0.0],
            [0.0, 0.0, cfg.box_dim[2]]]
        sys.atoms = [pc.Atom(pos=pos_el, id=id_el)
                        for id_el, pos_el in enumerate(posss)]
        sys.find_neighbors(method='voronoi')
        atom_ids= np.array([atom.id for atom in sys.atoms])
        vor_vol = np.array([atom.volume for atom in sys.atoms])
        valid_mask=vor_vol<20
        face_perimeters = [atom.face_perimeters for atom in sys.atoms]
        voronoi_face_perimeters_data_selected=[len(x) for id,x in enumerate(face_perimeters) if valid_mask[id]]
        no_of_edges = [atom.no_of_edges for atom in sys.atoms]
        voronoi_no_of_edges_data_selected=[x for id,x in enumerate(no_of_edges) if valid_mask[id]]
        vertex_vectors = [atom.vertex_vectors for atom in sys.atoms]
        voronoi_vertex_vectors_data_selected=[x for id,x in enumerate(vertex_vectors) if valid_mask[id]]
        voronoi_vertex_vectors_data_selected=[len(x)/3 for x in voronoi_vertex_vectors_data_selected]
        V = np.array(voronoi_vertex_vectors_data_selected)           # shape (N,)
        E = np.array(voronoi_no_of_edges_data_selected)              # shape (N,)
        F = np.array(voronoi_face_perimeters_data_selected)          # shape (N,)

        xi=V-E+F
        assert all(xi==2),'cant have non-covex voronoi cells!'

        if len(V):
            # Pack into structured array of shape (N, 3)
            polyhedra = np.stack([V, E, F], axis=1)   # shape (N,3)

            # Get unique (V,E,F) triples and their counts
            unique_polyhedra, counts = np.unique(polyhedra, axis=0, return_counts=True)

            # Probabilities / frequencies
            probs = counts / counts.sum()

            # Sort by probability descending
            idx = np.argsort(probs)[::-1]
            unique_polyhedra = unique_polyhedra[idx]
            probs = probs[idx]
            print('unique_polyhedra: ',unique_polyhedra)
            print('probs: ',probs)

        else:
            polyhedra = np.empty((0, 3))
            unique_polyhedra = np.empty((0, 3))
            probs = np.array([])

        # Enumerate the six most probable unique polyhedra starting at 347
        top_k = min(6, len(unique_polyhedra))
        enumerated_polyhedra = []
        enumeration_lookup = {}
        for offset, poly in enumerate(unique_polyhedra[:top_k]):
            enum_value = 347 + offset
            poly_key = tuple(int(val) for val in poly)
            enumerated_polyhedra.append((enum_value, poly_key))
            enumeration_lookup[poly_key] = enum_value

        # Map each selected particle id to its polyhedron and enumerated type
        print('enumeration_lookup:', enumeration_lookup)
        particle_polyhedra_types = []
        particles_for_vtf = []
        poly_idx = 0
        for atom_id, pos, is_valid in zip(atom_ids.tolist(), posss.tolist(), valid_mask.tolist()):
            if is_valid and poly_idx < len(polyhedra):
                poly = polyhedra[poly_idx]
                poly_idx += 1
                poly_key = tuple(int(val) for val in poly)
                enum_value = enumeration_lookup.get(poly_key, 1)
            else:
                poly_key = (-1, -1, -1)
                enum_value = 765
            particle_polyhedra_types.append((int(atom_id), poly_key, int(enum_value)))
            particles_for_vtf.append((int(atom_id), tuple(float(x) for x in pos), int(enum_value)))

        if top_k:
            assert probs[:top_k].sum()>0.8
        vtf_out = Path(cfg.path_to_output) if cfg.path_to_output else Path(cfg.data_path).with_suffix('.vtf')
        write_vtf_from_particles(particles_for_vtf, cfg.box_dim, vtf_out)
        data_with_context[cfg.data_path] = {
            'top_polyhedra': enumerated_polyhedra,
            'particle_polyhedra_types': particle_polyhedra_types,
            'probabilities': probs[:top_k],
            'vtf_path': vtf_out,
        }
        break

    return data_with_context

def calculate_volume_voronoi(cfg: AnalysisConfig):

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
    sel_dataview=data_per_fram.select_particles_by_predicate(cfg.particle_group, predicate=cfg.particle_predicate)
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

    For every selected timestep, builds a connectivity graph using a neighbour
    search (within ``cfg.crit`` and periodic box ``cfg.box_dim``), iterates
    over clusters via ``context.get_cluster_iterator``, and writes particle
    positions and dipoles for each cluster to individual VTK files in
    ``cfg.path_to_output``.

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
        sel_dataview=col.select_particles_by_predicate(cfg.particle_group, predicate=cfg.particle_predicate)
        posss = sel_dataview.pos_folded
        connectivity_list=get_neighbours(posss,cfg.box_dim[0],cfg.crit)
        edges=[]
        for part,niegh_parts in connectivity_list.items():
            for niegh in niegh_parts:
                edges.append((part,niegh))
        graph_iterator=context.get_cluster_iterator(sel_dataview, edges, cfg.box_dim, attibutes=['pos','pos_folded','dip'])
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
