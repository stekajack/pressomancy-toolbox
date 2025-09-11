from pmtools.resources.gryation_tensor import GyrationTensor
import numpy as np
from itertools import pairwise
import pmtools.refractored_toolbox as context
from pmtools.resources.kernel_config import AnalysisConfig
from pressomancy.analysis import H5DataSelector
from pressomancy.helper_functions import get_neighbours_cross_lattice
import h5py
# object_predicate= lambda subset: not (subset.type == 5).any()
# cfg.particle_predicate
def per_fil_gyr(cfg: AnalysisConfig):

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
            accumulated_gts.append(GyrationTensor(np.array(subgraph.vs['pos'])))
       
    data_with_context[cfg.data_path] = accumulated_gts
    return data_with_context


def lp_projection(cfg: AnalysisConfig):

    data_with_context = {}
    data_file=h5py.File(cfg.data_path, "r")
    data=H5DataSelector(data_file,particle_group=cfg.particle_group)
    monomer_no = int(context.determine_key_val_from_filename(cfg.template_hndl,cfg.data_path,'what_monomer_number'))
    accumulated_lp_seg = []
    start, end, step = cfg.chunk
    for col in data.timestep[start:end:step].timestep:
        fitered_fil_ids=col.get_connectivity_values(cfg.particle_group, predicate=cfg.object_predicate)
        
        pf_indices = [col.select_particles_by_object(cfg.particle_group, myed,predicate=cfg.particle_predicate).id.flatten() for myed in fitered_fil_ids]

        edges = [(int(x), int(y)) for pf_el in pf_indices for x,y in pairwise(pf_el)]
        graph_iterator=context.get_cluster_iterator(col.select_particles_by_object(cfg.particle_group, fitered_fil_ids,predicate=cfg.particle_predicate), edges, cfg.box_dim)
        for subgraph in graph_iterator:
            positions=np.array(subgraph.vs['pos'])
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


def calculate_stacking_fraction(cfg: AnalysisConfig):
    
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
    Calculate the structure factor for a given HDF5 data file.
    Uses the `sq_avx` module for efficient computation. See https://github.com/stekajack/espressoSq
    """

    import sq_avx

    data_with_context = {}
    data_file=h5py.File(cfg.data_path, "r")
    data=H5DataSelector(data_file,particle_group=cfg.particle_group)

    wavevectors_container, intensities_container = [], []
    start, end, step = cfg.chunk
    for col in data.timestep[start:end:step].timestep:
        posss = col.pos_folded
        types = col.type.flatten()
        mask = types != 5
        posss = posss[mask]
        wavevectors, intensities = sq_avx.calculate_structure_factor(
            posss, 120, cfg.box_dim[0], 100, 40)
        wavevectors_container.append(wavevectors)
        intensities_container.append(intensities)

    data_with_context[cfg.data_path] = wavevectors_container, intensities_container
    return data_with_context  

def write_vtk_frame(cfg: AnalysisConfig, frame=-1):
    data_file=h5py.File(cfg.data_path, "r")
    data=H5DataSelector(data_file ,particle_group=cfg.particle_group)
    data_per_fram=data.timestep[frame]
    positions=data_per_fram.pos_folded
    dipoles=data_per_fram.dip
    with open(cfg.path_to_output, 'w') as vtk:
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
    
    data_with_context = {}
    data_file=h5py.File(cfg.data_path, "r")
    data=H5DataSelector(data_file,particle_group=cfg.particle_group)
    data_with_context[cfg.data_path] = len(data.timestep)
    return data_with_context
