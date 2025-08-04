from pmtools.resources.gryation_tensor import GyrationTensor
import numpy as np
import igraph as ig
from itertools import pairwise
import pmtools.refractored_toolbox as context
from pressomancy.analysis import H5DataSelector
from pressomancy.helper_functions import get_neighbours_cross_lattice
import h5py

def per_fil_gyr_h5_modern(data_path, template_hndl, box_dim, chunk=(-5, None, 1), norm=1., crit=1.47, extra_flag=None):

    data_with_context = {}
    data_file=h5py.File(data_path, "r")
    data=H5DataSelector(data_file, particle_group="Filament")
    monomer_no = int(context.determine_key_val_from_filename(template_hndl,data_path,'what_monomer_number'))
    accumulated_gts = []
    start, end, step = chunk
    for col in data.timestep[start:end:step].timestep:
        fitered_fil_ids=[]
        for myed in list(col.get_connectivity_values('Filament')):
            parts=col.select_particles_by_object('Filament',myed)
            types=parts.type.flatten()
            if 5 not in types:
                fitered_fil_ids.append(myed)

        fitered_fil_ids.sort()
        pf_indices=[]
        filtered_pos=[]
        all_pos=[]
        for myed in fitered_fil_ids:
            ids_shuffled=col.select_particles_by_object('Filament',myed).id.flatten()
            pos_shuffled=col.select_particles_by_object('Filament',myed).pos
            order = np.argsort(ids_shuffled)
            ids_ordered = ids_shuffled[order]
            pos_ordered = pos_shuffled[order]
            # Remove patches for the iGraph unfolding to work correctly. 
            # Indices must be increasing monotonically (patches do not w.r.t rest of part)
            pf_indices.append(ids_ordered[:-2*monomer_no])
            filtered_pos.extend(pos_ordered[:-2*monomer_no])
            all_pos.extend(pos_ordered)

        # posss = [context.fold_coordinates_pp(x, box_dim=box_dim) for x in filtered_pos]
        edges = [(int(x), int(y)) for pf_el in pf_indices for x,
                     y in pairwise(pf_el)]
        g2 = ig.Graph(n=len(all_pos), edges=edges)
        g2.vs["pos"] = all_pos
        g2.simplify()
        decomposition = g2.decompose()
        for subgraph in decomposition:
            flag, pass_graph = context.check_breakage(
                subgraph, box_dim)
            if not flag:
                positions = context.unbreak_graph(
                    pass_graph, box_dim)
            else:
                positions = np.array(subgraph.vs['pos'])
            accumulated_gts.append(GyrationTensor(positions))
    data_with_context[data_path] = accumulated_gts
    return data_with_context

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

def get_data_timestep_len(data_path, particle_group):
    
    data_with_context = {}
    data_file=h5py.File(data_path, "r")
    data=H5DataSelector(data_file,particle_group=particle_group)
    data_with_context[data_path] = len(data.timestep)
    return data_with_context

def calculate_stacking_fraction(data_path, template_hndl, box_dim, chunk=(-5, None, 1), norm=1., crit=1.47, extra_flag=None):
    
    data_with_context = {}
    data_file=h5py.File(data_path, "r")
    data=H5DataSelector(data_file,particle_group="Filament")
    data_crowder=H5DataSelector(data_file,particle_group="Crowder")
    
    start, end, step = chunk
    data_per_timestep=[]
    for col_fil,col_crow in zip(data.timestep[start:end:step].timestep, data_crowder.timestep[start:end:step].timestep):
        mask_stack=col_fil.particles[:].type==4
        mask_ligand=col_crow.particles[:].type==5
        
        mask_stack=np.arange(len(mask_stack.flatten()))[mask_stack.flatten()]
        mask_ligand=np.arange(len(mask_ligand.flatten()))[mask_ligand.flatten()]
        
        stacking_sites=col_fil.particles[list(mask_stack)]
        ligands=col_crow.particles[list(mask_ligand)]
    
        grouped_indices=get_neighbours_cross_lattice(ligands.pos,stacking_sites.pos, box_dim[0],crit)
        data_per_timestep.append(grouped_indices)
        
    data_with_context[data_path] = data_per_timestep
    return data_with_context

def lp_prjection_h5(data_path, template_hndl, box_dim, chunk=(-5, None, 1), norm=1., crit=1.47, extra_flag=None):

    data_with_context = {}
    data_file=h5py.File(data_path, "r")
    data=H5DataSelector(data_file,particle_group="Filament")
    monomer_no = int(context.determine_key_val_from_filename(template_hndl,data_path,'what_monomer_number'))
    accumulated_lp_seg = []
    start, end, step = chunk
    for col in data.timestep[start:end:step].timestep:
        fitered_fil_ids=[]
        for myed in list(col.get_connectivity_values('Filament')):
            parts=col.select_particles_by_object('Filament',myed)
            types=parts.type.flatten()
            if 5 not in types:
                fitered_fil_ids.append(myed)

        fitered_fil_ids.sort()
        pf_indices=[]
        filtered_pos=[]
        all_pos=[]
        for myed in fitered_fil_ids:
            ids_shuffled=col.select_particles_by_object('Filament',myed).id.flatten()
            pos_shuffled=col.select_particles_by_object('Filament',myed).pos
            order = np.argsort(ids_shuffled)
            ids_ordered = ids_shuffled[order]
            pos_ordered = pos_shuffled[order]
            # Remove patches for the iGraph unfolding to work correctly. 
            # Indices must be increasing monotonically (patches do not w.r.t rest of part)
            pf_indices.append(ids_ordered[:-2*monomer_no])
            filtered_pos.extend(pos_ordered[:-2*monomer_no])
            all_pos.extend(pos_ordered)
            filtered_pos.extend(pos_ordered[:-2*monomer_no])

        edges = [(int(x), int(y)) for pf_el in pf_indices for x,
                     y in pairwise(pf_el)]
        g2 = ig.Graph(n=len(all_pos), edges=edges)
        g2.vs["pos"] = all_pos
        g2.simplify()
        decomposition = g2.decompose()
        for subgraph in decomposition:
            flag, pass_graph = context.check_breakage(
                subgraph, box_dim)
            if not flag:
                positions = context.unbreak_graph(
                    pass_graph, box_dim)
            else:
                positions = np.array(subgraph.vs['pos'])

            com_pos = np.mean(positions.reshape(
                monomer_no, -1, 3), axis=1)
            ete_vec = com_pos[-1]-com_pos[0]
            segments = np.diff(com_pos, axis=0)
            seg_norms = np.mean(np.linalg.norm(segments, axis=1))
            res = np.dot(segments, ete_vec)/pow(seg_norms,2)
            accumulated_lp_seg.append(res)            
    xax = np.arange(monomer_no-1)+1
    data_with_context[data_path] = np.mean(
        accumulated_lp_seg, axis=0), xax
    return data_with_context

def calculate_sf_h5(data_path, template_hndl, box_dim, chunk=(-5, None, 1), norm=1., crit=1.47, extra_flag=None):]

    """
    Calculate the structure factor for a given HDF5 data file.
    Uses the `sq_avx` module for efficient computation. See https://github.com/stekajack/espressoSq
    """

    import sq_avx

    data_with_context = {}
    data_file=h5py.File(data_path, "r")
    data=H5DataSelector(data_file,particle_group="Filament")

    wavevectors_container, intensities_container = [], []
    start, end, step = chunk
    for col in data.timestep[start:end:step].timestep:
        posss = col.pos_folded
        types = col.types.flatten()
        mask = types != 5
        posss = posss[mask]
        wavevectors, intensities = sq_avx.calculate_structure_factor(
            posss, 120, box_dim[0], 100, 40)
        wavevectors_container.append(wavevectors)
        intensities_container.append(intensities)

    data_with_context[data_path] = wavevectors_container, intensities_container
    return data_with_context