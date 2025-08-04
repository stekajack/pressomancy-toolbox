from itertools import product
import numpy as np
import igraph as ig
from string import Template
import re

def determine_key_val_from_filename(template, data_path, key):
    """
    Extract simulation parameters from a filename and compute box dimensions.

    The function converts a Template (or template string) into a regex pattern
    using `template_to_regex` and extracts parameters from the provided filename.

    Parameters
    ----------
    template : Template or str
        A filename template containing placeholders.
    data_path : str
        The filename or path matching the template.
    key : str
        The key of the paramter, matching the template.
    

    Returns
    -------
    float

    Raises
    ------
    LookupError
        If the filename does not match the provided template.
    """
    
    regex_pattern = template_to_regex(template)
    match = regex_pattern.fullmatch(data_path)
    if not match:
        raise LookupError("Failed to extract parameters from templated filename (no matching key)")
        
    params = match.groupdict()
    
    return float(params[key])

def get_template_keys(template_str):
    """
    Extract placeholder keys from a template string.
    Returns a set containing the names of all placeholders.
    """
    pattern = Template.pattern  # precompiled regex pattern for Template
    keys = set()
    for match in re.finditer(pattern, template_str):
        # Either the 'named' group or the 'braced' group holds the key.
        key = match.group('named') or match.group('braced')
        if key:  # Only add if a key was found
            keys.add(key)
    return keys

def template_to_regex(template_obj):
    """
    Converts a string.Template object (or template string) into a regex pattern
    that can parse a formatted string back into its component fields.
    
    This version includes a modification for fields that may include literal characters
    (like '/') in their value. For example, if the field 'what_am_I_looking_at' is
    immediately followed by '/' in the template, we use a greedy match to capture the
    entire value (e.g., 'chains/ligands') rather than stopping at the first '/'.
    """
    # If given a Template object, extract its underlying string.
    if isinstance(template_obj, Template):
        template_str = template_obj.template
    else:
        template_str = template_obj

    # Find all placeholder matches in the template using Template.pattern.
    matches = list(re.finditer(Template.pattern, template_str))
    regex_parts = []
    last_end = 0

    for i, match in enumerate(matches):
        # Add literal text preceding the placeholder.
        literal_before = template_str[last_end:match.start()]
        regex_parts.append(re.escape(literal_before))
        
        if match.group('escaped'):
            # Escaped dollar sign ($$): match a literal '$'.
            regex_parts.append(re.escape('$'))
        elif match.group('named') or match.group('braced'):
            # Get the placeholder name.
            field_name = match.group('named') or match.group('braced')
            
            # Determine the literal text that follows this placeholder.
            if i + 1 < len(matches):
                next_literal = template_str[match.end():matches[i+1].start()]
            else:
                next_literal = template_str[match.end():]
            
            if next_literal:
                # For ambiguous cases, for instance when the literal is "/" and the field
                # might include '/', use a greedy match.
                if next_literal == "/" and field_name == "what_am_I_looking_at":
                    regex_parts.append(f"(?P<{field_name}>.*)(?={re.escape(next_literal)})")
                else:
                    # Use a non-greedy match by default.
                    regex_parts.append(f"(?P<{field_name}>.*?)(?={re.escape(next_literal)})")
            else:
                # Capture the rest of the string.
                regex_parts.append(f"(?P<{field_name}>.*)")
        else:
            # Fallback: treat the match as literal text.
            regex_parts.append(re.escape(match.group()))
        
        last_end = match.end()

    # Append any remaining literal text after the last placeholder.
    regex_parts.append(re.escape(template_str[last_end:]))
    
    # Join all parts and compile the regex.
    full_regex = ''.join(regex_parts)
    return re.compile(full_regex)

def assemble_paths(master_dict, template_hndl, parallel_param_id=None):
    """
    Generate file paths by substituting all combinations of parameters into a template.
    
    If additional keys are present in master_dict that are not in the template,
    they are ignored in order to avoid unnecessary duplicates.
    
    Parameters
    ----------
    master_dict : dict
        Dictionary where keys are parameter names and values are iterables of possible values.
    template_hndl : Template
        A string.Template instance used for assembling the file paths.
    parallel_param_id : str, optional
        If provided, specifies the key in master_dict that should be expanded separately.
    
    Returns
    -------
    tuple
        (keys_assembler, paths_to_calc_with) where:
          - keys_assembler is a list of paths generated from the Cartesian product of the filtered parameters.
          - paths_to_calc_with is a list (or list of lists) of paths generated after expanding the parallel parameter.
    """
    
    # Extract only the keys that are present in the template.
    template_keys = get_template_keys(template_hndl.template)
    
    # Filter master_dict to include only keys present in the template.
    # Note: If parallel_param_id is specified, we expect it to be in the template.
    filtered_master = { key: master_dict[key] for key in master_dict if key in template_keys }
    
    # Make a local copy to avoid modifying the original dictionary.
    local_master_dict = filtered_master.copy()
    
    # Get the keys that we will use in the Cartesian product.
    keys = list(local_master_dict.keys())
    keys_assembler = []
    
    # Replace the parallel parameter with a placeholder if one was provided.
    if parallel_param_id:
        local_master_dict[parallel_param_id] = ('parallel_param_placeholer',)
    
    # Compute Cartesian product over values of the filtered dictionary.
    value_combinations = product(*local_master_dict.values())
    for combo in value_combinations:
        mapping = dict(zip(keys, combo))
        keys_assembler.append(template_hndl.substitute(**mapping))
    
    paths_to_calc_with = []    
    if parallel_param_id:
        # For each base path, expand the parallel parameter with its actual values.
        for path in keys_assembler:
            tmp_list = []
            for val in master_dict[parallel_param_id]:
                tmp_path = path.replace('parallel_param_placeholer', val)
                tmp_list.append(tmp_path)
            paths_to_calc_with.append(tmp_list)
    else:
        paths_to_calc_with = keys_assembler
    
    return keys_assembler, paths_to_calc_with

def determine_box_dim_from_filename(template, data_path, key_concentration, key_obj_no):
    """
    Extract simulation parameters from a filename and compute box dimensions.

    The function converts a Template (or template string) into a regex pattern
    using `template_to_regex` and extracts parameters from the provided filename.
    It uses the extracted key_concentration and key_obj_no to obtain concentration
    and object number values to calculate the box dimensions.

    Parameters
    ----------
    template : Template or str
        A filename template containing placeholders.
    data_path : str
        The filename or path matching the template.

    Returns
    -------
    numpy.ndarray
        A 3-element array [box_l, box_l, box_l] representing the box dimensions.

    Raises
    ------
    LookupError
        If the filename does not match the provided template.
    """
    regex_pattern = template_to_regex(template)
    match = regex_pattern.fullmatch(data_path)
    if not match:
        raise LookupError("Failed to extract parameters from templated filename (no matching key)")
        
    params = match.groupdict()
    concentration = float(params[key_concentration])
    no_obj = int(params[key_obj_no])
    N_avog = 6.02214076e23  # Avogadro's number
    rho_si = concentration * N_avog
    N = int(no_obj / 3)
    vol = N / rho_si
    box_l = pow(vol, 1/3) / 0.4e-09
    box_dim = box_l * np.ones(3)
        
    return box_dim

def check_breakage(graph_el, box_dim):
    # warnings.warn(f"The coordinates passed to check_breakage must be folded!. The logic doesnt work otherwise!")
    positions = np.array(graph_el.vs['pos'])
    edgers_filtered = [(pair_el1, pair_el2) for pair_el1, pair_el2 in graph_el.get_edgelist(
    ) if all(abs(positions[pair_el1]-positions[pair_el2]) < box_dim/2)]

    g_temp = ig.Graph(n=len(graph_el.vs), edges=edgers_filtered,
                      vertex_attrs={'pos': positions})
    g_temp.simplify()
    decomposition_bla = g_temp.decompose()
    return len(decomposition_bla) == 1, g_temp

def unbreak_graph(broken_graph, box_dim):
    """
    Stitch broken clusters in a graph back together.

    Parameters:
    - broken_graph: igraph.Graph
        Graph with broken clusters.

    Returns:
    - pos_flat: np.ndarray
        Flattened positions of the graph with broken clusters stitched back together.
    """
    # warnings.warn(f"The coordinates passed to unbreak_graph must be folded!. The logic doesnt work otherwise!")

    decomposition = broken_graph.decompose()
    pos_flat = np.concatenate([el.vs['pos'] for el in decomposition])

    means_and_positions = [(len(el.vs['pos']), np.mean(
        el.vs['pos'], axis=0)) for el in decomposition]
    start_idx = means_and_positions[0][0]
    # Start from index 1 to skip the first cluster
    for i in range(1, len(means_and_positions)):
        num_vertices, mean_pos = means_and_positions[i]
        mean_shift = means_and_positions[0][1] - mean_pos
        exceed_limit = np.abs(mean_shift) > box_dim / 2
        fin_shift = np.zeros(3)
        fin_shift[exceed_limit] = np.sign(
            mean_shift[exceed_limit]) * box_dim[exceed_limit]
        pos_flat[start_idx:start_idx + num_vertices] += fin_shift
        start_idx += num_vertices

    return pos_flat

def min_img_dist(s, t, box_dim):
    box_half = box_dim*0.5
    return np.remainder(s - t + box_half, box_dim) - box_half

def get_cluster_iterator(data, edges_list, box_dim, min_part = 0, attibutes=['pos',]):
    assert len(data.timestep) == 1, "Data must contain exactly one timestep for cluster iteration."
    g2 = ig.Graph(n=len(data.particles), edges=edges_list)
    for att in attibutes:
        g2.vs[att] = data.att
    g2.simplify()
    decomposition = g2.decompose(minelements=min_part)
    for subgraph in decomposition:
        flag, pass_graph = check_breakage(
            subgraph, box_dim)
        if not flag:
            positions = unbreak_graph(
                pass_graph, box_dim)
            subgraph.vs['pos'] = positions
        yield subgraph