from itertools import product
import numpy as np
import igraph as ig
from string import Template
import re
import warnings
from typing import Iterable, Mapping, Tuple, List, Set, Union, Iterator, Sequence, Optional

def determine_key_val_from_filename(template: Union[Template, str], data_path: str, key: str) -> float:
    """
    Extract a single templated value from a filename (or path).

    This converts a `string.Template` (or template string) into a compiled
    regular expression via :func:`template_to_regex`, matches it against
    `data_path`, and returns the value for the requested `key` as ``float``.

    Parameters
    ----------
    template : Template or str
        A filename template containing placeholders compatible with
        ``string.Template`` (e.g., ``"run_${temp}K_seed${seed}.dat"``).
    data_path : str
        The concrete filename or path to parse.
    key : str
        The placeholder name whose value should be returned.

    Returns
    -------
    float
        The numeric value corresponding to ``key`` parsed from ``data_path``.

    Raises
    ------
    LookupError
        If the path does not match the template or the key cannot be found.
    """
    
    regex_pattern = template_to_regex(template)
    match = regex_pattern.fullmatch(data_path)
    if not match:
        raise LookupError("Failed to extract parameters from templated filename (no matching key)")
        
    params = match.groupdict()
    
    return float(params[key])

def get_template_keys(template_str: str) -> Set[str]:
    """
    Extract placeholder names from a ``string.Template`` string.

    Parameters
    ----------
    template_str : str
        The template string to inspect (e.g., ``"x_${a}_y_${b}"``).

    Returns
    -------
    set of str
        A set containing all unique placeholder names found in the template.
    """
    pattern = Template.pattern  # precompiled regex pattern for Template
    keys = set()
    for match in re.finditer(pattern, template_str):
        # Either the 'named' group or the 'braced' group holds the key.
        key = match.group('named') or match.group('braced')
        if key:  # Only add if a key was found
            keys.add(key)
    return keys

def template_to_regex(template_obj: Union[Template, str]) -> re.Pattern:
    """
    Convert a ``string.Template`` (or template string) to a compiled regex.

    The resulting pattern can be used to parse strings produced from the
    template back into their component fields via named capture groups.
    For ambiguous cases where a field may contain literal characters that
    also appear directly after the placeholder in the template, a lookahead
    is used to stop the match at the following literal text. A special case
    is included for fields named ``what_am_I_looking_at`` immediately
    followed by ``'/'`` in the template: this field is matched greedily to
    allow values that may themselves contain slashes (e.g., ``"chains/ligands"``).

    Parameters
    ----------
    template_obj : Template or str
        A ``string.Template`` object or a compatible template string.

    Returns
    -------
    re.Pattern
        A compiled regular expression with one named group per placeholder.
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

def assemble_paths(
    master_dict: Mapping[str, Iterable[str]],
    template_hndl: Template,
    parallel_param_id: Optional[str] = None
) -> Tuple[List[str], Union[List[str], List[List[str]]]]:
    """
    Assemble paths by substituting parameter combinations into a template.

    Only parameters whose names appear in the template are used; any extra
    keys in ``master_dict`` are ignored to avoid producing duplicates.
    Optionally, one parameter can be expanded "in parallel": it is first
    substituted with a placeholder to generate a base list of paths, and then
    expanded per base path with the actual values for that parameter.

    Parameters
    ----------
    master_dict : dict[str, Iterable]
        Mapping from parameter name to iterable of values to substitute.
    template_hndl : Template
        A ``string.Template`` instance used to generate the paths.
    parallel_param_id : str, optional
        Name of a parameter in ``master_dict`` that should be expanded
        separately after the initial Cartesian product.

    Returns
    -------
    tuple[list[str], list[str] or list[list[str]]]
        ``(keys_assembler, paths_to_calc_with)`` where
        ``keys_assembler`` is the list of paths from the Cartesian product
        over the filtered parameters, and ``paths_to_calc_with`` is either
        the same list (if ``parallel_param_id`` is ``None``) or a list of
        lists, each containing the per-base-path expansion of the parallel
        parameter.
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

def determine_box_dim_from_filename(
    template: Union[Template, str],
    data_path: str,
    key_concentration: str,
    key_obj_no: str
) -> np.ndarray:
    """
    Compute cubic box dimensions from values parsed out of a templated path.

    This parses ``data_path`` using ``template`` and extracts a concentration
    (``key_concentration``) and object count (``key_obj_no``). It interprets
    the concentration as number density in mol/m³ and uses Avogadro's number
    to determine the simulation volume and a cubic box length.

    Parameters
    ----------
    template : Template or str
        A filename template containing placeholders.
    data_path : str
        The filename or path matching the template.
    key_concentration : str
        Placeholder name for the concentration value.
    key_obj_no : str
        Placeholder name for the total number of objects.

    Returns
    -------
    numpy.ndarray
        A length-3 array ``[L, L, L]`` of the cubic box dimensions.

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

def check_breakage(graph_el: ig.Graph, box_dim: np.ndarray) -> Tuple[bool, ig.Graph]:
    """
    Check whether a periodic cluster is contiguous under minimum-image criteria.

    Vertices are assumed to carry folded coordinates in ``'pos_folded'``.
    An edge is kept only if the absolute component-wise separation is less than
    ``box_dim/2``. The function then tests if the resulting graph is a single
    connected component.

    .. warning::
       The coordinates passed to this function **must be folded**. The
       logic does not work otherwise.

    Parameters
    ----------
    graph_el : igraph.Graph
        Input graph whose vertices have attribute ``'pos_folded'`` as
        an ``(N, 3)`` array-like.
    box_dim : array_like
        Simulation box dimensions (length-3).

    Returns
    -------
    tuple[bool, igraph.Graph]
        ``(is_connected, filtered_graph)`` where ``filtered_graph`` contains
        only edges satisfying the minimum-image criterion.

    Notes
    -----
    The returned graph is simplified via ``Graph.simplify()`` and decomposed
    via ``Graph.decompose()`` to test connectivity.
    """
    warnings.warn(f"The coordinates passed to check_breakage must be folded!. The logic doesnt work otherwise!")
    positions = np.array(graph_el.vs['pos_folded'])
    edgers_filtered = [(pair_el1, pair_el2) for pair_el1, pair_el2 in graph_el.get_edgelist(
    ) if all(abs(positions[pair_el1]-positions[pair_el2]) < box_dim/2)]

    g_temp = ig.Graph(n=len(graph_el.vs), edges=edgers_filtered,
                      vertex_attrs={'pos_folded': positions})
    g_temp.simplify()
    decomposition_bla = g_temp.decompose()
    return len(decomposition_bla) == 1, g_temp

def unbreak_graph(broken_graph: ig.Graph, box_dim: np.ndarray) -> np.ndarray:
    """
    Stitch clusters split by periodic boundaries back into a single cluster.

    The graph is decomposed into connected components (assuming vertex
    attribute ``'pos_folded'`` exists). Components are sorted by size,
    and each non-reference component is translated by an integer multiple
    of ``box_dim`` (per-axis) so that its center-of-mass aligns with the
    reference component under minimum-image logic.

    .. warning::
       The coordinates passed to this function **must be folded**. The
       logic does not work otherwise!

    Parameters
    ----------
    broken_graph : igraph.Graph
        Graph whose vertices have ``'pos_folded'`` coordinates.
    box_dim : array_like
        Simulation box dimensions (length-3).

    Returns
    -------
    numpy.ndarray
        Flattened array of shape ``(N, 3)`` containing the reassembled
        (unbroken) folded positions for all vertices in the original graph.
    """
    warnings.warn(f"The coordinates passed to unbreak_graph must be folded!. The logic doesnt work otherwise!")

    decomposition = broken_graph.decompose()
    len_and_pos=[(len(el.vs['pos_folded']), 
        el.vs['pos_folded']) for el in decomposition]
    len_and_pos.sort(key=lambda t: t[0], reverse=True)
    pos_flat = np.concatenate([y for _,y in len_and_pos])
    means_and_positions = [(x, np.mean(
        y, axis=0)) for x,y in len_and_pos]
    start_idx = means_and_positions[0][0]
    ref_com=means_and_positions[0][1]
    # Start from index 1 to skip the first cluster
    for i in range(1, len(means_and_positions)):
        num_vertices, mean_pos = means_and_positions[i]
        mean_shift = ref_com - mean_pos
        exceed_limit = np.abs(mean_shift) > box_dim / 2
        fin_shift = np.zeros(3)
        fin_shift[exceed_limit] = np.sign(
            mean_shift[exceed_limit]) * box_dim[exceed_limit]
        pos_flat[start_idx:start_idx + num_vertices] += fin_shift
        start_idx += num_vertices

    return pos_flat

def min_img_dist(s: np.ndarray, t: np.ndarray, box_dim: np.ndarray) -> np.ndarray:
    """
    Compute the minimum-image displacement vector between two points.

    Parameters
    ----------
    s : array_like
        First position (length-3).
    t : array_like
        Second position (length-3).
    box_dim : array_like
        Simulation box dimensions (length-3).

    Returns
    -------
    numpy.ndarray
        Displacement vector ``s - t`` wrapped into ``[-box_dim/2, box_dim/2)``.
    """
    box_half = box_dim*0.5
    return np.remainder(s - t + box_half, box_dim) - box_half

def get_cluster_iterator(
    data: "H5DataSelector",
    edges_list: Sequence[Tuple[int, int]],
    box_dim: np.ndarray,
    min_part: int = 0,
    attibutes: List[str] = ['pos_folded',],
) -> Iterator[ig.Graph]:
    """
    Iterate over connected components (clusters) in a dataset.

    Builds an ``igraph.Graph`` from ``edges_list`` over ``len(data.particles)``
    vertices, attaches requested vertex attributes from ``data``, simplifies the
    graph, and decomposes it into components of at least ``min_part`` vertices.
    For each component, checks for periodic breakage and, if necessary, stitches
    it back together; the resulting positions are stored in
    ``'pos_folded_unbroken'`` on the subgraph before yielding it.

    Parameters
    ----------
    data : object
        An H5DataSelector object with attributes:
        - ``timestep`` : iterable.
        - ``particles`` : iterable.
        - Each name in ``attibutes`` present as an attribute on ``data`` and
          indexable to length ``len(particles)`` (e.g., ``data.pos``).
    edges_list : list[tuple[int, int]]
        Edge list for constructing the graph.
    box_dim : array_like
        Simulation box dimensions (length-3).
    min_part : int, optional
        Minimum component size to yield (default is ``0``).
    attibutes : list[str], optional
        Vertex attribute names to copy from ``data`` onto the graph
        (default is ``['pos']``).

    Yields
    ------
    igraph.Graph
        Subgraphs representing clusters. Each subgraph has
        ``'pos_folded_unbroken'`` set to positions corrected for periodic
        breakage if needed.

    Raises
    ------
    AssertionError
        If ``data.timestep`` does not contain exactly one entry.
    """
    assert len(data.timestep) == 1, "Data must contain exactly one timestep for cluster iteration."
    g2 = ig.Graph(n=len(data.particles), edges=edges_list)
    for att in attibutes:
        try:
            values = getattr(data, att)
        except KeyError:
            if att != 'pos_folded':
                raise
            # Legacy HDF5 files contain only unwrapped ``pos``. Fold a
            # transient array for the periodic-breakage helpers below.
            values = np.mod(np.asarray(data.pos), box_dim)
        g2.vs[att] = values
    g2.simplify()
    decomposition = g2.decompose(minelements=min_part)
    for subgraph in decomposition:
        subgraph.vs['pos_folded_unbroken'] = subgraph.vs['pos_folded']
        flag, pass_graph = check_breakage(
            subgraph, box_dim)
        if not flag:
            positions = unbreak_graph(
                pass_graph, box_dim)
            subgraph.vs['pos_folded_unbroken'] = positions
        yield subgraph
