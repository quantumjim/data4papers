

# modified version of `get_error_coords` that doesn't mess up the numbers for the qubit used for logical readout
def get_error_coords(
    self,
    counts,
    decoding_graph,
    method="spitz",
    remove_invalid_edges=False,
    return_samples=False,
):
    """
    Uses the `get_error_probs` method of the given decoding graph to generate probabilities
    of single error events from given counts. The location and time of each error is
    also calculated.

    Args:
        counts (dict): Counts dictionary of the results to be analyzed.
        decoding_graph (DecodingGraph): Decoding graph object constructed
        from this code.
        method (string): Method to used for calculation. Supported
        methods are 'spitz' (default) and 'naive'.
        remove_invalid_edges (string): Whether to delete edges from the graph if
        they are found to be invalid.
        return_samples (bool): Whether to also return the number of
        samples used to calculated each probability.
    Returns:
        dict: Keys are the coordinates (qubit, start_time, end_time) for specific error
        events. Time refers to measurement rounds. Values are a dictionary whose keys are
        the edges that detected the event, and whose keys are the calculated probabilities.
    Additional information:
        Time calculation does not take into account get lengths. It assumes that the
        subrounds within the schedule and the measurement all take the same time. Time
        is in units of rounds.
    """

    # though the documented use case requires a decoding graph and a counts dict, there is also an
    # undocumented internal use case, where just the bare graph is provided and no counts. This is
    # to find and delete invalid edges
    if isinstance(decoding_graph, rx.PyGraph):
        graph = decoding_graph
    else:
        graph = decoding_graph.graph
    nodes = graph.nodes()
    if counts:
        if return_samples:
            error_probs, samples = decoding_graph.get_error_probs(
                counts, method=method, return_samples=True
            )
        else:
            error_probs = decoding_graph.get_error_probs(counts, method=method)
    else:
        error_probs = {}
        for n0, n1 in graph.edge_list():
            if nodes[n0].is_boundary:
                edge = (n1, n1)
            elif nodes[n1].is_boundary:
                edge = (n0, n0)
            else:
                edge = (n0, n1)
            error_probs[edge] = np.nan

    if hasattr(self, "z_logicals"):
        z_logicals = set(self.z_logicals)
    elif hasattr(self, "z_logical"):
        z_logicals = {self.z_logical}
    else:
        print("No qubits for z logicals found. Proceeding without.")
        z_logicals = set()

    round_length = len(self.schedule) + 1

    error_coords = {}
    sample_coords = {}
    for (n0, n1), prob in error_probs.items():
        node0 = nodes[n0]
        node1 = nodes[n1]
        if n0 != n1:
            qubits = graph.get_edge_data(n0, n1).qubits
            if qubits:
                # error on a code qubit between rounds, or during a round
                assert (node0.time == node1.time and node0.qubits != node1.qubits) or (
                    node0.time != node1.time and node0.qubits != node1.qubits
                )
                qubit = qubits[0]
                # error between rounds
                if node0.time == node1.time:
                    dts = []
                    for node in [node0, node1]:
                        pair = [qubit, node.properties["link qubit"]]
                        for dt, pairs in enumerate(self.schedule):
                            if pair in pairs or tuple(pair) in pairs:
                                dts.append(dt)
                    time = [max(0, node0.time - 1 + (max(dts) + 1) / round_length)]
                    time.append(min(self.T, node0.time + min(dts) / round_length))
                # error during a round
                else:
                    # put nodes in descending time order
                    if node0.time < node1.time:
                        node_pair = [node1, node0]
                    else:
                        node_pair = [node0, node1]
                    # see when in the schedule each node measures the qubit
                    dts = []
                    for node in node_pair:
                        pair = [qubit, node.properties["link qubit"]]
                        for dt, pairs in enumerate(self.schedule):
                            if pair in pairs or tuple(pair) in pairs:
                                dts.append(dt)
                    # use to define fractional time
                    if dts[0] < dts[1]:
                        time = [node_pair[1].time + (dts[0] + 1) / round_length]
                        time.append(node_pair[1].time + dts[1] / round_length)
                    else:
                        # impossible cases get no valid time
                        time = []
                        if remove_invalid_edges:
                            graph.remove_edge(n0, n1)
            else:
                # measurement error
                assert node0.time != node1.time and node0.qubits == node1.qubits
                qubit = node0.properties["link qubit"]
                t0 = min(node0.time, node1.time)
                if abs(node0.time - node1.time) == 1:
                    if self.resets:
                        time = [t0, t0 + 1]
                    else:
                        time = [t0, t0 + (round_length - 1) / round_length]
                else:
                    time = [t0 + (round_length - 1) / round_length, t0 + 1]
        else:
            pass
            # detected only by one stabilizer
            boundary_qubits = list(set(node0.qubits).intersection(z_logicals))
            # for the case of boundary stabilizers
            if False:#boundary_qubits:
                qubit = boundary_qubits[0]
                pair = [qubit, node0.properties["link qubit"]]
                for dt, pairs in enumerate(self.schedule):
                    if pair in pairs or tuple(pair) in pairs:
                        time = [max(0, node0.time - 1 + (dt + 1) / round_length)]
                        time.append(min(self.T, node0.time + dt / round_length))

            else:
                qubit = tuple(node0.qubits + [node0.properties["link qubit"]])
                time = [node0.time, node0.time + (round_length - 1) / round_length]

        if time != []:  # only record if not nan
            if (qubit, time[0], time[1]) not in error_coords:
                error_coords[qubit, time[0], time[1]] = {}
                sample_coords[qubit, time[0], time[1]] = {}
            error_coords[qubit, time[0], time[1]][n0, n1] = prob
            if return_samples:
                sample_coords[qubit, time[0], time[1]][n0, n1] = samples[n0, n1]

    if return_samples:
        return error_coords, sample_coords
    else:
        return error_coords
    
def overlap(link1,link2):
    '''Determine whether two links overlap'''
    return set(link1).intersection(link2) != set()

# figures out links and schedule for a given heavy hex device
def schedule_heavy_hex(backend, blacklist=[]):

    try:
        raw_coupling_map = backend.configuration().coupling_map
        n = backend.configuration().num_qubits
        faulty_qubits = backend.properties().faulty_qubits()
        faulty_gates = backend.properties().faulty_gates()
    except:
        raw_coupling_map = backend.coupling_map
        n = backend.num_qubits
        faulty_qubits = []
        faulty_gates = []


    # remove any double counted pairs in the coupling map
    coupling_map = []
    for pair in raw_coupling_map:
        pair = list(pair)
        pair.sort()
        if pair not in coupling_map:
            coupling_map.append(pair)

    # find the degree for each qubit
    degree = [0]*n
    for pair in coupling_map:
        for j in range(2):
            degree[pair[j]] += 1
    degree = [int(deg) for deg in degree]
    
    # bicolor the qubits
    color = [None]*n
    color[0] = 0
    k = 0
    while None in color:
        for pair in coupling_map:
            for j in range(2):
                if color[pair[j]] is not None:
                    color[pair[(j+1)%2]] = (color[pair[j]]+1)%2
    # determine the color of vertex qubits
    for q in range(n):
        if degree[q]==3:
            vertex_color = color[q]
            break
    # find  vertex qubits for each auxilliary
    link_dict = {}
    for q in range(n):
        if color[q]!=vertex_color:
            link_dict[q] = []
    link_list = list(link_dict.keys())
    for pair in coupling_map:
        for j in range(2):
            if pair[j] in link_list:
                q = pair[(j+1)%2]
                if q not in link_dict[pair[j]]:
                    link_dict[pair[j]].append(q)
    # create the links list
    links = []
    for a, v0v1 in link_dict.items():
        if len(v0v1)==2:
            links.append((v0v1[0],a,v0v1[1]))

    # find the plaquettes
    plaquettes = []
    all_paths = {}
    links_in_plaquettes = set({})
    for link in links:
        paths = [[[link]]]
        for l in range(6):
            paths.append([])
            for path in paths[l]:
                last_link = path[-1]
                for next_link in links:
                    if next_link!=last_link:
                        if overlap(next_link,last_link):
                            try:
                                turn_back = overlap(next_link,path[-2])
                            except:
                                turn_back = False
                            if not turn_back:
                                if (next_link not in path) or l==5:
                                    paths[-1].append(path.copy() + [next_link])
        for path in paths[6]:
            if path[0]==path[-1]:
                plaquette = set(path[:6])
                if plaquette not in plaquettes:
                    plaquettes.append(plaquette)
        all_paths[link] = paths

    # find the plaquettes neighbouring each link
    wings = {link:[] for link in links}
    for p, plaquette in enumerate(plaquettes):
        for link in plaquette:
            wings[link].append(p)

    # now assign a type (x, y or z) to each link so that none overlap
    link_type = {link:None for link in links}
    for unwinged in [False, True]:
        for r in ['x','y','z']:
            # assign a single unassigned link as the current type
            for link in link_type:
                if link_type[link] is None and len(wings[link])==2:
                    link_type[link] = r
                    break

            # assign links that are 3 away in the plaquette or 2 away in different plaquettes as the same type
            all_done = False
            k = 0
            while all_done == False:
                newly_assigned = 0
                for l in [2,3]:
                    for first in all_paths:
                        for path in all_paths[first][l]:
                            last = path[-1]
                            share_plaquette = False
                            for plaquette in plaquettes:
                                if first in plaquette and last in plaquette:
                                    share_plaquette = True
                            bulk = len(wings[first])==2 and len(wings[last])==2
                            if share_plaquette == (l==3):
                                if l==3 or bulk:
                                    link_pair = [first,last]
                                    for j in range(2):
                                        if link_type[link_pair[j]] is not None and link_type[link_pair[(j+1)%2]] is None:
                                            link_type[link_pair[(j+1)%2]] = link_type[link_pair[j]]
                                            newly_assigned += 1
                all_done = newly_assigned==0

        # if plaquettes have a single type missing, fill them in
        for plaquette in plaquettes:
            types = [link_type[link] for link in plaquette]
            for (r1,r2,r3) in [('x','y','z'), ('x','z','y'), ('z','y','x')]:
                if r1 in types and r2 in types and r3 not in types:
                    for link in plaquette:
                        if link_type[link] is None:
                            link_type[link] = r3

    # restrict `links` to only links with a type
    links = [link for link, r in link_type.items() if r]

    # bicolour the vertices
    vcolor = {links[0][0]:0}
    for link in links:
        for j,k in [(0,-1), (-1,0)]:
            if link[j] in vcolor and link[k] not in vcolor:
                vcolor[link[k]] = (vcolor[link[j]]+1)%2

    # find links around each vertex
    triplets = {v:{} for v in vcolor}
    for link, r in link_type.items():
        if link in links:
            for j in [0,-1]:
                if link[j] in triplets:
                    assert r not in triplets[link[j]]
                    triplets[link[j]][r] = link

    # schedule the entangling gates
    rounds = ['xz', 'yx', 'zy']
    schedule = []
    for rr in rounds:
        round_schedule = []
        for v in triplets:
            r = rr[vcolor[v]]
            if r in triplets[v]:
                link = triplets[v][r]
                round_schedule.append((v,link[1]))
        schedule.append(round_schedule)

    # determine which pairs are blacklisted
    blacklist = set(blacklist + faulty_qubits)
    blacklisted_pairs = [set(g.qubits) for g in faulty_gates if len(g.qubits) > 1]
    for pair in raw_coupling_map:
        pair = set(pair)
        if pair not in blacklisted_pairs:
            if pair.intersection(blacklist):
                blacklisted_pairs.append(pair)

    # remove links with a blacklisted pair,
    # and blacklist the other pair in the link
    working_links = []
    for link in links:
        if set(link[0:2]) not in blacklisted_pairs and set(link[1:3]) not in blacklisted_pairs:
            working_links.append(link)
        else:
            if set(link[0:2]) in blacklisted_pairs:
                blacklisted_pairs.append(set(link[1:3]))
            if set(link[1:3]) in blacklisted_pairs:
                blacklisted_pairs.append(set(link[0:2]))
    links = working_links

    # remove corresponding gates from the schedule
    working_schedule = []
    for layer in schedule:
        working_layer = []
        for pair in layer:
            if set(pair) not in blacklisted_pairs:
                working_layer.append(pair)
        working_schedule.append(working_layer)
    schedule = working_schedule

    # check that it all worked
    num_cnots = 0
    cxs = []
    for round_schedule in schedule:
        num_cnots += len(round_schedule)
        cxs += round_schedule
        round_list = []
        for pair in round_schedule:
            round_list += list(pair)
        assert len(round_list)==len(set(round_list)), (len(round_list), len(set(round_list)))
    assert num_cnots == len(cxs)
    for link in links:
        for pair in [tuple(link[0:2]), tuple(link[1:3])]:
            if pair not in cxs and pair[::-1] not in cxs:
                print(link)
    assert num_cnots == 2*len(links), (num_cnots, 2*len(links))

    return links, schedule, triplets