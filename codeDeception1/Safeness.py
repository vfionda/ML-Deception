import networkx as nx
import copy
import random

class Safeness:
    @staticmethod
    def getBestIntraEdgeAddition(g, targetC):
        targetC = set(targetC)

        if len(targetC) <= 1:
            return None, None

        induced_subgraph = g.subgraph(targetC)
        components = [set(component) for component in nx.connected_components(induced_subgraph)]

        if len(components) <= 1:
            return None, None

        component_of = {}
        for idx, component in enumerate(components):
            for node in component:
                component_of[node] = idx

        target_size = len(targetC)
        denominator = 2 * (target_size - 1)

        candidates = []

        target_nodes = sorted(targetC)

        for i, u in enumerate(target_nodes):
            comp_u = components[component_of[u]]
            size_cu = len(comp_u)

            for w in target_nodes[i + 1:]:
                if g.has_edge(u, w):
                    continue

                if component_of[u] == component_of[w]:
                    continue

                comp_w = components[component_of[w]]
                size_cw = len(comp_w)

                degree_u = g.degree[u]
                degree_w = g.degree[w]

                external_u = sum(1 for neighbor in g.neighbors(u) if neighbor not in targetC)
                external_w = sum(1 for neighbor in g.neighbors(w) if neighbor not in targetC)

                penalty_u = 0.0 if degree_u == 0 else external_u / (2 * degree_u * (degree_u + 1))
                penalty_w = 0.0 if degree_w == 0 else external_w / (2 * degree_w * (degree_w + 1))

                gain = 0.0
                gain += ((size_cu - 1) * size_cw) / denominator
                gain += ((size_cw - 1) * size_cu) / denominator
                gain += (size_cw - 1) / denominator
                gain += (size_cu - 1) / denominator
                gain -= penalty_u
                gain -= penalty_w

                candidates.append((Safeness.norm_edge(u, w), gain))

        if not candidates:
            return None, None

        max_gain = max(gain for _, gain in candidates)
        threshold = 0.9 * max_gain
        eligible = [(edge, gain) for edge, gain in candidates if gain >= threshold]

        if not eligible:
            return None, None

        best_edge, chosen_gain = random.choice(eligible)
        return best_edge, chosen_gain


    ##INTRA-EDGE DELETION may bring a safeness increase
    @staticmethod
    def norm_edge(u, v):
        return tuple(sorted((u, v)))

    @staticmethod
    def getBestIntraEdgeDeletion(g, targetC):
        """Exact safeness gain of deleting a non-bridge intra-community edge.

        For sigma(n) = 1/2 (R(n) - E_in(n))/(|C|-1) + 1/2 E_out(n)/deg(n),
        deleting a non-bridge edge (u,w) inside C leaves R unchanged, drops
        E_in(u) and E_in(w) by one each, and drops deg(u), deg(w) by one each:

            delta = 1/(|C|-1)
                  + 1/2 E_out(u) / (deg(u) (deg(u)-1))
                  + 1/2 E_out(w) / (deg(w) (deg(w)-1))

        with PRE-deletion degrees and external counts.

        Fixed 2026-09-08. The previous version read the degrees after removing
        the edge, used the internal rather than the external neighbour count,
        and omitted the reachability term; it therefore did not compute the
        change in safeness it was selecting on. Original kept alongside as
        Safeness.py.orig-buggy-20260908.
        """
        targetC = set(targetC)
        induced_subgraph = g.subgraph(targetC)
        if induced_subgraph.number_of_edges() == 0:
            return None, None
        bridges = {Safeness.norm_edge(*e) for e in nx.bridges(induced_subgraph)}
        size = len(targetC)
        max_value = -float('inf')
        best_edge = None

        for u, w in induced_subgraph.edges:
            edge = Safeness.norm_edge(u, w)
            if edge in bridges:
                continue

            value = 1.0 / (size - 1) if size > 1 else 0.0
            for n in (u, w):
                deg = g.degree[n]
                if deg > 1:
                    e_out = sum(1 for nb in g.neighbors(n) if nb not in targetC)
                    value += 0.5 * e_out / (deg * (deg - 1))

            if value > max_value:
                max_value = value
                best_edge = edge

        if best_edge is None:
            return None, None
        return best_edge, max_value

    @staticmethod
    def getBestInterEdgeAddition(g, targetC):
        max_value = -float("inf")
        best_nodes = []
        candidate_nodes_by_u = {}

        for u in targetC:
            # find external nodes not already connected
            candidate_nodes = [
                node for node in g.nodes
                if node not in targetC and not g.has_edge(u, node)
            ]

            if not candidate_nodes:
                continue

            degree_u = g.degree[u]
            edges_u_to_targetC = sum(1 for neighbor in g.neighbors(u) if neighbor in targetC)

            if degree_u == 0:
                value = 0.5
            else:
                value = edges_u_to_targetC / (2 * degree_u * (degree_u + 1))

            if value > 0 and value > max_value:
                max_value = value
                best_nodes = [u]
                candidate_nodes_by_u = {u: candidate_nodes}
            elif value > 0 and value == max_value:
                best_nodes.append(u)
                candidate_nodes_by_u[u] = candidate_nodes

        if not best_nodes:
            return None, max_value

        chosen_u = random.choice(best_nodes)
        chosen_v = random.choice(candidate_nodes_by_u[chosen_u])
        return Safeness.norm_edge(chosen_u, chosen_v), max_value

    @staticmethod
    def _run_safdec(g, targetC, budget, budget_percentage=False, allow_intra_add=True):
        targetC = set(targetC)

        # Determine the total budget
        if budget_percentage:
            beta = round(budget * len(targetC))
        else:
            beta = int(budget)

        remaining_budget = beta
        max_intra_first_steps = beta // 2 if allow_intra_add else 0
        intra_first_steps_used = 0

        modified_graph = copy.deepcopy(g)
        edits = []

        while remaining_budget > 0:
            best_added_intra_edge, intra_add_value = (None, None)
            if allow_intra_add:
                best_added_intra_edge, intra_add_value = Safeness.getBestIntraEdgeAddition(modified_graph, targetC)

            best_intra_edge, intra_value = Safeness.getBestIntraEdgeDeletion(modified_graph, targetC)
            best_inter_edge, inter_value = Safeness.getBestInterEdgeAddition(modified_graph, targetC)

            intra_add_value = -float("inf") if intra_add_value is None else intra_add_value
            intra_value = -float("inf") if intra_value is None else intra_value
            inter_value = -float("inf") if inter_value is None else inter_value

            if (
                allow_intra_add
                and intra_first_steps_used < max_intra_first_steps
                and best_added_intra_edge is not None
                and intra_add_value >= 0
            ):
                edge = Safeness.norm_edge(*best_added_intra_edge)
                edit = ("intraA", edge)
                modified_graph.add_edge(*edge)
                edits.append(edit)
                intra_first_steps_used += 1
                remaining_budget -= 1
                continue

            best_value = max(intra_add_value, intra_value, inter_value)

            if best_value == -float("inf"):
                break

            if allow_intra_add and best_value == intra_add_value and best_added_intra_edge is not None:
                edge = Safeness.norm_edge(*best_added_intra_edge)
                edit = ("intraA", edge)
                modified_graph.add_edge(*edge)
                edits.append(edit)
                remaining_budget -= 1
            elif best_value == intra_value and best_intra_edge is not None:
                edge = Safeness.norm_edge(*best_intra_edge)
                edit = ("intraD", edge)
                modified_graph.remove_edge(*edge)
                edits.append(edit)
                remaining_budget -= 1
            elif best_value == inter_value and best_inter_edge is not None:
                edge = Safeness.norm_edge(*best_inter_edge)
                edit = ("interA", edge)
                modified_graph.add_edge(*edge)
                edits.append(edit)
                remaining_budget -= 1
            else:
                break

        return modified_graph, edits

    @staticmethod
    def runSAFDEC(g, targetC, budget, budget_percentage=False):
        return Safeness._run_safdec(g, targetC, budget, budget_percentage=budget_percentage, allow_intra_add=True)

    @staticmethod
    def runSAFDEC_noIntraAdd(g, targetC, budget, budget_percentage=False):
        return Safeness._run_safdec(g, targetC, budget, budget_percentage=budget_percentage, allow_intra_add=False)

    @staticmethod
    def computeSafeness(g, targetC, coms):
        targetC = set(targetC)
        safeness = {}

        # Precompute the subgraph induced by targetC for reachability
        induced_subgraph = g.subgraph(targetC)

        for n in targetC:
            # Compute reachable nodes within targetC
            reachable_nodes = nx.single_source_shortest_path_length(induced_subgraph, n)
            reachable_count = len(reachable_nodes)

            # Compute the number of edges connecting n to nodes in targetC
            edges_in_targetC = sum(1 for neighbor in g.neighbors(n) if neighbor in targetC)

            # Compute the number of edges connecting n to nodes not in targetC
            edges_out_targetC = sum(1 for neighbor in g.neighbors(n) if neighbor not in targetC)

            # Compute degree of n
            degree_n = g.degree[n]

            # Avoid division by zero in case of isolated nodes
            if len(targetC) > 1:
                first_term = (1 / 2) * ((reachable_count - edges_in_targetC) / (len(targetC) - 1))
            else:
                first_term = 0

            if degree_n > 0:
                second_term = (1 / 2) * (edges_out_targetC / degree_n)
            else:
                second_term = 0

            # Add to total safeness
            safeness[n] = first_term + second_term

        return safeness
