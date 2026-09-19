import networkx as nx
import copy

class Modularity:

###INTRA-EDGE addition

    @staticmethod
    def getBestIntraEdgeAddition(g, coms, targetC):
        # Convert communities to partition format required by `modularity`
        #partition = {node: idx for idx, community in enumerate(coms) for node in community}

        # Compute modularity before any modification
        modularity_before = nx.community.modularity(g, coms)

        max_modularity_diff = -float('inf')
        best_com = None
        best_node_pair = None

        # Iterate over communities
        for i, comi in enumerate(coms):
            # Check if comi contains any node in targetC
            if not any(node in targetC for node in comi):
                continue

            found_pair = False
            # Find node pairs within comi
            for n in comi:
                if n not in targetC:
                    continue  # Node n must be in targetC

                for n_prime in comi:
                    if n_prime != n and not g.has_edge(n, n_prime):
                        # Temporarily add the edge
                        g.add_edge(n, n_prime)

                        # Compute modularity after the modification
                        modularity_after = nx.community.modularity(g, coms)

                        # Calculate modularity difference
                        modularity_diff = modularity_before - modularity_after

                        # Restore the graph by removing the edge
                        g.remove_edge(n, n_prime)

                        # Update the best pair if modularity difference is maximal
                        if modularity_diff > max_modularity_diff and modularity_diff>0:
                            max_modularity_diff = modularity_diff
                            best_com = i
                            best_node_pair = (n, n_prime)

                        found_pair = True
                        break
        return best_com, best_node_pair, max_modularity_diff

###INTER-EDGE addition


    @staticmethod
    def getBestInterEdgeAddition(g, coms, targetC):
        community_degrees = {
            i: sum(g.degree[node] for node in com) for i, com in enumerate(coms)
        }

        max_degree_sum = -float('inf')
        best_pair = None
        best_node_pair = None
        nodePair = None

        # Convert communities to partition format required by `modularity`
        #partition = {node: idx for idx, community in enumerate(coms) for node in community}

        # Compute modularity before adding any edge
        modularity_before = nx.community.modularity(g, coms)

        # Iterate through all pairs of communities
        for i, com1 in enumerate(coms):
            for j, com2 in enumerate(coms):
                # Check for nodes satisfying the conditions
                found_pair = False
                for n in com1:
                    if n in targetC:  # Node in com1 must be in targetC
                        for n_prime in com2:
                            if n_prime not in targetC and not g.has_edge(n, n_prime):
                                # Node in com2 must not be in targetC, and (n, n') must not exist
                                found_pair = True
                                nodePair = (n,n_prime)
                                break
                    if found_pair:
                        break

                # If valid pair found, compute the sum of degrees
                if found_pair:
                    degree_sum = community_degrees[i] + community_degrees[j]
                    if degree_sum > max_degree_sum:
                        max_degree_sum = degree_sum
                        best_pair = (i, j)
                        best_node_pair = nodePair

        if best_node_pair:
            # Temporarily add the edge
            g.add_edge(*best_node_pair)
            # Compute modularity after adding the edge
            modularity_after = nx.community.modularity(g, coms)
            # Remove the edge to restore the original graph
            g.remove_edge(*best_node_pair)

            # Modularity loss
            mod_loss = modularity_before - modularity_after
        else:
            mod_loss = None  # No valid edge found

        return best_node_pair, mod_loss


###INTER-EDGE deletion

    @staticmethod
    def getBestInterEdgeDeletion(g, coms, targetC):
        # Convert communities to partition format required by `modularity`
        partition = {node: idx for idx, community in enumerate(coms) for node in community}

        # Compute modularity before any modification
        modularity_before = nx.community.modularity(g, coms)

        max_modularity_loss = -float('inf')
        best_com_pair = None
        best_edge = None

        # Iterate over communities
        for i, comi in enumerate(coms):
            # Check if comi contains any node in targetC
            target_nodes_in_comi = [n for n in comi if n in targetC]
            if not target_nodes_in_comi:
                continue

            # Iterate over other communities
            for j, comj in enumerate(coms):
                if i == j:
                    continue  # Skip the same community

                found_edge = False
                # Iterate over edges between nodes in comi (from targetC) and comj
                for n in target_nodes_in_comi:
                    for n_prime in comj:
                        if g.has_edge(n, n_prime):  # Ensure the edge exists
                            # Temporarily remove the edge
                            g.remove_edge(n, n_prime)

                            # Compute modularity after the deletion
                            modularity_after = nx.community.modularity(g, coms)

                            # Calculate modularity loss
                            modularity_loss = modularity_before - modularity_after

                            # Restore the edge
                            g.add_edge(n, n_prime)

                            # Update the best pair and edge if modularity loss is maximal
                            if modularity_loss > max_modularity_loss and modularity_loss>0:
                                max_modularity_loss = modularity_loss
                                best_com_pair = (i, j)
                                best_edge = (n, n_prime)

                            found_edge = True
                            break  # Stop searching for edges in comj

                    if found_edge:
                        break  # Stop searching for edges in comi

        return best_com_pair, best_edge, max_modularity_loss

###INTRA-EDGE deletion

    @staticmethod
    def getBestIntraEdgeDeletion(g, coms, targetC):
        # Convert communities to partition format required by `modularity`
        #partition = {node: idx for idx, community in enumerate(coms) for node in community}

        # Compute modularity before any modification
        modularity_before = nx.community.modularity(g, coms)

        # Find the community with the lowest degree that contains a node in targetC
        min_degree = float('inf')
        selected_com = None
        deletable_edge = None

        for i, comi in enumerate(coms):
            # Check if the community contains at least one node in targetC
            target_nodes_in_comi = [n for n in comi if n in targetC]
            if not target_nodes_in_comi:
                continue

            # Find a deletable edge within the community
            for n in target_nodes_in_comi:
                for n_prime in comi:
                    if n_prime != n and g.has_edge(n, n_prime):
                        # Compute the total degree of the community
                        community_degree = sum(g.degree[node] for node in comi)

                        # Update the selected community and edge if it has the lowest degree
                        if community_degree < min_degree:
                            min_degree = community_degree
                            selected_com = (i, comi)
                            deletable_edge = (n, n_prime)
                        break
                if deletable_edge:
                    break

        if not selected_com or not deletable_edge:
            return None, None, None  # No valid community or edge found

        # Compute modularity loss for the deletable edge
        n, n_prime = deletable_edge
        g.remove_edge(n, n_prime)  # Temporarily remove the edge
        modularity_after = nx.community.modularity(g, coms)
        g.add_edge(n, n_prime)  # Restore the edge

        modularity_loss = modularity_before - modularity_after

        return selected_com[0], deletable_edge, modularity_loss

    @staticmethod
    def runModularity(g, targetC, coms, budget, budget_percentage=False):
        if budget_percentage:
            beta = round(budget * len(targetC))
        else:
            beta = int(budget)

        remaining_budget = beta
        modified_graph = copy.deepcopy(g)

        edits = []
        applied_edits = set()  # (op, normalized_edge)
        used_edges = set()  # normalized_edge only, if you want to forbid any reuse

        while remaining_budget > 0:
            selected_com, best_intra_edge, intraD_value = Modularity.getBestIntraEdgeDeletion(modified_graph, coms, targetC)
            bestD_com_pair, bestD_inter_edge, interD_value = Modularity.getBestInterEdgeDeletion(modified_graph, coms,
                                                                                             targetC)
            bestA_node_pair, interA_value = Modularity.getBestInterEdgeAddition(modified_graph, coms, targetC)
            bestIA_com, bestIA_node_pair, intraA_value = Modularity.getBestIntraEdgeAddition(modified_graph, coms, targetC)

            candidate_values = [v for v in [intraD_value, interD_value, interA_value, intraA_value] if v is not None]
            if not candidate_values:
                break

            maximum_value = max(candidate_values)

            applied = False

            if maximum_value == intraD_value and best_intra_edge is not None:
                edge = tuple(sorted(best_intra_edge))
                if edge not in used_edges and modified_graph.has_edge(*edge):
                    modified_graph.remove_edge(*edge)
                    edits.append(("intraD", edge))
                    applied_edits.add(("intraD", edge))
                    used_edges.add(edge)
                    remaining_budget -= 1
                    applied = True

            elif maximum_value == interD_value and bestD_inter_edge is not None:
                edge = tuple(sorted(bestD_inter_edge))
                if edge not in used_edges and modified_graph.has_edge(*edge):
                    modified_graph.remove_edge(*edge)
                    edits.append(("interD", edge))
                    applied_edits.add(("interD", edge))
                    used_edges.add(edge)
                    remaining_budget -= 1
                    applied = True

            elif maximum_value == interA_value and bestA_node_pair is not None:
                edge = tuple(sorted(bestA_node_pair))
                if edge not in used_edges and not modified_graph.has_edge(*edge):
                    modified_graph.add_edge(*edge)
                    edits.append(("interA", edge))
                    applied_edits.add(("interA", edge))
                    used_edges.add(edge)
                    remaining_budget -= 1
                    applied = True

            elif maximum_value == intraA_value and bestIA_node_pair is not None:
                edge = tuple(sorted(bestIA_node_pair))
                if edge not in used_edges and not modified_graph.has_edge(*edge):
                    modified_graph.add_edge(*edge)
                    edits.append(("intraA", edge))
                    applied_edits.add(("intraA", edge))
                    used_edges.add(edge)
                    remaining_budget -= 1
                    applied = True

            if not applied:
                break

        return (modified_graph, edits)