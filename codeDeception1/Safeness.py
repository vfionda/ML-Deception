import networkx as nx
import random
import copy

class Safeness:

    ###INTRA-EDGE addition always brings a safeness decrease thus it is not considered

    ##INTRA-EDGE DELETION may bring a safeness increase
    @staticmethod
    def getBestIntraEdgeDeletion(g,targetC):
        induced_subgraph = g.subgraph(targetC)
        bridges = set(nx.bridges(induced_subgraph))
        max_value = -float('inf')
        best_edge = None

        for edge in induced_subgraph.edges:
            if edge not in bridges and (edge[1], edge[0]) not in bridges:  # Account for undirected edge ordering
                u, w = edge

                # Temporarily remove the edge
                g.remove_edge(u, w)

                # Compute the number of edges from u and w to nodes in targetC
                edges_u_to_targetC = sum(1 for neighbor in g.neighbors(u) if neighbor in targetC)
                edges_w_to_targetC = sum(1 for neighbor in g.neighbors(w) if neighbor in targetC)

                # Compute the degrees of u and w
                degree_u = g.degree[u]
                degree_w = g.degree[w]

                # Ensure no division by zero
                value = 0
                if degree_u > 1:
                    value += edges_u_to_targetC / (2 * degree_u * (degree_u - 1))
                if degree_w > 1:
                    value += edges_w_to_targetC / (2 * degree_w * (degree_w - 1))

                # Restore the edge
                g.add_edge(u, w)

                # Update the maximum value and the corresponding edge
                if value > max_value and value>0:
                    max_value = value
                    best_edge = edge

        return best_edge,max_value

    ###INTER-EDGE addition always brings a safeness increase
    @staticmethod
    def getBestInterEdgeAddition(g, targetC):
        max_value = -float('inf')
        best_node = None

        # Step 2: Iterate over nodes in targetC
        for u in targetC:
            degree_u = g.degree[u]

            # Skip if degree is zero to avoid division by zero
            if degree_u == 0:
                continue

            # Step 3: Count the number of edges from u to nodes in targetC
            edges_u_to_targetC = sum(1 for neighbor in g.neighbors(u) if neighbor in targetC)

            # Step 4: Compute the ratio
            value = edges_u_to_targetC / degree_u

            # Step 5: Update the maximum value and corresponding node
            if value > max_value and value>0:
                max_value = value
                best_node = u

        return best_node,max_value

    @staticmethod
    def runSAFDEC(g, targetC, budget, budget_percentage=False):
        # Determine the total budget
        if budget_percentage:
            beta = round(budget * len(targetC))
        else:
            beta = int(budget)

        # Track the remaining budget
        remaining_budget = beta

        modified_graph = copy.deepcopy(g)

        edits=[]

        while remaining_budget > 0:
            # Get the best intra-edge deletion
            best_intra_edge, intra_value = Safeness.getBestIntraEdgeDeletion(modified_graph, targetC)

            # Get the best inter-edge addition
            best_inter_node, inter_value = Safeness.getBestInterEdgeAddition(modified_graph, targetC)

            # Determine which operation to apply
            if intra_value > inter_value and best_intra_edge is not None:
                # Apply the best intra-edge deletion
                modified_graph.remove_edge(*best_intra_edge)
                edits.append(("intraD",best_intra_edge))
                remaining_budget -= 1
                #print(f"SAF Deleted intra-edge {best_intra_edge} with value {intra_value}")
            elif inter_value >= intra_value and best_inter_node is not None:
                # Apply the best inter-edge addition (find a suitable node to connect)
                nodes_outside_targetC = [node for node in modified_graph.nodes if node not in targetC]
                if nodes_outside_targetC:  # Ensure there are nodes outside targetC
                    random_node = random.choice(nodes_outside_targetC)
                    modified_graph.add_edge(best_inter_node, random_node)
                    edits.append(("interA", (best_inter_node, random_node)))
                    remaining_budget -= 1
                    #print(f"SAF Added inter-edge from {best_inter_node} to {random_node} with value {inter_value}")
            else:
                # No beneficial operation left
                #print("No further beneficial operations possible.")
                break
        return (modified_graph,edits)

    def computeSafeness(g,targetC,coms):
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
