import networkx as nx
from .Safeness import Safeness
from .Modularity import Modularity
import numpy as np
from sklearn.metrics import adjusted_rand_score
import sklearn





class Utils:

    @staticmethod
    def applyDeception(deception_algo, g, com, budget, coms=None, subgraph=None):
        if deception_algo == 'SAFDEC':
            if subgraph is not None and nx.is_connected(subgraph):
                print("CONNECTED --> APPLYING NO_INTRA")
                return Safeness.runSAFDEC_noIntraAdd(g, com, budget)
            return Safeness.runSAFDEC(g, com, budget)
        elif deception_algo == 'MOD':
            return Modularity.runModularity(g, com, coms, budget)


    @staticmethod
    def getDeceptionScore(coms, target_community, g):
        target_community = set(target_community)

        #intersezione tra target e le comunita della com structure
        member_for_community = [
            sum(node in target_community for node in community)
            for community in coms
        ]

        intersecting = [
            (community, members)
            for community, members in zip(coms, member_for_community)
            if members > 0
        ]

        #if not intersecting:
        #    return 0.0

        # Recall R(C_i, C) = |C_i ∩ C| / |C|
        recall_values = [
            members / len(target_community)
            for _, members in intersecting
        ]

        # Precision P(C_i, C) = |C_i ∩ C| / |C_i|
        precision_values = [
            members / len(community)
            for community, members in intersecting
        ]

        second_part = (
                0.5 * (1 - max(recall_values)) +
                0.5 * (1 - sum(precision_values) / len(precision_values))
        )

        num_components = nx.number_connected_components(
            g.subgraph(target_community)
        )

        if len(target_community) > 1:
            first_part = 1 - ((num_components - 1) / (len(target_community) - 1))
        else:
            first_part = 1.0  # reachability is trivial for a singleton

        return first_part * second_part


    @staticmethod
    def communities_to_label_array(community_list, num_nodes):
        
        # Initialize vectors;
        labels = np.full(num_nodes, -1, dtype=int)
        
        #ALIAS ALREADY PRESENT AS NODE IDS
        
        for cluster_id, nodes in enumerate(community_list):
            for node_id in nodes:
                # Map 1-based node ID to 0-based array index
                idx=int(node_id)
                if 1 <= idx <= num_nodes:
                    labels[idx - 1] = cluster_id
                    
        return labels
    
    @staticmethod
    def adjusted_random_index(original_communities, modified_communities):
        
        # Get the highest node ID present in either set to define the array bounds
        
        num_nodes = max(int(node) for cluster in original_communities for node in cluster)
        
        # Convert both community structures into distribution vectors
        dist_original = Utils.communities_to_label_array(original_communities, num_nodes)
        print(dist_original)
        dist_modified = Utils.communities_to_label_array(modified_communities, num_nodes)
        print(dist_modified)
        # Calculate the Adjusted Rand Index
        return sklearn.metrics.adjusted_rand_score(dist_original, dist_modified)
    

    

    @staticmethod
    def adjusted_random_index_target(original_communities, modified_communities, target_community):
        num_nodes = max(int(node) for cluster in original_communities for node in cluster)

        dist_original = Utils.communities_to_label_array(original_communities, num_nodes)
        dist_modified = Utils.communities_to_label_array(modified_communities, num_nodes)

        target_indices = set(int(node_id) - 1 for node_id in target_community)
        non_target_indices = [i for i in range(num_nodes) if i not in target_indices]

        non_target_dist_original = dist_original[non_target_indices]
        non_target_dist_modified = dist_modified[non_target_indices]

        print(non_target_dist_original)
        print(non_target_dist_modified)

        return sklearn.metrics.adjusted_rand_score(non_target_dist_original, non_target_dist_modified)
    
    
    @staticmethod
    def normalized_mutual_info(original_communities, modified_communities):
     
        # num_nodes calculation exactly as in your ARI method
        num_nodes = max(int(node) for cluster in original_communities for node in cluster)
        
        dist_original = Utils.communities_to_label_array(original_communities, num_nodes)
        dist_modified = Utils.communities_to_label_array(modified_communities, num_nodes)
        
        print("NMI---")
        print(dist_original)
        print(dist_modified)
        
        # calculate the onrmalized mutual info score
        return sklearn.metrics.normalized_mutual_info_score(dist_original, dist_modified)

    @staticmethod
    def normalized_mutual_info_target(original_communities, modified_communities, target_community):
        num_nodes = max(int(node) for cluster in original_communities for node in cluster)

        dist_original = Utils.communities_to_label_array(original_communities, num_nodes)
        dist_modified = Utils.communities_to_label_array(modified_communities, num_nodes)

        target_indices = set(int(node_id) - 1 for node_id in target_community)
        non_target_indices = [i for i in range(num_nodes) if i not in target_indices]

        non_target_dist_original = dist_original[non_target_indices]
        non_target_dist_modified = dist_modified[non_target_indices]

        print("NMI---")
        print(non_target_dist_original)
        print(non_target_dist_modified)

        return sklearn.metrics.normalized_mutual_info_score(non_target_dist_original, non_target_dist_modified)
