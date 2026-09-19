import networkx as nx
from Safeness import Safeness
from Modularity import Modularity

class Utils:

    @staticmethod
    def applyDeception(deception_algo, g, com, budget, coms=None):
        if deception_algo == 'SAFDEC':
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


