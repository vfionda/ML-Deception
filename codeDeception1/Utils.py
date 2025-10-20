import networkx as nx
from .Safeness import Safeness
from .Modularity import Modularity

class Utils:

    @staticmethod
    def applyDeception(deception_algo, g, com, budget, coms=None):
        if deception_algo == 'SAFDEC':
            return Safeness.runSAFDEC(g, com, budget)
        elif deception_algo == 'MOD':
            return Modularity.runModularity(g, com, coms, budget)


    @staticmethod
    def getDeceptionScore(coms, target_community, g):
        number_communities = len(coms)
        # number of the targetCommunity members in the various communities
        member_for_community = [
            sum(node in target_community for node in community)
            for community in coms
        ]

        # ratio of the targetCommunity members in the various communities
        ratio_community_members = [
            members_for_c / len(com) for members_for_c, com in zip(member_for_community, coms)
        ]

        ##In how many commmunities are the members of the target spread?
        spread_members = sum([1 if mc > 0 else 0 for mc in member_for_community])

        second_part = 1 / 2 * ((spread_members - 1) / number_communities) + 1 / 2 * (
                    1 - sum(ratio_community_members) / spread_members)
        #####

        num_components = nx.number_connected_components(
            g.subgraph(target_community))  # induced subraph only on target community nodes

        if len(target_community) > 1:
            first_part = 1 - ((num_components - 1) / (len(target_community) - 1))
        else:
            first_part = 0  # Default to 0 if target_community has fewer than 2 nodes

        dec_score = first_part * second_part
        return dec_score


