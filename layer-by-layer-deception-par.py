import os
import subprocess
import sys
import time
from pathlib import Path
import glob
from multiprocessing import Pool

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from codeDeception1.Safeness import Safeness
from codeDeception1.Modularity import Modularity
from codeDeception1.Utils import Utils

from MultiLayerGraph import MultiLayerGraph
from GraphUtils import GraphUtils

from cdlib import NodeClustering
import pandas as pd
from collections import defaultdict
import networkx as nx
import random
import copy


def communities_to_label_array(community_list, num_nodes):
    labels = np.full(num_nodes, -1, dtype=int)
    for cluster_id, nodes in enumerate(community_list):
        for node_id in nodes:
            idx = int(node_id)
            if 1 <= idx <= num_nodes:
                labels[idx - 1] = cluster_id
    return labels


def load_communities_cdlib(path_to_csv, graph):
    df = pd.read_csv(path_to_csv, header=None, names=["node", "layer", "community"])
    grouped = defaultdict(set)
    for _, row in df.iterrows():
        grouped[row["community"]].add(row["node"])
    communities = [list(nodes) for nodes in grouped.values()]
    return NodeClustering(communities=communities, graph=graph, method_name="custom_csv_import")


def runRandomDeception(g, targetC, coms, budget):
    beta = int(budget)
    modified_graph = copy.deepcopy(g)
    edits = []
    remaining_budget = beta
    targetC = set(targetC)
    all_nodes = list(modified_graph.nodes())
    node_to_community = {node: idx for idx, community in enumerate(coms) for node in community}
    communities = [set(c) for c in coms]

    target_com_idx = node_to_community[next(iter(targetC))]
    intra_community = communities[target_com_idx]
    non_targets = list(set(all_nodes) - targetC)

    intra_del_candidates = {
        (u, v)
        for u in intra_community
        for v in intra_community
        if u < v and modified_graph.has_edge(u, v)
    }

    inter_del_candidates = {
        (min(u, v), max(u, v))
        for u in targetC
        for v in modified_graph.neighbors(u)
        if node_to_community[v] != target_com_idx
    }

    intra_add_candidates = {
        (u, v)
        for u in targetC
        for v in targetC
        if u < v and not modified_graph.has_edge(u, v)
    }

    inter_add_candidates = {
        (min(u, v), max(u, v))
        for u in targetC
        for v in non_targets
        if node_to_community[v] != target_com_idx
        and not modified_graph.has_edge(u, v)
    }

    while remaining_budget > 0:
        feasible_ops = []
        if intra_add_candidates: feasible_ops.append("intra_add")
        if intra_del_candidates: feasible_ops.append("intra_del")
        if inter_add_candidates: feasible_ops.append("inter_add")
        if inter_del_candidates: feasible_ops.append("inter_del")

        if not feasible_ops:
            break

        op_type = random.choice(feasible_ops)

        if op_type == "intra_add":
            edge = random.choice(list(intra_add_candidates))
            u, v = edge
            modified_graph.add_edge(u, v)
            edits.append(("intraA", (u, v)))
            intra_add_candidates.discard(edge)
            remaining_budget -= 1

        elif op_type == "intra_del":
            edge = random.choice(list(intra_del_candidates))
            u, v = edge
            modified_graph.remove_edge(u, v)
            edits.append(("intraD", (u, v)))
            intra_del_candidates.discard(edge)
            remaining_budget -= 1

        elif op_type == "inter_add":
            edge = random.choice(list(inter_add_candidates))
            u, v = edge
            modified_graph.add_edge(u, v)
            edits.append(("interA", (u, v)))
            inter_add_candidates.discard(edge)
            remaining_budget -= 1

        elif op_type == "inter_del":
            edge = random.choice(list(inter_del_candidates))
            u, v = edge
            modified_graph.remove_edge(u, v)
            edits.append(("interD", (u, v)))
            inter_del_candidates.discard(edge)
            remaining_budget -= 1

    return modified_graph, edits


# ---------------------------------------------------------------------------
# SCML layerwise parallel worker
# ---------------------------------------------------------------------------

def _scml_layerwise_run_worker(args):
    (
        n, idx, target_community,
        data_path, original_communities_sets,
        modified_graphs_dir, dataset_name,
        budget, alias_to_node,
        modularity_before, safeness_before, deception_before,
        comm_algo,
    ) = args

    import copy, random, time, os, subprocess, glob
    import numpy as np
    import networkx as nx
    import pandas as pd
    from collections import defaultdict
    from cdlib import NodeClustering

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from codeDeception1.Safeness import Safeness
    from codeDeception1.Utils import Utils
    from MultiLayerGraph import MultiLayerGraph

    run_suffix_random   = f"random_c{idx}_r{n}"
    run_suffix_modified = f"modified_c{idx}_r{n}"
    run_suffix_saf      = f"saf_c{idx}_r{n}"

    communities_dir = os.path.join(os.getcwd(), "Communities", dataset_name)

    def find_community_file(suffix):
        pattern = os.path.join(communities_dir, f"comm-{comm_algo}-{dataset_name}-*-{suffix}.csv")
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No community file found with suffix '{suffix}'")
        return matches[0]

    def run_scml_cd(data_path, suffix):
        command = [
            "python3", "../discover-communities-scml.py",
            data_path,
            "--repeats", "1",
            "--suffix", suffix
        ]
        subprocess.run(command, check=True)

    per_layer_budget = budget

    print(f"\n=== RUN {n} ===")

   
    

    # RANDOM SECTION --------------
    graph = MultiLayerGraph.from_3d_matrix_file(data_path)
    
    flat_graph_original = graph.get_flatten()
    all_edits_random = {}

    start_time_random = time.time()
    for layer in graph.get_layers():
        print(f" Applying RANDOM deception on layer: {layer}")
        layer_graph_copy = graph.get_layer_graph(layer).copy()
        coms_filtered = [c for c in original_communities_sets]

        modified_random_layer, random_edits = runRandomDeception(layer_graph_copy, target_community, coms_filtered, per_layer_budget)
        graph.multilayer_graph[layer] = modified_random_layer
        all_edits_random[layer] = random_edits

    end_time_random = time.time()
    time_random = end_time_random - start_time_random

    random_graph_path = os.path.join(modified_graphs_dir, f"{dataset_name}-scml_layerwise-random_c{idx}_r{n}.m")
    graph.to_m_file(random_graph_path)

    run_scml_cd(random_graph_path, run_suffix_random)
    random_com_path = find_community_file(run_suffix_random)

    random_flat_graph = graph.get_flatten()

    df_r = pd.read_csv(random_com_path, header=None, names=["alias", "community"])
    grouped_r = defaultdict(set)
    for _, row in df_r.iterrows():
        node_id = alias_to_node.get(row["alias"])
        if node_id:
            grouped_r[row["community"]].add(node_id)
    random_communities_list = [list(nodes) for nodes in grouped_r.values()]

    random_clustering = NodeClustering(communities=random_communities_list, graph=random_flat_graph, method_name="scml")
    random_communities_sets = [set(c) for c in random_clustering.communities]

    #### ----------------------------  METRICS ------------------------------
    ari_gen_random  = Utils.adjusted_random_index(original_communities_sets, random_communities_sets)
    ari_spec_random = Utils.adjusted_random_index_target(original_communities_sets, random_communities_sets, target_community)
    print("THIS IS ORIGINAL AND TARGETTTT", original_communities_sets, target_community, random_communities_sets)
    nmi_gen_random  = Utils.normalized_mutual_info(original_communities_sets, random_communities_sets)
    nmi_spec_random = Utils.normalized_mutual_info_target(original_communities_sets, random_communities_sets, target_community)

    modularity_random = nx.community.modularity(random_flat_graph, random_communities_sets)
    safeness_random   = Safeness.computeSafeness(random_flat_graph, target_community, random_communities_sets)
    deception_random  = Utils.getDeceptionScore(random_communities_sets, target_community, random_flat_graph)

    print(f"[Random] Modularity: {modularity_random}, Safeness: {safeness_random}, Deception: {deception_random}")

    # MOD STEP ----------------
    graph = MultiLayerGraph.from_3d_matrix_file(data_path)
    all_edits_mod = {}

    start_time_mod = time.time()
    for layer in graph.get_layers():
        print(f" Applying MOD deception on layer: {layer}")
        layer_graph_copy = graph.get_layer_graph(layer).copy()
        coms_filtered = [c for c in original_communities_sets]

        modified_layer, edits_mod = Utils.applyDeception(
            'MOD', layer_graph_copy, target_community, per_layer_budget, coms_filtered
        )
        graph.multilayer_graph[layer] = modified_layer
        all_edits_mod[layer] = edits_mod

    end_time_mod = time.time()
    time_mod = end_time_mod - start_time_mod

    modified_graph_path = os.path.join(modified_graphs_dir, f"{dataset_name}-scml_layerwise_c{idx}_r{n}.m")
    graph.to_m_file(modified_graph_path)

    run_scml_cd(modified_graph_path, run_suffix_modified)
    modified_com_path = find_community_file(run_suffix_modified)

    df_m = pd.read_csv(modified_com_path, header=None, names=["alias", "community"])
    grouped_m = defaultdict(set)
    for _, row in df_m.iterrows():
        node_id = alias_to_node.get(row["alias"])
        if node_id:
            grouped_m[row["community"]].add(node_id)
    modified_communities_m = [list(nodes) for nodes in grouped_m.values()]

    flat_graph_after_mod = graph.get_flatten()
    mod_clustering_obj = NodeClustering(communities=modified_communities_m, graph=flat_graph_after_mod, method_name="scml")
    mod_communities_sets = [set(c) for c in mod_clustering_obj.communities]

    #### ----------------------------  METRICS ------------------------------
    ari_gen_mod  = Utils.adjusted_random_index(original_communities_sets, mod_communities_sets)
    ari_spec_mod = Utils.adjusted_random_index_target(original_communities_sets, mod_communities_sets, target_community)
    print("THIS IS ORIGINAL AND TARGETTTT", original_communities_sets, target_community, mod_communities_sets)
    nmi_gen_mod  = Utils.normalized_mutual_info(original_communities_sets, mod_communities_sets)
    nmi_spec_mod = Utils.normalized_mutual_info_target(original_communities_sets, mod_communities_sets, target_community)

    modularity_after_mod = nx.community.modularity(flat_graph_after_mod, mod_communities_sets)
    safeness_after_mod   = Safeness.computeSafeness(flat_graph_after_mod, target_community, mod_communities_sets)
    deception_after_mod  = Utils.getDeceptionScore(mod_communities_sets, target_community, flat_graph_after_mod)

    print(f"[After MOD] Modularity: {modularity_after_mod}, Safeness: {safeness_after_mod}, Deception: {deception_after_mod}")

    # SAFDEC STEP ----------------
    graph = MultiLayerGraph.from_3d_matrix_file(data_path)
    all_edits_saf = {}
    
    target_nodes_in_flat = [node for node in target_community if node in flat_graph_original]
    induced_subgraph = flat_graph_original.subgraph(target_nodes_in_flat)

    start_time_saf = time.time()
    for layer in graph.get_layers():
        print(f" Applying deception on layer: {layer}")
        layer_graph_copy = graph.get_layer_graph(layer).copy()
        coms_filtered = [c for c in original_communities_sets]

        modified_layer, edits_saf = Utils.applyDeception(
            'SAFDEC', layer_graph_copy, target_community, per_layer_budget, coms_filtered,
            subgraph=induced_subgraph
        )
        graph.multilayer_graph[layer] = modified_layer
        all_edits_saf[layer] = edits_saf

    end_time_saf = time.time()
    time_saf = end_time_saf - start_time_saf

    modified_graph_path_saf = os.path.join(modified_graphs_dir, f"{dataset_name}-scml_layerwise-saf_c{idx}_r{n}.m")
    graph.to_m_file(modified_graph_path_saf)

    run_scml_cd(modified_graph_path_saf, run_suffix_saf)
    modified_com_path_saf = find_community_file(run_suffix_saf)

    df_s = pd.read_csv(modified_com_path_saf, header=None, names=["alias", "community"])
    grouped_s = defaultdict(set)
    for _, row in df_s.iterrows():
        node_id = alias_to_node.get(row["alias"])
        if node_id:
            grouped_s[row["community"]].add(node_id)
    modified_communities_s = [list(nodes) for nodes in grouped_s.values()]

    flat_graph_after_saf = graph.get_flatten()
    saf_clustering_obj = NodeClustering(communities=modified_communities_s, graph=flat_graph_after_saf, method_name="scml")
    saf_communities_sets = [set(c) for c in saf_clustering_obj.communities]

    #### ----------------------------  METRICS ------------------------------
    ari_gen_saf  = Utils.adjusted_random_index(original_communities_sets, saf_communities_sets)
    ari_spec_saf = Utils.adjusted_random_index_target(original_communities_sets, saf_communities_sets, target_community)
    print("THIS IS ORIGINAL AND TARGETTTT", original_communities_sets, target_community, saf_communities_sets)
    nmi_gen_saf  = Utils.normalized_mutual_info(original_communities_sets, saf_communities_sets)
    nmi_spec_saf = Utils.normalized_mutual_info_target(original_communities_sets, saf_communities_sets, target_community)
    

    modularity_after_saf = nx.community.modularity(flat_graph_after_saf, saf_communities_sets)
    safeness_after_saf   = Safeness.computeSafeness(flat_graph_after_saf, target_community, saf_communities_sets)
    deception_after_saf  = Utils.getDeceptionScore(saf_communities_sets, target_community, flat_graph_after_saf)

    print(f"[After SAFDEC] Modularity: {modularity_after_saf}, Safeness: {safeness_after_saf}, Deception: {deception_after_saf}")

    num_nodes = max(int(n) for c in original_communities_sets for n in c)
    community_structure_original = communities_to_label_array(original_communities_sets, num_nodes).tolist()
    community_structure_random   = communities_to_label_array(random_communities_sets,   num_nodes).tolist()
    community_structure_mod      = communities_to_label_array(mod_communities_sets,      num_nodes).tolist()
    community_structure_saf      = communities_to_label_array(saf_communities_sets,      num_nodes).tolist()

    run_key = f"run_{n}_community_{idx}"
    result = {
        "target_community_index": idx,
        "target_community_nodes": list(target_community),
        "modularity_before": modularity_before,
        "modularity_random": modularity_random,
        "modularity_after_mod": modularity_after_mod,
        "modularity_after_saf": modularity_after_saf,
        "safeness_before": safeness_before,
        "safeness_random": safeness_random,
        "safeness_after_mod": safeness_after_mod,
        "safeness_after_saf": safeness_after_saf,
        "deception_before": deception_before,
        "deception_random": deception_random,
        "deception_after_mod": deception_after_mod,
        "deception_after_saf": deception_after_saf,
        "ari_gen_random": ari_gen_random,
        "ari_gen_mod": ari_gen_mod,
        "ari_gen_saf": ari_gen_saf,
        "nmi_gen_random": nmi_gen_random,
        "nmi_gen_mod": nmi_gen_mod,
        "nmi_gen_saf": nmi_gen_saf,
        "ari_spec_random": ari_spec_random,
        "ari_spec_mod": ari_spec_mod,
        "ari_spec_saf": ari_spec_saf,
        "nmi_spec_random": nmi_spec_random,
        "nmi_spec_mod": nmi_spec_mod,
        "nmi_spec_saf": nmi_spec_saf,
        "time_random": time_random,
        "time_mod": time_mod,
        "time_saf": time_saf,
        "edits_random": all_edits_random,
        "edits_mod": all_edits_mod,
        "edits_saf": all_edits_saf,
        "communities_random": [list(c) for c in random_communities_sets],
        "communities_mod": [list(c) for c in mod_communities_sets],
        "communities_saf": [list(c) for c in saf_communities_sets],
        "community_structure_original": community_structure_original,
        "community_structure_random":   community_structure_random,
        "community_structure_mod":      community_structure_mod,
        "community_structure_saf":      community_structure_saf,
    }
    return run_key, result

# ---------------------------------------------------------------------------
# PMM layerwise parallel worker
# ---------------------------------------------------------------------------

def _pmm_layerwise_run_worker(args):
    (
        n, idx, target_community,
        data_path, original_communities_sets,
        modified_graphs_dir, dataset_name,
        budget, alias_to_node,
        modularity_before, safeness_before, deception_before,
        comm_algo, method,
    ) = args

    import copy, random, time, os, subprocess, glob
    import numpy as np
    import networkx as nx
    import pandas as pd
    from collections import defaultdict
    from cdlib import NodeClustering

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from codeDeception1.Safeness import Safeness
    from codeDeception1.Utils import Utils
    from MultiLayerGraph import MultiLayerGraph

    run_suffix_random   = f"random_c{idx}_r{n}"
    run_suffix_modified = f"modified_c{idx}_r{n}"
    run_suffix_saf      = f"saf_c{idx}_r{n}"

    communities_dir = os.path.join(os.getcwd(), "Communities", dataset_name)

    def find_community_file(suffix):
        pattern = os.path.join(communities_dir, f"comm-{comm_algo}-{dataset_name}-*-{suffix}.csv")
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No community file found with suffix '{suffix}'")
        return matches[0]

    def run_pmm_cd(data_path, suffix):
        command = [
            "python3", "../discover-communities-pmm.py",
            data_path,
            "--method", method,
            "--repeats", "1",
            "--suffix", suffix
        ]
        subprocess.run(command, check=True)

    per_layer_budget = budget

    print(f"\n=== RUN {n} ===")
    
    
    graph = MultiLayerGraph.from_pmm_edgelist(data_path, alias_to_node)
    flat_graph_original = graph.get_flatten()

    # RANDOM SECTION --------------
    all_edits_random = {}

    start_time_random = time.time()
    for layer in graph.get_layers():
        print(f" Applying RANDOM deception on layer: {layer}")
        layer_graph = graph.get_layer_graph(layer)
        layer_graph_copy = layer_graph.copy()
        coms_filtered = [c for c in original_communities_sets]

        modified_random_layer, random_edits = runRandomDeception(layer_graph_copy, target_community, coms_filtered, per_layer_budget)
        graph.multilayer_graph[layer] = modified_random_layer
        all_edits_random[layer] = random_edits

    end_time_random = time.time()
    time_random = end_time_random - start_time_random

    random_graph_path = os.path.join(modified_graphs_dir, f"edgelist-random-layerwise_c{idx}_r{n}.csv")
    graph.to_edgelist_csv(random_graph_path)

    print(f"Running community detection on random modified graph via script...")
    run_pmm_cd(random_graph_path, run_suffix_random)
    random_com_path = find_community_file(run_suffix_random)

    random_flat_graph = graph.get_flatten()

    df = pd.read_csv(random_com_path, header=None, names=["alias", "community"])
    grouped = defaultdict(set)
    for _, row in df.iterrows():
        node_id = alias_to_node.get(row["alias"])
        if node_id:
            grouped[row["community"]].add(node_id)
    random_communities = [list(nodes) for nodes in grouped.values()]

    random_clustering = NodeClustering(communities=random_communities, graph=random_flat_graph)
    random_communities_sets = [set(c) for c in random_clustering.communities]

       

    ###--------------METRICS--.---
    ari_gen_random  = Utils.adjusted_random_index(original_communities_sets, random_communities_sets)
    ari_spec_random = Utils.adjusted_random_index_target(original_communities_sets, random_communities_sets, target_community)
    nmi_gen_random  = Utils.normalized_mutual_info(original_communities_sets, random_communities_sets)
    nmi_spec_random = Utils.normalized_mutual_info_target(original_communities_sets, random_communities_sets, target_community)

    modularity_random = nx.community.modularity(random_flat_graph, random_communities_sets)
    safeness_random   = Safeness.computeSafeness(random_flat_graph, target_community, random_communities_sets)
    deception_random  = Utils.getDeceptionScore(random_communities_sets, target_community, random_flat_graph)
    print(" RANDOM RESULTS:      --------------------------")
    print(f"[Random] Modularity: {modularity_random}, Safeness: {safeness_random}, Deception: {deception_random}")
    print(" END RANDOM RESULTS    ------------------------------")

    # END RANDOM STEP ----------------

    # apply MOD deception
    graph = MultiLayerGraph.from_pmm_edgelist(data_path, alias_to_node)
    all_edits_mod = {}

    start_time_mod = time.time()
    for layer in graph.get_layers():
        print(f" Applying deception on layer: {layer}")
        layer_graph = graph.get_layer_graph(layer)
        layer_graph_copy = layer_graph.copy()
        layer_nodes = set(layer_graph_copy.nodes())
        coms_filtered = [c for c in original_communities_sets]

        modified_layer, edits_mod = Utils.applyDeception(
            'MOD', layer_graph_copy, target_community, per_layer_budget, coms_filtered
        )
        graph.multilayer_graph[layer] = modified_layer
        all_edits_mod[layer] = edits_mod

    end_time_mod = time.time()
    time_mod = end_time_mod - start_time_mod

    modified_graph_path = os.path.join(modified_graphs_dir, f"edgelist-layerwise_c{idx}_r{n}.csv")
    graph.to_edgelist_csv(modified_graph_path)

    print(f"Running community detection on modified graph via script...")
    run_pmm_cd(modified_graph_path, run_suffix_modified)
    modified_com_path = find_community_file(run_suffix_modified)

    df = pd.read_csv(modified_com_path, header=None, names=["alias", "community"])
    grouped = defaultdict(set)
    for _, row in df.iterrows():
        node_id = alias_to_node.get(row["alias"])
        if node_id:
            grouped[row["community"]].add(node_id)
    modified_communities = [list(nodes) for nodes in grouped.values()]

    flat_graph_after_mod = graph.get_flatten()
    modified_clustering = NodeClustering(communities=modified_communities, graph=flat_graph_after_mod)
    mod_communities_sets = [set(c) for c in modified_clustering.communities]

  

    #### ----------------------------  METRICS ------------------------------
    ari_gen_mod  = Utils.adjusted_random_index(original_communities_sets, mod_communities_sets)
    ari_spec_mod = Utils.adjusted_random_index_target(original_communities_sets, mod_communities_sets, target_community)
    nmi_gen_mod  = Utils.normalized_mutual_info(original_communities_sets, mod_communities_sets)
    nmi_spec_mod = Utils.normalized_mutual_info_target(original_communities_sets, mod_communities_sets, target_community)

    modularity_after_mod = nx.community.modularity(flat_graph_after_mod, mod_communities_sets)
    safeness_after_mod   = Safeness.computeSafeness(flat_graph_after_mod, target_community, mod_communities_sets)
    deception_after_mod  = Utils.getDeceptionScore(mod_communities_sets, target_community, flat_graph_after_mod)

    print(f"[After] Modularity: {modularity_after_mod}, Safeness: {safeness_after_mod}, Deception: {deception_after_mod}")

    # apply SAFDEC deception
    graph = MultiLayerGraph.from_pmm_edgelist(data_path, alias_to_node)
    all_edits_saf = {}
    
    target_nodes_in_flat = [n for n in target_community if n in flat_graph_original]
    induced_subgraph = flat_graph_original.subgraph(target_nodes_in_flat)

    start_time_saf = time.time()
    for layer in graph.get_layers():
        print(f" Applying deception on layer: {layer}")
        layer_graph = graph.get_layer_graph(layer)
        layer_graph_copy = layer_graph.copy()
        layer_nodes = set(layer_graph_copy.nodes())
        coms_filtered = [c for c in original_communities_sets]
        
        # subgraph induced by target community nodes in the original flat graph
        

        modified_layer, edits_saf = Utils.applyDeception(
            'SAFDEC', layer_graph_copy, target_community, per_layer_budget, coms_filtered, subgraph=induced_subgraph
        )
        graph.multilayer_graph[layer] = modified_layer
        all_edits_saf[layer] = edits_saf

    end_time_saf = time.time()
    time_saf = end_time_saf - start_time_saf

    modified_graph_path_saf = os.path.join(modified_graphs_dir, f"edgelist-layerwise-saf_c{idx}_r{n}.csv")
    graph.to_edgelist_csv(modified_graph_path_saf)

    print(f"Running community detection on modified graph via script...")
    run_pmm_cd(modified_graph_path_saf, run_suffix_saf)
    modified_com_path_saf = find_community_file(run_suffix_saf)

    df = pd.read_csv(modified_com_path_saf, header=None, names=["alias", "community"])
    grouped = defaultdict(set)
    for _, row in df.iterrows():
        node_id = alias_to_node.get(row["alias"])
        if node_id:
            grouped[row["community"]].add(node_id)
    modified_communities = [list(nodes) for nodes in grouped.values()]

    flat_graph_after_saf = graph.get_flatten()
    modified_clustering = NodeClustering(communities=modified_communities, graph=flat_graph_after_saf)
    saf_communities_sets = [set(c) for c in modified_clustering.communities]

   

    ########################################################
    ari_gen_saf  = Utils.adjusted_random_index(original_communities_sets, saf_communities_sets)
    ari_spec_saf = Utils.adjusted_random_index_target(original_communities_sets, saf_communities_sets, target_community)
    nmi_gen_saf  = Utils.normalized_mutual_info(original_communities_sets, saf_communities_sets)
    nmi_spec_saf = Utils.normalized_mutual_info_target(original_communities_sets, saf_communities_sets, target_community)

    modularity_after_saf = nx.community.modularity(flat_graph_after_saf, saf_communities_sets)
    safeness_after_saf   = Safeness.computeSafeness(flat_graph_after_saf, target_community, saf_communities_sets)
    deception_after_saf  = Utils.getDeceptionScore(saf_communities_sets, target_community, flat_graph_after_saf)

    print(f"[After SAFDEC] Modularity: {modularity_after_saf}, Safeness: {safeness_after_saf}, Deception: {deception_after_saf}")

    num_nodes = max(int(n) for c in original_communities_sets for n in c)
    community_structure_original = communities_to_label_array(original_communities_sets, num_nodes).tolist()
    community_structure_random   = communities_to_label_array(random_communities_sets,   num_nodes).tolist()
    community_structure_mod      = communities_to_label_array(mod_communities_sets,      num_nodes).tolist()
    community_structure_saf      = communities_to_label_array(saf_communities_sets,      num_nodes).tolist()

    run_key = f"run_{n}_community_{idx}"
    result = {
        "target_community_index": idx,
        "target_community_nodes": list(target_community),
        "modularity_before": modularity_before,
        "modularity_random": modularity_random,
        "modularity_after_mod": modularity_after_mod,
        "modularity_after_saf": modularity_after_saf,
        "safeness_before": safeness_before,
        "safeness_random": safeness_random,
        "safeness_after_mod": safeness_after_mod,
        "safeness_after_saf": safeness_after_saf,
        "deception_before": deception_before,
        "deception_random": deception_random,
        "deception_after_mod": deception_after_mod,
        "deception_after_saf": deception_after_saf,
        "ari_gen_random": ari_gen_random,
        "ari_gen_mod": ari_gen_mod,
        "ari_gen_saf": ari_gen_saf,
        "nmi_gen_random": nmi_gen_random,
        "nmi_gen_mod": nmi_gen_mod,
        "nmi_gen_saf": nmi_gen_saf,
        "ari_spec_random": ari_spec_random,
        "ari_spec_mod": ari_spec_mod,
        "ari_spec_saf": ari_spec_saf,
        "nmi_spec_random": nmi_spec_random,
        "nmi_spec_mod": nmi_spec_mod,
        "nmi_spec_saf": nmi_spec_saf,
        "time_random": time_random,
        "time_mod": time_mod,
        "time_saf": time_saf,
        "edits_random": all_edits_random,
        "edits_mod": all_edits_mod,
        "edits_saf": all_edits_saf,
        "communities_random": [list(c) for c in random_communities_sets],
        "communities_mod": [list(c) for c in mod_communities_sets],
        "communities_saf": [list(c) for c in saf_communities_sets],
        "community_structure_original": community_structure_original,
        "community_structure_random":   community_structure_random,
        "community_structure_mod":      community_structure_mod,
        "community_structure_saf":      community_structure_saf,
    }
    return run_key, result


# ---------------------------------------------------------------------------
# MPX layerwise parallel worker
# ---------------------------------------------------------------------------

def _mpx_layerwise_run_worker(args):
    (
        n, idx, target_community,
        data_path, original_communities_sets,
        modified_graphs_dir, dataset_name,
        budget, alias_to_node, node_to_alias,
        modularity_before, safeness_before, deception_before,
        comm_algo,
    ) = args

    import copy, random, time, os, subprocess, glob
    import numpy as np
    import networkx as nx
    import pandas as pd
    from collections import defaultdict
    from cdlib import NodeClustering

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from codeDeception1.Safeness import Safeness
    from codeDeception1.Utils import Utils
    from MultiLayerGraph import MultiLayerGraph

    run_suffix_random   = f"random_c{idx}_r{n}"
    run_suffix_modified = f"modified_c{idx}_r{n}"
    run_suffix_saf      = f"saf_c{idx}_r{n}"

    communities_dir = os.path.join(os.getcwd(), "Communities", dataset_name)

    def find_community_file(suffix):
        pattern = os.path.join(communities_dir, f"comm-{comm_algo}-{dataset_name}-*-{suffix}.csv")
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No community file found with suffix '{suffix}'")
        return matches[0]

    def run_mpx_cd(data_path, suffix):
        command = [
            "python3", "../discover-communities-mpx.py",
            data_path,
            "--algorithm", comm_algo,
            "--repeats", "1",
            "--timeout", "3600",
            "--suffix", suffix
        ]
        subprocess.run(command, check=True)

    per_layer_budget = budget

    print(f"\n=== Run {n} ===")

    graph = MultiLayerGraph.from_file(
        os.path.dirname(data_path),
        os.path.basename(data_path),
        alias_map=alias_to_node
    )
    flat_graph_original = graph.get_flatten()

    # RANDOM SECTION ----------------
    all_edits_random = {}

    start_time_random = time.time()
    for layer in graph.get_layers():
        print(f" Applying RANDOM deception on layer: {layer}")
        layer_graph = graph.get_layer_graph(layer)
        layer_graph_copy = layer_graph.copy()
        layer_nodes = set(layer_graph_copy.nodes())
        coms_filtered = [c for c in original_communities_sets]

        modified_random_layer, edits_random = runRandomDeception(layer_graph_copy, target_community, coms_filtered, per_layer_budget)
        graph.multilayer_graph[layer] = modified_random_layer
        all_edits_random[layer] = edits_random

    end_time_random = time.time()
    time_random = end_time_random - start_time_random

    random_graph_path = os.path.join(modified_graphs_dir, f"{dataset_name}_layerwise-random_c{idx}_r{n}.mpx")
    graph.to_mpx(random_graph_path, node_map=node_to_alias)

    print(f"Running community detection on random modified graph via script...")
    run_mpx_cd(random_graph_path, run_suffix_random)
    random_com_path = find_community_file(run_suffix_random)

    random_flat_graph = graph.get_flatten()

    df = pd.read_csv(random_com_path, header=None, names=["alias", "layer", "community"])
    grouped = defaultdict(set)
    for _, row in df.iterrows():
        node_id = alias_to_node.get(row["alias"])
        if node_id:
            grouped[row["community"]].add(node_id)
    random_communities = [list(nodes) for nodes in grouped.values()]

    random_clustering = NodeClustering(communities=random_communities, graph=random_flat_graph)
    random_communities_sets = [set(c) for c in random_clustering.communities]

    

    ###--------------METRICS--.---
    ari_gen_random  = Utils.adjusted_random_index(original_communities_sets, random_communities_sets)
    ari_spec_random = Utils.adjusted_random_index_target(original_communities_sets, random_communities_sets, target_community)
    nmi_gen_random  = Utils.normalized_mutual_info(original_communities_sets, random_communities_sets)
    nmi_spec_random = Utils.normalized_mutual_info_target(original_communities_sets, random_communities_sets, target_community)

    modularity_random = nx.community.modularity(random_flat_graph, random_communities_sets)
    safeness_random   = Safeness.computeSafeness(random_flat_graph, target_community, random_communities_sets)
    deception_random  = Utils.getDeceptionScore(random_communities_sets, target_community, random_flat_graph)
    

    print(" RANDOM RESULTS:      --------------------------")
    print(f"[Random] Modularity: {modularity_random}, Safeness: {safeness_random}, Deception: {deception_random}")
    print(" END RANDOM RESULTS    ------------------------------")

    # END RANDOM SECTION ------------

    # apply MOD deception
    graph = MultiLayerGraph.from_file(
        os.path.dirname(data_path),
        os.path.basename(data_path),
        alias_map=alias_to_node
    )

    all_edits_mod = {}

    start_time_mod = time.time()
    for layer in graph.get_layers():
        print(f" Applying deception on layer: {layer}")
        layer_graph = graph.get_layer_graph(layer)
        layer_graph_copy = layer_graph.copy()
        layer_nodes = set(layer_graph_copy.nodes())
        coms_filtered = [c for c in original_communities_sets]

        modified_layer, edits_mod = Utils.applyDeception(
            'MOD', layer_graph_copy, target_community, per_layer_budget, coms_filtered
        )
        graph.multilayer_graph[layer] = modified_layer
        all_edits_mod[layer] = edits_mod

    end_time_mod = time.time()
    time_mod = end_time_mod - start_time_mod

    modified_graph_path = os.path.join(modified_graphs_dir, f"{dataset_name}_layerwise_c{idx}_r{n}.mpx")
    graph.to_mpx(modified_graph_path, node_map=node_to_alias)

    print(f"Running community detection on modified graph via script...")
    run_mpx_cd(modified_graph_path, run_suffix_modified)
    modified_com_path = find_community_file(run_suffix_modified)

    df = pd.read_csv(modified_com_path, header=None, names=["alias", "layer", "community"])
    grouped = defaultdict(set)
    for _, row in df.iterrows():
        node_id = alias_to_node.get(row["alias"])
        if node_id:
            grouped[row["community"]].add(node_id)
    modified_communities = [list(nodes) for nodes in grouped.values()]

    flat_graph_after_mod = graph.get_flatten()
    modified_clustering = NodeClustering(communities=modified_communities, graph=flat_graph_after_mod)
    mod_communities_sets = [set(c) for c in modified_clustering.communities]


    #### ----------------------------  METRICS ------------------------------
    ari_gen_mod  = Utils.adjusted_random_index(original_communities_sets, mod_communities_sets)
    ari_spec_mod = Utils.adjusted_random_index_target(original_communities_sets, mod_communities_sets, target_community)
    nmi_gen_mod  = Utils.normalized_mutual_info(original_communities_sets, mod_communities_sets)
    nmi_spec_mod = Utils.normalized_mutual_info_target(original_communities_sets, mod_communities_sets, target_community)

    modularity_after_mod = nx.community.modularity(flat_graph_after_mod, mod_communities_sets)
    safeness_after_mod   = Safeness.computeSafeness(flat_graph_after_mod, target_community, mod_communities_sets)
    deception_after_mod  = Utils.getDeceptionScore(mod_communities_sets, target_community, flat_graph_after_mod)

    print(f"[After] Modularity: {modularity_after_mod}, Safeness: {safeness_after_mod}, Deception: {deception_after_mod}")

    # apply SAFDEC deception
    graph = MultiLayerGraph.from_file(
        os.path.dirname(data_path),
        os.path.basename(data_path),
        alias_map=alias_to_node
    )

    target_nodes_in_flat = [node for node in target_community if node in flat_graph_original]
    induced_subgraph = flat_graph_original.subgraph(target_nodes_in_flat)

    all_edits_saf = {}

    start_time_saf = time.time()
    for layer in graph.get_layers():
        print(f"Applying deception on layer: {layer}")
        layer_graph = graph.get_layer_graph(layer)
        layer_graph_copy = layer_graph.copy()
        layer_nodes = set(layer_graph_copy.nodes())
        coms_filtered = [c for c in original_communities_sets]

        modified_layer, edits_saf = Utils.applyDeception(
            'SAFDEC', layer_graph_copy, target_community, per_layer_budget, coms_filtered,
            subgraph=induced_subgraph
        )
        graph.multilayer_graph[layer] = modified_layer
        all_edits_saf[layer] = edits_saf

    end_time_saf = time.time()
    time_saf = end_time_saf - start_time_saf

    modified_graph_path_saf = os.path.join(modified_graphs_dir, f"{dataset_name}_layerwise-saf_c{idx}_r{n}.mpx")
    graph.to_mpx(modified_graph_path_saf, node_map=node_to_alias)

    print(f"Running community detection on modified graph via script...")
    run_mpx_cd(modified_graph_path_saf, run_suffix_saf)
    modified_com_path_saf = find_community_file(run_suffix_saf)

    df = pd.read_csv(modified_com_path_saf, header=None, names=["alias", "layer", "community"])
    grouped = defaultdict(set)
    for _, row in df.iterrows():
        node_id = alias_to_node.get(row["alias"])
        if node_id:
            grouped[row["community"]].add(node_id)
    modified_communities = [list(nodes) for nodes in grouped.values()]

    flat_graph_after_saf = graph.get_flatten()
    modified_clustering = NodeClustering(communities=modified_communities, graph=flat_graph_after_saf)
    saf_communities_sets = [set(c) for c in modified_clustering.communities]

 
    #### ----------------------------  METRICS ------------------------------
    ari_gen_saf  = Utils.adjusted_random_index(original_communities_sets, saf_communities_sets)
    ari_spec_saf = Utils.adjusted_random_index_target(original_communities_sets, saf_communities_sets, target_community)
    nmi_gen_saf  = Utils.normalized_mutual_info(original_communities_sets, saf_communities_sets)
    nmi_spec_saf = Utils.normalized_mutual_info_target(original_communities_sets, saf_communities_sets, target_community)

    modularity_after_saf = nx.community.modularity(flat_graph_after_saf, saf_communities_sets)
    safeness_after_saf   = Safeness.computeSafeness(flat_graph_after_saf, target_community, saf_communities_sets)
    deception_after_saf  = Utils.getDeceptionScore(saf_communities_sets, target_community, flat_graph_after_saf)

    print(f"[After SAFDEC] Modularity: {modularity_after_saf}, Safeness: {safeness_after_saf}, Deception: {deception_after_saf}")

    num_nodes = max(int(n) for c in original_communities_sets for n in c)
    community_structure_original = communities_to_label_array(original_communities_sets, num_nodes).tolist()
    community_structure_random   = communities_to_label_array(random_communities_sets,   num_nodes).tolist()
    community_structure_mod      = communities_to_label_array(mod_communities_sets,      num_nodes).tolist()
    community_structure_saf      = communities_to_label_array(saf_communities_sets,      num_nodes).tolist()

    run_key = f"run_{n}_community_{idx}"
    result = {
        "target_community_index": idx,
        "target_community_nodes": list(target_community),
        "modularity_before": modularity_before,
        "modularity_random": modularity_random,
        "modularity_after_mod": modularity_after_mod,
        "modularity_after_saf": modularity_after_saf,
        "safeness_before": safeness_before,
        "safeness_random": safeness_random,
        "safeness_after_mod": safeness_after_mod,
        "safeness_after_saf": safeness_after_saf,
        "deception_before": deception_before,
        "deception_random": deception_random,
        "deception_after_mod": deception_after_mod,
        "deception_after_saf": deception_after_saf,
        "ari_gen_random": ari_gen_random,
        "ari_gen_mod": ari_gen_mod,
        "ari_gen_saf": ari_gen_saf,
        "nmi_gen_random": nmi_gen_random,
        "nmi_gen_mod": nmi_gen_mod,
        "nmi_gen_saf": nmi_gen_saf,
        "ari_spec_random": ari_spec_random,
        "ari_spec_mod": ari_spec_mod,
        "ari_spec_saf": ari_spec_saf,
        "nmi_spec_random": nmi_spec_random,
        "nmi_spec_mod": nmi_spec_mod,
        "nmi_spec_saf": nmi_spec_saf,
        "time_random": time_random,
        "time_mod": time_mod,
        "time_saf": time_saf,
        "edits_random": all_edits_random,
        "edits_mod": all_edits_mod,
        "edits_saf": all_edits_saf,
        "communities_random": [list(c) for c in random_communities_sets],
        "communities_mod": [list(c) for c in mod_communities_sets],
        "communities_saf": [list(c) for c in saf_communities_sets],
        "community_structure_original": community_structure_original,
        "community_structure_random":   community_structure_random,
        "community_structure_mod":      community_structure_mod,
        "community_structure_saf":      community_structure_saf,
    }
    return run_key, result

# ---------------------------------------------------------------------------


class LayerwiseDeceptionEvaluator:

    NUM_WORKERS = 4

    def __init__(self, data_path, deception_algo, budget, n_run, comm_algo="flat_nw"):
        self.data_path = os.path.abspath(data_path)
        self.deception_algo = deception_algo.upper()
        self.budget = budget
        self.comm_algo = comm_algo
        self.n_run = n_run
        if (self.comm_algo == "scml" and self.data_path.endswith(".m")) or self.comm_algo.startswith("pmm_"):
            self.dataset_name = os.path.basename(os.path.dirname(self.data_path))
            print(self.dataset_name)
        else:
            self.dataset_name = os.path.splitext(os.path.basename(self.data_path))[0]
            print(self.dataset_name)
        self.communities_dir = os.path.join(os.getcwd(), "Communities", self.dataset_name)
        self.modified_graphs_dir = os.path.join(os.getcwd(), "NewMultiGraph", self.dataset_name)
        Path(self.modified_graphs_dir).mkdir(parents=True, exist_ok=True)
        Path(self.communities_dir).mkdir(parents=True, exist_ok=True)


    def run_community_detection_script(self, data_path, suffix):
        command = [
            "python3", "../discover-communities-mpx.py",
            data_path,
            "--algorithm", self.comm_algo,
            "--repeats", "1",
            "--timeout", "3600",
            "--suffix", suffix
        ]
        try:
            subprocess.run(command, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Community detection script failed: {e}")
            return False

    def run_scml_community_detection(self, data_path, suffix):
        command = [
            "python3", "../discover-communities-scml.py",
            data_path,
            "--repeats", "1",
            "--suffix", suffix
        ]
        try:
            subprocess.run(command, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ SCML community detection failed: {e}")
            return False

    def run_pmm_community_detection(self, data_path, suffix, method="h"):
        command = [
            "python3", "../discover-communities-pmm.py",
            data_path,
            "--method", method,
            "--repeats", "1",
            "--suffix", suffix
        ]
        try:
            subprocess.run(command, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ PMM community detection failed: {e}")
            return False

    def find_community_file(self, suffix="original"):
        pattern = os.path.join(self.communities_dir, f"comm-{self.comm_algo}-{self.dataset_name}-*-{suffix}.csv")
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No community detection files found with suffix '{suffix}'")
        return matches[0]


    def evaluate(self):

        if self.comm_algo.lower() == "scml":
            print(self.dataset_name)
            print(f"Loading SCML .m file as multilayer graph from {self.data_path}")
            graph = MultiLayerGraph.from_3d_matrix_file(self.data_path)

            actors_csv_path = os.path.join(os.path.dirname(self.data_path), "actors.csv")
            with open(actors_csv_path, "r") as f:
                actors = [line.strip() for line in f if line.strip()]
            alias_to_node = {alias: str(i + 1) for i, alias in enumerate(actors)}

            print(f"Running initial SCML community detection via script...")
            start_time = time.time()
            self.run_scml_community_detection(self.data_path, "original")
            time_of_community_detection = time.time() - start_time
            print("TIMEDETECTION:", time_of_community_detection)
            original_com_path = self.find_community_file("original")

            df = pd.read_csv(original_com_path, header=None, names=["alias", "community"])
            grouped = defaultdict(set)
            for _, row in df.iterrows():
                node_id = alias_to_node.get(row["alias"])
                if node_id:
                    grouped[row["community"]].add(node_id)
            original_communities = [list(nodes) for nodes in grouped.values()]

            flat_graph_before = graph.get_flatten()
            original_clustering = NodeClustering(communities=original_communities, graph=flat_graph_before, method_name="scml")
            original_communities_sets = [set(c) for c in original_clustering.communities]

            before_metrics = {}
            for idx, target_community in enumerate(original_communities_sets):
                print(f"\n=== Targeting Community {idx} ===")
                modularity_before = nx.community.modularity(flat_graph_before, original_communities_sets)
                safeness_before   = Safeness.computeSafeness(flat_graph_before, target_community, original_communities_sets)
                deception_before  = Utils.getDeceptionScore(original_communities_sets, target_community, flat_graph_before)
                print(f"[Before] Modularity: {modularity_before}, Safeness: {safeness_before}, Deception: {deception_before}")
                before_metrics[idx] = (target_community, modularity_before, safeness_before, deception_before)

            all_args = [
                (
                    n, idx, target_community,
                    self.data_path, original_communities_sets,
                    self.modified_graphs_dir, self.dataset_name,
                    self.budget, alias_to_node,
                    modularity_before, safeness_before, deception_before,
                    self.comm_algo,
                )
                for idx, (target_community, modularity_before, safeness_before, deception_before)
                in before_metrics.items()
                for n in range(self.n_run)
            ]

            print(f"Dispatching {len(all_args)} total jobs to Pool({self.NUM_WORKERS})...")
            results = {}
            with Pool(processes=self.NUM_WORKERS) as pool:
                for run_key, result in pool.imap_unordered(_scml_layerwise_run_worker, all_args):
                    results[run_key] = result

            return results, time_of_community_detection


        elif self.comm_algo.lower().startswith("pmm_"):

            print(f"Loading Edgelist (for PMM) file as multilayer graph from {self.data_path}")
            actors_csv_path = os.path.join(os.path.dirname(self.data_path), "actors.csv")
            with open(actors_csv_path, "r") as f:
                actors = [line.strip() for line in f if line.strip()]
            alias_to_node = {alias: str(i + 1) for i, alias in enumerate(actors)}

            graph = MultiLayerGraph.from_pmm_edgelist(self.data_path, alias_to_node)

            method = self.comm_algo.lower().split("_")[-1]

            print(f"Running initial PMM community detection on original graph...")
            start_time = time.time()
            self.run_pmm_community_detection(self.data_path, "original", method)
            time_of_community_detection = time.time() - start_time
            print("TIMEDETECTION:", time_of_community_detection)
            original_com_path = self.find_community_file("original")

            df = pd.read_csv(original_com_path, header=None, names=["alias", "community"])
            grouped = defaultdict(set)
            for _, row in df.iterrows():
                node_id = alias_to_node.get(row["alias"])
                if node_id:
                    grouped[row["community"]].add(node_id)
            original_communities = [list(nodes) for nodes in grouped.values()]

            flat_graph_before = graph.get_flatten()
            original_clustering = NodeClustering(communities=original_communities, graph=flat_graph_before, method_name="pmm")
            original_communities_sets = [set(c) for c in original_clustering.communities]

            before_metrics = {}
            for idx, target_community in enumerate(original_communities_sets):
                print(f"\n=== Targeting Community {idx} ===")
                modularity_before = nx.community.modularity(flat_graph_before, original_communities_sets)
                safeness_before   = Safeness.computeSafeness(flat_graph_before, target_community, original_communities_sets)
                deception_before  = Utils.getDeceptionScore(original_communities_sets, target_community, flat_graph_before)
                print(f"[Before] Modularity: {modularity_before}, Safeness: {safeness_before}, Deception: {deception_before}")
                before_metrics[idx] = (target_community, modularity_before, safeness_before, deception_before)

            all_args = [
                (
                    n, idx, target_community,
                    self.data_path, original_communities_sets,
                    self.modified_graphs_dir, self.dataset_name,
                    self.budget, alias_to_node,
                    modularity_before, safeness_before, deception_before,
                    self.comm_algo, method,
                )
                for idx, (target_community, modularity_before, safeness_before, deception_before)
                in before_metrics.items()
                for n in range(self.n_run)
            ]

            print(f"Dispatching {len(all_args)} total jobs to Pool({self.NUM_WORKERS})...")
            results = {}
            with Pool(processes=self.NUM_WORKERS) as pool:
                for run_key, result in pool.imap_unordered(_pmm_layerwise_run_worker, all_args):
                    results[run_key] = result

            return results, time_of_community_detection


        else:      # --------------------------------MPX ALGORITHMS ---------------------------

            # ALIASING
            actors_csv_path = os.path.join(os.path.dirname(self.data_path), "actors.csv")
            with open(actors_csv_path, "r") as f:
                actors = [line.strip() for line in f if line.strip()]
                alias_to_node = {alias: str(i + 1) for i, alias in enumerate(actors)}

            node_to_alias = {v: k for k, v in alias_to_node.items()}

            graph = MultiLayerGraph.from_file(
                os.path.dirname(self.data_path),
                os.path.basename(self.data_path),
                alias_map=alias_to_node
            )

            print(f"Running initial community detection via script...")
            start_time = time.time()
            self.run_community_detection_script(self.data_path, "original")
            time_of_community_detection = time.time() - start_time
            original_com_path = self.find_community_file("original")

            df = pd.read_csv(original_com_path, header=None, names=["alias", "layer", "community"])
            grouped = defaultdict(set)
            for _, row in df.iterrows():
                node_id = alias_to_node.get(row["alias"])
                if node_id:
                    grouped[row["community"]].add(node_id)
            original_communities = [list(nodes) for nodes in grouped.values()]

            flat_graph_before = graph.get_flatten()
            original_clustering = NodeClustering(communities=original_communities, graph=flat_graph_before, method_name="mpx")
            original_communities_sets = [set(c) for c in original_clustering.communities]

            before_metrics = {}
            for idx, target_community in enumerate(original_communities_sets):
                print(f"\n=== Targeting Community {idx} ===")
                modularity_before = nx.community.modularity(flat_graph_before, original_communities_sets)
                safeness_before   = Safeness.computeSafeness(flat_graph_before, target_community, original_communities_sets)
                deception_before  = Utils.getDeceptionScore(original_communities_sets, target_community, flat_graph_before)
                print(f"[Before] Modularity: {modularity_before}, Safeness: {safeness_before}, Deception: {deception_before}")
                before_metrics[idx] = (target_community, modularity_before, safeness_before, deception_before)

            all_args = [
                (
                    n, idx, target_community,
                    self.data_path, original_communities_sets,
                    self.modified_graphs_dir, self.dataset_name,
                    self.budget, alias_to_node, node_to_alias,
                    modularity_before, safeness_before, deception_before,
                    self.comm_algo,
                )
                for idx, (target_community, modularity_before, safeness_before, deception_before)
                in before_metrics.items()
                for n in range(self.n_run)
            ]

            print(f"Dispatching {len(all_args)} total jobs to Pool({self.NUM_WORKERS})...")
            results = {}
            with Pool(processes=self.NUM_WORKERS) as pool:
                for run_key, result in pool.imap_unordered(_mpx_layerwise_run_worker, all_args):
                    results[run_key] = result

            return results, time_of_community_detection


import argparse

def main():
    parser = argparse.ArgumentParser(description="Run layerwise deception evaluation pipeline.")
    parser.add_argument("data_path", help="Path to the input .mpx file")
    parser.add_argument("deception_algo", help="Deception algorithm to use (e.g., SAFDEC, MOD)")
    parser.add_argument("budget", type=int, help="Per-Layer budget for deception")
    parser.add_argument("--n_run", type=int, help="Number of runs")
    parser.add_argument("--comm_algo", default="flat_nw", help="Community detection algorithm (default: flat_nw)")

    args = parser.parse_args()

    print("=== Layerwise Deception Evaluation ===")
    print(f"Data Path      : {args.data_path}")
    print(f"Deception Algo: {args.deception_algo}")
    print(f"Budget        : {args.budget}")
    print(f"N_Runs       : {args.n_run}")
    print(f"Comm Algo     : {args.comm_algo}")
    print("======================================")

    evaluator = LayerwiseDeceptionEvaluator(
        data_path=args.data_path,
        deception_algo=args.deception_algo,
        budget=args.budget,
        n_run=args.n_run,
        comm_algo=args.comm_algo
    )

    metrics, initial_comm_time = evaluator.evaluate()

    dataset_name = ""
    if (args.comm_algo == "scml" and args.data_path.endswith(".m")) or args.comm_algo.startswith("pmm_"):
        dataset_name = os.path.basename(os.path.dirname(args.data_path))
    else:
        dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]

    output_dir = f"./results/{dataset_name}/BUDGET-{args.budget}"
    os.makedirs(output_dir, exist_ok=True)

    base        = f"{dataset_name}_{args.comm_algo}_budget{args.budget}"
    csv_dir     = os.path.join(output_dir, f"deception_{args.comm_algo}_budget-{args.budget}_Layer-by-Layer-DECEPTION")
    os.makedirs(csv_dir, exist_ok=True)
    runs_csv    = os.path.join(csv_dir, f"results_runs_{base}.csv")
    summary_txt = os.path.join(csv_dir, f"results_summary_{base}.txt")

    import json, csv
    from collections import defaultdict

    SCALAR_METRICS = [
        "modularity_before", "modularity_random", "modularity_after_mod", "modularity_after_saf",
        "safeness_before", "safeness_random", "safeness_after_mod", "safeness_after_saf",
        "deception_before", "deception_random", "deception_after_mod", "deception_after_saf",
        "ari_gen_random", "ari_gen_mod", "ari_gen_saf",
        "nmi_gen_random", "nmi_gen_mod", "nmi_gen_saf",
        "ari_spec_random", "ari_spec_mod", "ari_spec_saf",
        "nmi_spec_random", "nmi_spec_mod", "nmi_spec_saf",
        "time_random", "time_mod", "time_saf",
    ]
    LIST_FIELDS = [
        "target_community_nodes",
        "edits_random", "edits_mod", "edits_saf",
        "communities_random", "communities_mod", "communities_saf",
        "community_structure_original",
        "community_structure_random",
        "community_structure_mod",
        "community_structure_saf",
    ]
    AGGREGATE_METRICS = [m for m in SCALAR_METRICS if m not in (
        "modularity_before", "safeness_before", "deception_before",
        "safeness_random", "safeness_after_mod", "safeness_after_saf",
    )]

    def _to_int_ids(obj):
        if isinstance(obj, str):
            return int(obj) if obj.isdigit() else obj
        if isinstance(obj, dict):
            return {k: _to_int_ids(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_int_ids(x) for x in obj]
        return obj

    run_rows = []
    for run_key, data in metrics.items():
        parts    = run_key.split("_")
        run_idx  = int(parts[1])
        comm_idx = int(parts[3])
        row = {
            "run_index":              run_idx,
            "target_community_index": comm_idx,
            "dataset_name":           dataset_name,
            "comm_algo":              args.comm_algo,
            "budget":                 args.budget,
        }
        for m in SCALAR_METRICS:
            v = data.get(m, "")
            row[m] = round(v, 4) if isinstance(v, float) else v
        for lf in LIST_FIELDS:
            raw = data.get(lf, [])
            if isinstance(raw, str):
                raw = json.loads(raw)
            row[lf] = json.dumps(_to_int_ids(raw))
        run_rows.append(row)

    run_rows.sort(key=lambda r: (r["target_community_index"], r["run_index"]))

    run_fields = (
        ["run_index", "target_community_index", "dataset_name", "comm_algo", "budget"]
        + SCALAR_METRICS
        + LIST_FIELDS
    )
    # AFTER
    import ast

    def _safe_serialize(val):
        """Convert Python dict/list strings to proper JSON strings."""
        if isinstance(val, str):
            try:
                parsed = ast.literal_eval(val)
                return json.dumps(parsed)
            except (ValueError, SyntaxError):
                return val
        return val

    # Serialize the dict-valued scalar fields (safeness_*, modularity per-node dicts etc.)
    DICT_SCALAR_FIELDS = [
        "safeness_before", "safeness_random", "safeness_after_mod", "safeness_after_saf",
    ]
    for row in run_rows:
        for field in DICT_SCALAR_FIELDS:
            if field in row:
                row[field] = _safe_serialize(row[field])

    with open(runs_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=run_fields,
            delimiter=";",
            quotechar='"',
            quoting=csv.QUOTE_NONNUMERIC,
        )
        writer.writeheader()
        writer.writerows(run_rows)
   
    print(f"Per-run CSV written to {runs_csv}")

    per_community_data = defaultdict(list)
    for run_key, data in metrics.items():
        per_community_data[data["target_community_index"]].append(data)

    community_all_means = defaultdict(dict)
    for comm_idx in sorted(per_community_data):
        agg = defaultdict(list)
        for run_data in per_community_data[comm_idx]:
            for metric, value in run_data.items():
                if metric in AGGREGATE_METRICS and isinstance(value, (int, float)):
                    agg[metric].append(value)
        for metric, vals in agg.items():
            community_all_means[comm_idx][metric] = sum(vals) / len(vals)

    metric_groups = {
        "Deception":      ["deception_random", "deception_after_mod", "deception_after_saf"],
        "Modularity":     ["modularity_random", "modularity_after_mod", "modularity_after_saf"],
        "Execution Time": ["time_random", "time_mod", "time_saf"],
        "ARI Metrics":    ["ari_gen_random", "ari_gen_mod", "ari_gen_saf",
                           "ari_spec_random", "ari_spec_mod", "ari_spec_saf"],
        "NMI Metrics":    ["nmi_gen_random", "nmi_gen_mod", "nmi_gen_saf",
                           "nmi_spec_random", "nmi_spec_mod", "nmi_spec_saf"],
    }
    HAS_MINMAX = {"Deception", "Modularity", "ARI Metrics", "NMI Metrics"}

    delta_configs = [
        ("Deception Delta (SAF - Random)",   "deception_after_saf", "deception_random"),
        ("Deception Delta (MOD - Random)",   "deception_after_mod", "deception_random"),
        ("Time Delta (SAF - Random)",        "time_saf",            "time_random"),
        ("Time Delta (MOD - Random)",        "time_mod",            "time_random"),
        ("ARI General Delta (SAF - Random)", "ari_gen_saf",         "ari_gen_random"),
        ("ARI General Delta (MOD - Random)", "ari_gen_mod",         "ari_gen_random"),
        ("ARI Target Delta (SAF - Random)",  "ari_spec_saf",        "ari_spec_random"),
        ("ARI Target Delta (MOD - Random)",  "ari_spec_mod",        "ari_spec_random"),
        ("NMI General Delta (SAF - Random)", "nmi_gen_saf",         "nmi_gen_random"),
        ("NMI General Delta (MOD - Random)", "nmi_gen_mod",         "nmi_gen_random"),
        ("NMI Target Delta (SAF - Random)",  "nmi_spec_saf",        "nmi_spec_random"),
        ("NMI Target Delta (MOD - Random)",  "nmi_spec_mod",        "nmi_spec_random"),
    ]

    with open(summary_txt, "w") as f:
        f.write("--- Deception Evaluation Metadata ---\n")
        f.write(f"Dataset Name        : {dataset_name}\n")
        f.write(f"Deception Algorithm : {args.deception_algo}\n")
        f.write(f"Community Algorithm : {args.comm_algo}\n")
        f.write(f"Budget              : {args.budget}\n")
        f.write(f"Number of Runs      : {args.n_run}\n")
        f.write(f"Community Detection Execution Time (one execution)   : {initial_comm_time:.4f}s\n")
        f.write("-------------------------------------\n")

        for group_name, methods in metric_groups.items():
            f.write(f"\n\n===== Global {group_name} Summary =====\n")
            for method in methods:
                comm_vals = [(c, m[method]) for c, m in community_all_means.items() if method in m]
                if not comm_vals:
                    continue
                mean_of_means = sum(val for _, val in comm_vals) / len(comm_vals)
                f.write(f"\n>> Method: {method}\n")
                if group_name in HAS_MINMAX:
                    min_comm, min_val = min(comm_vals, key=lambda x: x[1])
                    max_comm, max_val = max(comm_vals, key=lambda x: x[1])
                    f.write(f"   Community with MIN {group_name.lower()}: {min_comm} ({min_val:.4f})\n")
                    f.write(f"   Community with MAX {group_name.lower()}: {max_comm} ({max_val:.4f})\n")
                f.write(f"   Mean {group_name.lower()} across all communities: {mean_of_means:.4f}\n")

        f.write("\n\n===== Global Delta Summary (vs Random) =====\n")
        for title, target, baseline in delta_configs:
            deltas = []
            for c_idx, values in community_all_means.items():
                if target in values and baseline in values:
                    deltas.append((c_idx, values[target] - values[baseline]))
            if deltas:
                mean_delta = sum(v for _, v in deltas) / len(deltas)
                f.write(f"\n>> {title}\n")
                if "Deception" in title:
                    min_c, min_v = min(deltas, key=lambda x: x[1])
                    max_c, max_v = max(deltas, key=lambda x: x[1])
                    f.write(f"   Community with MIN delta: {min_c} ({min_v:.4f})\n")
                    f.write(f"   Community with MAX delta: {max_c} ({max_v:.4f})\n")
                f.write(f"   Mean delta across all communities: {mean_delta:.4f}\n")

    print(f"Summary TXT written to {summary_txt}")


if __name__ == "__main__":
    main()
