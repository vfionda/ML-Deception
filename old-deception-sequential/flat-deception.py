import os
import subprocess
import sys
import copy
import random
from pathlib import Path
import glob

# used for 
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


def load_communities_cdlib(path_to_csv, graph):
    """
    Load a community CSV into a cdlib NodeClustering.
    Keeps only node ids (e.g., 'a36'), ignoring layers (e.g., 'l1').
    """
    df = pd.read_csv(path_to_csv, header=None, names=["node", "layer", "community"])
    grouped = defaultdict(set)
    for _, row in df.iterrows():
        base_node = row["node"]
        grouped[row["community"]].add(base_node)

    communities = [list(nodes) for nodes in grouped.values()]
    
    return NodeClustering(communities=communities, graph=graph, method_name="custom_csv_import")


def get_target_community(clustering):
    """
    Returns the nodes in the first community (for now).
    """
    return random.choice(clustering.communities) 





def runRandomDeception(g, targetC, coms, budget):
    
    beta = int(budget)

    modified_graph = copy.deepcopy(g)
    edits = []
    remaining_budget = beta
    targetC=set(targetC)

    all_nodes = list(modified_graph.nodes())

    # Flatten communities for quick membership checks
    node_to_community = {node: idx for idx, community in enumerate(coms) for node in community}
    communities = [set(c) for c in coms]

    while remaining_budget > 0:
        op_type = random.choice(["intra_add", "intra_del", "inter_add", "inter_del"])

        if op_type == "intra_add":
            if len(targetC)>1:
                u, v = random.sample(sorted(targetC), 2)
                if not modified_graph.has_edge(u, v):
                    modified_graph.add_edge(u, v)
                    edits.append(("intraA", (u, v)))
                    remaining_budget -= 1

        elif op_type == "intra_del":
            # selecat a random edge inside a community that overlaps with targetC
            candidates = []
            for com in communities:
                if len(com & targetC) > 0:
                    for u in com:
                        for v in com:
                            if u != v and modified_graph.has_edge(u, v):
                                candidates.append((u, v))
            if not candidates:
                continue
            u, v = random.choice(candidates)
            modified_graph.remove_edge(u, v)
            edits.append(("intraD", (u, v)))
            remaining_budget -= 1

        elif op_type == "inter_add":
            # Pick one node in targetC and one in non-targetC from different communities
            valid_targets = list(targetC)
            non_targets = [n for n in all_nodes if n not in targetC]
            if not valid_targets or not non_targets:
                continue
            u = random.choice(valid_targets)
            v = random.choice(non_targets)
            if node_to_community.get(u) != node_to_community.get(v) and not modified_graph.has_edge(u, v):
                modified_graph.add_edge(u, v)
                edits.append(("interA", (u, v)))
                remaining_budget -= 1

        elif op_type == "inter_del":
            # Select an edge between targetC and another community
            candidates = []
            for u in targetC:
                for v in modified_graph.neighbors(u):
                    if node_to_community.get(u) != node_to_community.get(v):
                        candidates.append((u, v))
            if not candidates:
                continue
            u, v = random.choice(candidates)
            if modified_graph.has_edge(u, v):
                modified_graph.remove_edge(u, v)
                edits.append(("interD", (u, v)))
                remaining_budget -= 1

    return modified_graph, edits




class DeceptionEvaluator:
    
    
    def __init__(self, data_path, deception_algo, budget, n_run,comm_algo="flat_nw"):
        self.data_path = os.path.abspath(data_path)
        self.deception_algo = deception_algo.upper()
        self.budget = budget
        self.comm_algo = comm_algo
        self.n_run=n_run
        
        if (self.comm_algo == "scml" and self.data_path.endswith(".m")) or self.comm_algo.startswith("pmm_"):
            # for SCML, use the parent folder name of the .m file
            self.dataset_name = os.path.basename(os.path.dirname(self.data_path))
            print(self.dataset_name)
        else:
            # Default: use filename without extension
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
        """
        Runs community detection for SCML .m files using Octave.
        Assumes that 'data_path' points to 'scml.m' and the same directory contains 'actors.csv' and 'k.m'.
        """
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
        """
        Runs community detection for PMM .m files using Octave.
        Assumes that 'data_path' points to 'edgelist.csv' and the same directory contains 'actors.csv' and 'k.m'.
        
        Parameters:
        - data_path: Path to edgelist.csv (or any file in the dataset directory)
        - suffix: One of 'original', 'modified', or 'random'
        - method: 'h' for high-level PMM (default), 'l' for low-level
        """
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
        """Find the first community .csv file in the Communities folder with given suffix."""
        pattern = os.path.join(self.communities_dir, f"comm-{self.comm_algo}-{self.dataset_name}-*-{suffix}.csv")
        matches = sorted(glob.glob(pattern))  # sorted to ensure deterministic order
        if not matches:
            raise FileNotFoundError(f"No community detection files found with suffix '{suffix}'")
        return matches[0]








    def evaluate(self):
        if self.comm_algo.lower() == "scml":
            
            print(f"Loading SCML .m file as multilayer graph from {self.data_path}")
            graph = MultiLayerGraph.from_3d_matrix_file(self.data_path)

            # Load actors.csv mapping aliases to node IDs (1-based)
            actors_csv_path = os.path.join(os.path.dirname(self.data_path), "actors.csv")
            with open(actors_csv_path, "r") as f:
                actors = [line.strip() for line in f if line.strip()]
            alias_to_node = {alias: str(i + 1) for i, alias in enumerate(actors)}

            # Run community detection on original graph
            print(f"Running SCML community detection on original graph...")
            self.run_scml_community_detection(self.data_path, "original")
            original_com_path = self.find_community_file("original")

            # Load original communities and map alias to node ID
            df = pd.read_csv(original_com_path, header=None, names=["alias", "community"]) #here the layer information won't be displayed (same for every layer as always)
            grouped = defaultdict(set)
            for _, row in df.iterrows():
                node_id = alias_to_node.get(row["alias"])
                print(node_id,row["alias"] )
                if node_id:
                    grouped[row["community"]].add(node_id)
            
            original_communities = [list(nodes) for nodes in grouped.values()]
            
            

            # Flatten graph
            flat_graph = graph.get_flatten()
            

            # Build cdlib-compatible clustering and target,
            # this time the mapping was needed
            original_clustering = NodeClustering(communities=original_communities, graph=flat_graph, method_name="scml")
            
            original_communities = [set(c) for c in original_clustering.communities]
            
            #target_community=get_target_community(original_clustering) # random
            results = {}
            
            # Metrics before deception
            for idx, target_community in enumerate(original_clustering.communities):
                print(f"\n=== Targeting Community {idx} ===")
                
                modularity_before = nx.community.modularity(flat_graph, original_communities)
                safeness_before = Safeness.computeSafeness(flat_graph, target_community, original_communities)
                deception_before = Utils.getDeceptionScore(original_communities, target_community, flat_graph)
                print(f"[Before] Modularity: {modularity_before}, Safeness: {safeness_before}, Deception: {deception_before}")
                            
                k_path = os.path.join(os.path.dirname(self.data_path), "k.m")
                actors_path = os.path.join(os.path.dirname(self.data_path), "actors.csv")
                        
                
                
                
                
                # N runs
                for n in range(self.n_run):
                    print(f"RUN {n} ===")

                    
                    
                    
                    # RANDOM SECTION  --------------------------------------------------------------------------------------------
                    
                    random_flat_graph, edits = runRandomDeception(flat_graph, target_community,original_communities, self.budget)
                    print("random edits:  ", edits)
                    
                    random_multilayer_graph = GraphUtils.reconstruct_multilayer_from_flat(random_flat_graph, graph.multilayer_graph)

                    random_graph_path = os.path.join(self.modified_graphs_dir, f"{self.dataset_name}-scml-random.m")
                    random_multilayer_graph.to_m_file(random_graph_path)
                    
                    
                    # Copy k.m and actors.csv to same folder
                    for file_path in [k_path, actors_path]:
                        if os.path.isfile(file_path):
                            dest = os.path.join(self.modified_graphs_dir, os.path.basename(file_path))
                            subprocess.run(["cp", file_path, dest], check=True)
                        else:
                            print(f"Warning: {file_path} not found, skipping copy.")

                    print(f"Running SCML community detection on RANDOM graph...")
                    self.run_scml_community_detection(random_graph_path, "random")
                    random_com_path = self.find_community_file("random")

                    # Load random communities
                    df = pd.read_csv(random_com_path, header=None, names=["alias", "community"])
                    grouped = defaultdict(set)
                    for _, row in df.iterrows():
                        node_id = alias_to_node.get(row["alias"])
                        if node_id:
                            grouped[row["community"]].add(node_id)
                    random_communities = [set(nodes) for nodes in grouped.values()]

                    modularity_random = nx.community.modularity(random_flat_graph, random_communities)
                    safeness_random = Safeness.computeSafeness(random_flat_graph, target_community, random_communities)
                    deception_random = Utils.getDeceptionScore(random_communities, target_community, random_flat_graph)
                    

                    print(" RANDOM RESULTS:      --------------------------")
                    print(f"[Random] Modularity: {modularity_random}, Safeness: {safeness_random}, Deception: {deception_random}")
                    print(" END RANDOM RESULTS    ------------------------------")
                    
                    
                    # END RANDOM SECTION   --------------------------------------------------------------------------------------------
                    

                    # Apply MOD deception on multilayer graph
                    
                    modified_flat_graph_mod, edits = Utils.applyDeception(
                        'MOD', flat_graph, target_community, self.budget, original_communities
                    )
                    
                    print("normal edits", edits)

                    modified_multilayer_graph = GraphUtils.reconstruct_multilayer_from_flat(modified_flat_graph_mod, graph.multilayer_graph)
                    
                    # Save modified graph to .m file
                    modified_graph_path = os.path.join(self.modified_graphs_dir, f"{self.dataset_name}-scml.m")
                    modified_multilayer_graph.to_m_file(modified_graph_path)
                    
                    # Also copy k.m and actors.csv to the same folder (if they exist)
                    k_path = os.path.join(os.path.dirname(self.data_path), "k.m")
                    actors_path = os.path.join(os.path.dirname(self.data_path), "actors.csv")
                    
                    for file_path in [k_path, actors_path]:
                        if os.path.isfile(file_path):
                            dest = os.path.join(self.modified_graphs_dir, os.path.basename(file_path))
                            subprocess.run(["cp", file_path, dest], check=True)
                        else:
                            print(f"Warning: {file_path} not found, skipping copy.")
                            
                    # all files updated
                    
                    # Run community detection on modified graph
                    print(f"Running SCML community detection on modified graph...")
                    self.run_scml_community_detection(modified_graph_path, "modified")
                    modified_com_path = self.find_community_file("modified")

                    # Load modified communities and map alias to node ID
                    df = pd.read_csv(modified_com_path, header=None, names=["alias", "community"]) #here the layer information won't be displayed (same for every layer as always)
                    grouped = defaultdict(set)
                    for _, row in df.iterrows():
                        node_id = alias_to_node.get(row["alias"])
                        if node_id:
                            grouped[row["community"]].add(node_id)
                    modified_communities = [list(nodes) for nodes in grouped.values()]
                    
                    
                    modified_clustering = NodeClustering(communities=modified_communities, graph=modified_flat_graph_mod, method_name="scml")
                    modified_communities = [set(c) for c in modified_clustering.communities]

                    # Metrics after deception
                    modularity_after_mod = nx.community.modularity(modified_flat_graph_mod, modified_communities)
                    safeness_after_mod = Safeness.computeSafeness(modified_flat_graph_mod, target_community, modified_communities)
                    deception_after_mod= Utils.getDeceptionScore(modified_communities, target_community, modified_flat_graph_mod)
                    
                    print(f"[After] Modularity: {modularity_after_mod}, Safeness: {safeness_after_mod}, Deception: {deception_after_mod}")
                    
                    
                    #print( f"random Deception: {deception_random}, NOrmal Deception: {deception_after}")
                    
                    
                    
                    
                    
                    
                    # Apply SAFDEC deception on multilayer graph
                    
                    modified_flat_graph_saf, edits = Utils.applyDeception(
                        'SAFDEC', flat_graph, target_community, self.budget, original_communities
                    )
                    
                    print("normal edits", edits)

                    modified_multilayer_graph = GraphUtils.reconstruct_multilayer_from_flat(modified_flat_graph_saf, graph.multilayer_graph)
                    
                    # Save modified graph to .m file
                    modified_graph_path = os.path.join(self.modified_graphs_dir, f"{self.dataset_name}-scml.m")
                    modified_multilayer_graph.to_m_file(modified_graph_path)
                    
                    # Also copy k.m and actors.csv to the same folder (if they exist)
                    k_path = os.path.join(os.path.dirname(self.data_path), "k.m")
                    actors_path = os.path.join(os.path.dirname(self.data_path), "actors.csv")
                    
                    for file_path in [k_path, actors_path]:
                        if os.path.isfile(file_path):
                            dest = os.path.join(self.modified_graphs_dir, os.path.basename(file_path))
                            subprocess.run(["cp", file_path, dest], check=True)
                        else:
                            print(f"Warning: {file_path} not found, skipping copy.")
                            
                    # all files updated
                    
                    # Run community detection on modified graph
                    print(f"Running SCML community detection on modified graph...")
                    self.run_scml_community_detection(modified_graph_path, "modified")
                    modified_com_path = self.find_community_file("modified")

                    # Load modified communities and map alias to node ID
                    df = pd.read_csv(modified_com_path, header=None, names=["alias", "community"]) #here the layer information won't be displayed (same for every layer as always)
                    grouped = defaultdict(set)
                    for _, row in df.iterrows():
                        node_id = alias_to_node.get(row["alias"])
                        if node_id:
                            grouped[row["community"]].add(node_id)
                    modified_communities = [list(nodes) for nodes in grouped.values()]
                    
                    
                    modified_clustering = NodeClustering(communities=modified_communities, graph=modified_flat_graph_mod, method_name="scml")
                    modified_communities = [set(c) for c in modified_clustering.communities]

                    # Metrics after deception
                    modularity_after_saf = nx.community.modularity(modified_flat_graph_saf, modified_communities)
                    safeness_after_saf = Safeness.computeSafeness(modified_flat_graph_saf, target_community, modified_communities)
                    deception_after_saf= Utils.getDeceptionScore(modified_communities, target_community, modified_flat_graph_saf)
                    
                    print(f"[After SAF] Modularity: {modularity_after_saf}, Safeness: {safeness_after_saf}, Deception: {deception_after_saf}")
                    

                    results[f"run_{n}_community_{idx}"] = {
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
                                
                            }

                
            
        
            return results
        
        
         
        
        elif self.comm_algo.lower().startswith("pmm_"):
            #  # --------------------------------PMM ALGORITHMS ---------------------------
            print(f"Loading Edgelist (for PMM) file as multilayer graph from {self.data_path}")
            # Load actors.csv mapping aliases to node IDs (1-based)
            actors_csv_path = os.path.join(os.path.dirname(self.data_path), "actors.csv")
            with open(actors_csv_path, "r") as f:
                actors = [line.strip() for line in f if line.strip()]
            alias_to_node = {alias: str(i + 1) for i, alias in enumerate(actors)}
            
            graph = MultiLayerGraph.from_pmm_edgelist(self.data_path, alias_to_node)
            
            
               # Extract method from algo name (e.g., pmm_h → h)
            method = self.comm_algo.lower().split("_")[-1]

            

            # Run community detection on original graph
            print(f"Running PMM community detection on original graph...")
            self.run_pmm_community_detection(self.data_path, "original",method)
            original_com_path = self.find_community_file("original")

            # Load original communities and map alias to node ID
            df = pd.read_csv(original_com_path, header=None, names=["alias", "community"]) #here the layer information won't be displayed (same for every layer as always)
            grouped = defaultdict(set)
            for _, row in df.iterrows():
                node_id = alias_to_node.get(row["alias"])
                #print(node_id,row["alias"] )
                if node_id:
                    grouped[row["community"]].add(node_id)
            
            original_communities = [list(nodes) for nodes in grouped.values()]
            
            

            # Flatten graph
            flat_graph = graph.get_flatten()
            

            # Build cdlib-compatible clustering and target,
            # this time the mapping was needed
            original_clustering = NodeClustering(communities=original_communities, graph=flat_graph, method_name="pmm")
            
            original_communities = [set(c) for c in original_clustering.communities]
            
            #target_community=get_target_community(original_clustering) # random
            results = {}
            
            # Metrics before deception
            for idx, target_community in enumerate(original_clustering.communities):
                print(f"\n=== Targeting Community {idx} ===")
                
                modularity_before = nx.community.modularity(flat_graph, original_communities)
                safeness_before = Safeness.computeSafeness(flat_graph, target_community, original_communities)
                deception_before = Utils.getDeceptionScore(original_communities, target_community, flat_graph)
                print(f"[Before] Modularity: {modularity_before}, Safeness: {safeness_before}, Deception: {deception_before}")
                            
                k_path = os.path.join(os.path.dirname(self.data_path), "k.m")
                actors_path = os.path.join(os.path.dirname(self.data_path), "actors.csv")
                        
                
                
                
                
                # N runs
                for n in range(self.n_run):
                    print(f"RUN {n} ===")

                    
                    
                    
                    # RANDOM SECTION  --------------------------------------------------------------------------------------------
                    
                    random_flat_graph, edits = runRandomDeception(flat_graph, target_community,original_communities, self.budget)
                    print("random edits:  ", edits)
                    
                    random_multilayer_graph = GraphUtils.reconstruct_multilayer_from_flat(random_flat_graph, graph.multilayer_graph)

                    random_graph_path = os.path.join(self.modified_graphs_dir, "edgelist-random.csv")
                    random_multilayer_graph.to_edgelist_csv(random_graph_path)
                    
                    
                    # Copy k.m and actors.csv to same folder
                    for file_path in [k_path, actors_path]:
                        if os.path.isfile(file_path):
                            dest = os.path.join(self.modified_graphs_dir, os.path.basename(file_path))
                            subprocess.run(["cp", file_path, dest], check=True)
                        else:
                            print(f"Warning: {file_path} not found, skipping copy.")

                    print(f"Running PMM community detection on RANDOM graph...")
                    self.run_pmm_community_detection(random_graph_path, "random",method)
                    random_com_path = self.find_community_file("random")

                    # Load random communities
                    df = pd.read_csv(random_com_path, header=None, names=["alias", "community"])
                    grouped = defaultdict(set)
                    for _, row in df.iterrows():
                        node_id = alias_to_node.get(row["alias"])
                        if node_id:
                            grouped[row["community"]].add(node_id)
                    random_communities = [set(nodes) for nodes in grouped.values()]
                    
                    

                    modularity_random = nx.community.modularity(random_flat_graph, random_communities)
                    safeness_random = Safeness.computeSafeness(random_flat_graph, target_community, random_communities)
                    deception_random = Utils.getDeceptionScore(random_communities, target_community, random_flat_graph)
                    

                    print(" RANDOM RESULTS:      --------------------------")
                    print(f"[Random] Modularity: {modularity_random}, Safeness: {safeness_random}, Deception: {deception_random}")
                    print(" END RANDOM RESULTS    ------------------------------")
                    
                    
                    # END RANDOM SECTION   --------------------------------------------------------------------------------------------
                    

                    # Apply MOD deception on multilayer graph
                    print( " I am Continuing")
                    
                    modified_flat_graph_mod, edits = Utils.applyDeception(
                        'MOD', flat_graph, target_community, self.budget, original_communities
                    )
                    
                    print("normal edits", edits)

                    modified_multilayer_graph = GraphUtils.reconstruct_multilayer_from_flat(modified_flat_graph_mod, graph.multilayer_graph)
                    
                    # Save modified graph to .m file
                    modified_graph_path = os.path.join(self.modified_graphs_dir, "edgelist.csv")
                    modified_multilayer_graph.to_edgelist_csv(modified_graph_path)
                    
                    # Also copy k.m and actors.csv to the same folder (if they exist)
                    k_path = os.path.join(os.path.dirname(self.data_path), "k.m")
                    actors_path = os.path.join(os.path.dirname(self.data_path), "actors.csv")
                    
                    for file_path in [k_path, actors_path]:
                        if os.path.isfile(file_path):
                            dest = os.path.join(self.modified_graphs_dir, os.path.basename(file_path))
                            subprocess.run(["cp", file_path, dest], check=True)
                        else:
                            print(f"Warning: {file_path} not found, skipping copy.")
                            
                    # all files updated
                    
                    # Run community detection on modified graph
                    print(f"Running PMM community detection on modified graph...")
                    self.run_pmm_community_detection(modified_graph_path, "modified",method)
                    modified_com_path = self.find_community_file("modified")

                    # Load modified communities and map alias to node ID
                    df = pd.read_csv(modified_com_path, header=None, names=["alias", "community"]) #here the layer information won't be displayed (same for every layer as always)
                    grouped = defaultdict(set)
                    for _, row in df.iterrows():
                        node_id = alias_to_node.get(row["alias"])
                        if node_id:
                            grouped[row["community"]].add(node_id)
                    modified_communities = [list(nodes) for nodes in grouped.values()]
                    
                    
                    modified_clustering = NodeClustering(communities=modified_communities, graph=modified_flat_graph_mod, method_name="pmm")
                    modified_communities = [set(c) for c in modified_clustering.communities]
                    
                    
                     
                    # Metrics after deception
                    modularity_after_mod = nx.community.modularity(modified_flat_graph_mod, modified_communities)
                    safeness_after_mod = Safeness.computeSafeness(modified_flat_graph_mod, target_community, modified_communities)
                    deception_after_mod= Utils.getDeceptionScore(modified_communities, target_community, modified_flat_graph_mod)
                    
                    print(f"[After] Modularity: {modularity_after_mod}, Safeness: {safeness_after_mod}, Deception: {deception_after_mod}")
                    
                    
                    #print( f"random Deception: {deception_random}, NOrmal Deception: {deception_after}")
                    
                    
                    
                    
                    
                    
                    # Apply SAFDEC deception on multilayer graph
                    
                    modified_flat_graph_saf, edits = Utils.applyDeception(
                        'SAFDEC', flat_graph, target_community, self.budget, original_communities
                    )
                    
                    print("normal edits", edits)

                    modified_multilayer_graph = GraphUtils.reconstruct_multilayer_from_flat(modified_flat_graph_saf, graph.multilayer_graph)
                    
                    # Save modified graph to .m file
                    modified_graph_path = os.path.join(self.modified_graphs_dir,"edgelist.csv")
                    modified_multilayer_graph.to_edgelist_csv(modified_graph_path)
                    
                    # Also copy k.m and actors.csv to the same folder (if they exist)
                    k_path = os.path.join(os.path.dirname(self.data_path), "k.m")
                    actors_path = os.path.join(os.path.dirname(self.data_path), "actors.csv")
                    
                    for file_path in [k_path, actors_path]:
                        if os.path.isfile(file_path):
                            dest = os.path.join(self.modified_graphs_dir, os.path.basename(file_path))
                            subprocess.run(["cp", file_path, dest], check=True)
                        else:
                            print(f"Warning: {file_path} not found, skipping copy.")
                            
                    # all files updated
                    
                    # Run community detection on modified graph
                    print(f"Running PMM community detection on modified graph...")
                    self.run_pmm_community_detection(modified_graph_path, "modified",method)
                    modified_com_path = self.find_community_file("modified")

                    # Load modified communities and map alias to node ID
                    df = pd.read_csv(modified_com_path, header=None, names=["alias", "community"]) #here the layer information won't be displayed (same for every layer as always)
                    grouped = defaultdict(set)
                    for _, row in df.iterrows():
                        node_id = alias_to_node.get(row["alias"])
                        if node_id:
                            grouped[row["community"]].add(node_id)
                    modified_communities = [list(nodes) for nodes in grouped.values()]
                    
                    
                    modified_clustering = NodeClustering(communities=modified_communities, graph=modified_flat_graph_mod, method_name="pmm")
                    modified_communities = [set(c) for c in modified_clustering.communities]
                    

                    # Metrics after deception
                    modularity_after_saf = nx.community.modularity(modified_flat_graph_saf, modified_communities)
                    safeness_after_saf = Safeness.computeSafeness(modified_flat_graph_saf, target_community, modified_communities)
                    deception_after_saf= Utils.getDeceptionScore(modified_communities, target_community, modified_flat_graph_saf)
                    
                    print(f"[After SAF] Modularity: {modularity_after_saf}, Safeness: {safeness_after_saf}, Deception: {deception_after_saf}")
                    

                    results[f"run_{n}_community_{idx}"] = {
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
                                
                            }

                
            
        
            return results
        
        
        else:
            
             # --------------------------------MPX ALGORITHMS ---------------------------
            
            graph = MultiLayerGraph.from_file(os.path.dirname(self.data_path), os.path.basename(self.data_path))

            # Step 2: Run community detection on original graph
            print(f"Running initial community detection via script...")
            self.run_community_detection_script(self.data_path, "original")
            original_com_path = self.find_community_file("original")

            # Step 3: Flatten the graph
            flat_graph = graph.get_flatten()

            # Step 4: Load original communities
            original_communities = load_communities_cdlib(original_com_path, flat_graph)
            #target_community=get_target_community(original_communities) # random
            #print("Target Community to Hide: ", target_community)
            
            
            
            
            original_communities = [set(c) for c in original_communities.communities]
            
            
            
            
            


            
            
            results = {}
            for idx, target_community in enumerate(original_communities):
                print(f"\n=== Targeting Community {idx} ===")
                
                modularity_before = nx.community.modularity(flat_graph, original_communities)
                safeness_before = Safeness.computeSafeness(flat_graph, target_community, original_communities)
                deception_before = Utils.getDeceptionScore(original_communities, target_community, flat_graph)
                print(f"[Before] Modularity: {modularity_before}, Safeness: {safeness_before}, Deception: {deception_before}")
            
                # Loop through each community as the target
                for n in range(self.n_run):
                    print(f"\n=== RUN {n} ===")
                    
                    
                    # RANDOM SECTION  --------------------------------------------------------------------------------------------
                    
                    random_flat_graph, edits = runRandomDeception(flat_graph, target_community,original_communities, self.budget)
                    print("random edits:  ", edits)
                    
                    random_multilayer_graph = GraphUtils.reconstruct_multilayer_from_flat(random_flat_graph, graph.multilayer_graph)

                    random_graph_path = os.path.join(self.modified_graphs_dir, f"{self.dataset_name}-random.mpx")
                    random_multilayer_graph.to_mpx(random_graph_path)

                    # Run community detection on RANDOM graph
                    print(f"Running community detection on RANDOM graph...")
                    self.run_community_detection_script(random_graph_path, "random")
                    random_com_path = self.find_community_file("random")

                    # Load random communities
                    random_communities = load_communities_cdlib(random_com_path, random_flat_graph)
                    random_communities = [set(c) for c in random_communities.communities]

                    # Evaluate metrics on RANDOM graph
                    modularity_random = nx.community.modularity(random_flat_graph, random_communities)
                    safeness_random = Safeness.computeSafeness(random_flat_graph, target_community, random_communities)
                    deception_random = Utils.getDeceptionScore(random_communities, target_community, random_flat_graph)
                    print(" RANDOM RESULTS:      --------------------------")
                    print(f"[Random] Modularity: {modularity_random}, Safeness: {safeness_random}, Deception: {deception_random}")
                    print(" END RANDOM RESULTS    ------------------------------")
                    
                    # END RANDOM SECTION   --------------------------------------------------------------------------------------------
                    
                    
                    # Apply MOD deception on multilayer graph
                    modified_flat_graph_mod, edits= Utils.applyDeception(
                        "MOD", flat_graph, target_community, self.budget, original_communities
                    )

                    
                    print("normal edits", edits)
                    

                    # Step 8: Reconstruct multilayer graph from flat graph
                    modified_multilayer_graph = GraphUtils.reconstruct_multilayer_from_flat(modified_flat_graph_mod, graph.multilayer_graph)

                    # Step 9: Save the reconstructed multilayer graph to file
                    modified_graph_path = os.path.join(self.modified_graphs_dir, f"{self.dataset_name}.mpx")
                    modified_multilayer_graph.to_mpx(modified_graph_path)

                    # Step 10: Run community detection on modified graph
                    print(f"Running community detection on modified graph via script...")
                    self.run_community_detection_script(modified_graph_path, "modified")
                    modified_com_path = self.find_community_file("modified")

                    # Step 11: Load modified communities
                    modified_communities = load_communities_cdlib(modified_com_path,modified_flat_graph_mod)
                    modified_communities = [set(c) for c in modified_communities.communities]
                    
                    
                    # Step 12: Metrics after deception
                    modularity_after_mod =  nx.community.modularity(modified_flat_graph_mod, modified_communities)
                    safeness_after_mod = Safeness.computeSafeness(modified_flat_graph_mod, target_community, modified_communities)
                    deception_after_mod = Utils.getDeceptionScore(modified_communities,target_community , modified_flat_graph_mod)
                    #print( f"random Deception: {deception_random}, NOrmal Deception: {deception_after}")
                    
                    
                    print(f"[After] Modularity: {modularity_after_mod}, Safeness: {safeness_after_mod}, Deception: {deception_after_mod}")
                    
                    
                    
                    
                    
                    
                    # Apply SAF deception on multilayer graph
                    modified_flat_graph_saf, edits= Utils.applyDeception(
                        "SAFDEC", flat_graph, target_community, self.budget, original_communities
                    )

                    
                    print("normal edits", edits)
                    

                    # Step 8: Reconstruct multilayer graph from flat graph
                    modified_multilayer_graph = GraphUtils.reconstruct_multilayer_from_flat(modified_flat_graph_saf, graph.multilayer_graph)

                    # Step 9: Save the reconstructed multilayer graph to file
                    modified_graph_path = os.path.join(self.modified_graphs_dir, f"{self.dataset_name}.mpx")
                    modified_multilayer_graph.to_mpx(modified_graph_path)

                    # Step 10: Run community detection on modified graph
                    print(f"Running community detection on modified graph via script...")
                    self.run_community_detection_script(modified_graph_path, "modified")
                    modified_com_path = self.find_community_file("modified")

                    # Step 11: Load modified communities
                    modified_communities = load_communities_cdlib(modified_com_path,modified_flat_graph_saf)
                    modified_communities = [set(c) for c in modified_communities.communities]
                    
                    
                    # Step 12: Metrics after deception
                    modularity_after_saf =  nx.community.modularity(modified_flat_graph_saf, modified_communities)
                    safeness_after_saf = Safeness.computeSafeness(modified_flat_graph_saf, target_community, modified_communities)
                    deception_after_saf = Utils.getDeceptionScore(modified_communities,target_community , modified_flat_graph_saf)
                    
                    print(f"[After SAF] Modularity: {modularity_after_saf}, Safeness: {safeness_after_saf}, Deception: {deception_after_saf}")

                    results[f"run_{n}_community_{idx}"] = {
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
                                    
                                }
                
            return results
            
            


import argparse
import re
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Run the full deception evaluation pipeline.")
    parser.add_argument("data_path", help="Path to the input .mpx file (original graph)")
    parser.add_argument("deception_algo", help="Deception algorithm to use (e.g., SAFDEC, MOD)")
    parser.add_argument("budget", type=int, help="Budget for deception algorithm")
    parser.add_argument("--n_run", type=int, help="Number of runs")
    parser.add_argument("--comm_algo", default="flat_nw", help="Community detection algorithm (default: flat_nw)")

    args = parser.parse_args()

    print("=== Deception Evaluation Pipeline ===")
    print(f"Data Path      : {args.data_path}")
    print(f"Deception Algo: {args.deception_algo}")
    print(f"Budget        : {args.budget}")
    print(f"N_Runs       : {args.n_run}")
    print(f"Comm Algo     : {args.comm_algo}")
    print("=====================================")

    evaluator = DeceptionEvaluator(
        data_path=args.data_path,
        deception_algo=args.deception_algo,
        budget=args.budget,
        n_run=args.n_run,
        comm_algo=args.comm_algo
    )

    metrics = evaluator.evaluate()

    


    
    
    
    #  OVERVIEW RESULTS ========
    
    
    
    
    
    
    
    
    
    dataset_name=""
    if (args.comm_algo == "scml" and args.data_path.endswith(".m")) or args.comm_algo.startswith("pmm_"):
            # For SCML, use the parent folder name of the .m file
            dataset_name = os.path.basename(os.path.dirname(args.data_path))
        
    else:
            # Default: use filename without extension
            dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]
    print("METRICS",metrics)
   
   
   
    
    
    output_filename = f"./results/{dataset_name}/deception_{dataset_name}_{args.comm_algo}_budget{args.budget}_Flat-DECEPTION.txt"
    
    # Extract the directory path from the filename
    output_dir = os.path.dirname(output_filename)

    # if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    
    with open(output_filename, "w") as f:
        f.write("--- Deception Evaluation Metadata ---\n")
        f.write(f"Dataset Name        : {dataset_name}\n")
        f.write(f"Deception Algorithm : {args.deception_algo}\n")
        f.write(f"Community Algorithm : {args.comm_algo}\n")
        f.write(f"Budget              : {args.budget}\n")
        f.write(f"Number of Runs      : {args.n_run}\n")
        f.write("-------------------------------------\n\n")
        
        f.write("\n===== Aggregated Per-Community Statistics =====\n")



        per_community_data = defaultdict(list)  # {community_idx: [dict of metrics per run]}
        
        community_deception_means = defaultdict(dict)

        for run_key, data in metrics.items():
            comm_idx = data["target_community_index"]
            per_community_data[comm_idx].append(data)

        # Now write details for each community
        for comm_idx in sorted(per_community_data):
            f.write(f"\n>> Community Index: {comm_idx}\n")

            community_runs = per_community_data[comm_idx]

            # Print all individual runs
            for i, run_data in enumerate(community_runs):
                f.write(f"\n  - Run {i + 1}:\n")
                for metric, value in run_data.items():
                    # if metric in ["target_community_index", "target_community_nodes"]:
                    #     continue
                    if isinstance(value, float):
                        f.write(f"    {metric}: {value:.4f}\n")
                    else:
                        f.write(f"    {metric}: {value}\n")

            # Collect per-metric values for aggregation
            aggregated_metrics = defaultdict(list)
            for run_data in community_runs:
                for metric, value in run_data.items():
                    if metric in ["target_community_index", "target_community_nodes", "modularity_before","safeness_before","safeness_random","safeness_after_mod","safeness_after_saf","deception_before"]:
                        continue
                    aggregated_metrics[metric].append(value)
                    
             

            # Print aggregate statistics
            f.write("\n  >>> Aggregated Statistics:\n")
            for metric, values in aggregated_metrics.items():
                
                min_val = min(values)
                max_val = max(values)
                mean_val = sum(values) / len(values)
                f.write(f"    {metric} -> Min: {min_val:.4f}, Max: {max_val:.4f}, Mean: {mean_val:.4f}\n")
                # Track deception means per community
                if metric.startswith("deception_"):
                    community_deception_means[comm_idx][metric] = mean_val
                
        f.write("\n\n===== Global Deception Summary (per method) =====\n")

        deception_methods = ["deception_random", "deception_after_mod", "deception_after_saf"]
        deception_summary = {method: [] for method in deception_methods}  # {method: list of (comm_idx, mean_val)}

        for comm_idx in sorted(per_community_data):
            community_runs = per_community_data[comm_idx]

            # Recompute aggregated_metrics for current community
            aggregated_metrics = defaultdict(list)
            for run_data in community_runs:
                for metric, value in run_data.items():
                    if metric in ["target_community_index", "target_community_nodes", "modularity_before", "safeness_before", "safeness_random", "safeness_after_mod", "safeness_after_saf", "deception_before"]:
                        continue
                    aggregated_metrics[metric].append(value)

            for method in deception_methods:
                if method in aggregated_metrics:
                    mean_val = sum(aggregated_metrics[method]) / len(aggregated_metrics[method])
                    deception_summary[method].append((comm_idx, mean_val))

        for method, comm_vals in deception_summary.items():
            min_comm, min_val = min(comm_vals, key=lambda x: x[1])
            max_comm, max_val = max(comm_vals, key=lambda x: x[1])
            mean_of_means = sum(val for _, val in comm_vals) / len(comm_vals)

            f.write(f"\n>> Method: {method}\n")
            f.write(f"   Community with MIN deception: {min_comm} ({min_val:.4f})\n")
            f.write(f"   Community with MAX deception: {max_comm} ({max_val:.4f})\n")
            f.write(f"   Mean deception across communities: {mean_of_means:.4f}\n")
        # ===== DELTA DECEPTION (safeness - random) and (mod - random) =====
        f.write("\n\n===== Delta Deception Summary (vs Random) =====\n")

        delta_safeness = []  # list of (comm_idx, delta)
        delta_mod = []

        for comm_idx, values in community_deception_means.items():
            dr = values.get("deception_random", 0)
            ds = values.get("deception_after_saf", 0)
            dm = values.get("deception_after_mod", 0)

            delta_safeness.append((comm_idx, ds - dr))
            delta_mod.append((comm_idx, dm - dr))

        def write_delta_summary(name, deltas):
            min_c, min_v = min(deltas, key=lambda x: x[1])
            max_c, max_v = max(deltas, key=lambda x: x[1])
            mean_v = sum(v for _, v in deltas) / len(deltas)

            f.write(f"\n>> {name}\n")
            f.write(f"   Community with MIN delta: {min_c} ({min_v:.4f})\n")
            f.write(f"   Community with MAX delta: {max_c} ({max_v:.4f})\n")
            f.write(f"   Mean delta across communities: {mean_v:.4f}\n")

        write_delta_summary("Deception After SAF - Deception Random", delta_safeness)
        write_delta_summary("Deception After MOD - Deception Random", delta_mod)





if __name__ == "__main__":
    main()
