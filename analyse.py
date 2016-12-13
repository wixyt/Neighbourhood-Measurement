#!/usr/bin/env python
import sys
import argparse
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import json
import os
from sklearn.decomposition import PCA
from networkx.readwrite import json_graph

class TestingError(Exception):
    """
    Errors for specific problems occuring with the calculation of normality of use of graphs
    TODO change name 
    """

class Normality(object):
    """
        Class for calculating normality of a given subset of nodes and associated edge nodes    
    """
    def __init__(self):
        """
            params: 
            total-graph: surrounding graph structure set
            subgraph: subset of total graph
            target: used for evaluation of normality of adding additional node to the subset
        """
        self.total_graph = None
        self.subgraph = None # node list
        self.target_node = int


    def calculate(self, total_graph, subgraph, target=None):
        # Ensure total_graph and subgraph is of the right type
        assert (type(total_graph) == type(nx.Graph()) or type(total_graph) == (nx.DiGraph())), "Total Graph not type of Nx graph"
        assert type(subgraph) == type(set()), "Subgraph not of type list"
        self.total_graph = total_graph
        self.subgraph = total_graph.subgraph(subgraph)
        
        # Mark subgraph on total graph 
        self.mark_subgraph()
        # find edge nodes of subgraph
        self.edge_nodes = self.boundary_edges()
        
        I = self.internal_consistency(self.subgraph)
        E = self.external_separability(self.total_graph, self.edge_nodes)
        N = I - E
        return N


    def mark_subgraph(self):
        graph_nodes = set(self.total_graph.nodes())
        subg_nodes = set(self.subgraph.nodes())
        for node in graph_nodes:
            if node in subg_nodes:
                self.total_graph.node[node]['subgraph'] = 1 # ? make this part of subgraph count
            else:
                self.total_graph.node[node]['subgraph'] = 0
        
    
    def boundary_edges(self):
        E_list = []
        for edge in self.total_graph.edges():
            if self.total_graph.node[edge[0]]['subgraph'] != self.total_graph.node[edge[1]]['subgraph']:
                E_list.append(edge)
        return E_list

    
    def internal_consistency(self, G, w_i=None):
        node_list = G.nodes()
        internal_consistency = 0.0
        adj_m = nx.adj_matrix(G, node_list).toarray()
        length = (adj_m.shape)[1]
        total_edges = G.size()
        
        for i in range(length):
            for j in range(length):
                # get index for the "suprise value" between two nodes
                suprise_index = adj_m[i][j] - \
                (G.degree(node_list[i])*G.degree(node_list[j]))/(2.0*total_edges)

                x_i = np.array(G.node[node_list[i]]['feature_vector'])
                x_j = np.array(G.node[node_list[j]]['feature_vector'])
                # x_i = np.array(graph.node[node_list[i]]['decomp_features'])
                # x_j = np.array(graph.node[node_list[j]]['decomp_features'])
            
                elewise_product = np.multiply(x_i, x_j)
                if w_i is not None:
                    elewise_product = np.multiply(elewise_product, w_i)
                total = np.multiply(elewise_product, suprise_index)
                internal_consistency += np.sum(total)
        
        return internal_consistency

    def external_separability(self, G, E, w_e=None):
        external_separability = 0.0 
        # ideally it should be low for high quality neighbourhood

        for edge in E:
            unsup_metric = unsuprising_metric(
                G.degree(edge[0]), 
                G.degree(edge[1]),
                G.size()
            )
            
            x_i = np.array(G.node[edge[0]]['feature_vector'])
            x_b = np.array(G.node[edge[1]]['feature_vector'])
            # x_i = np.array(G.node[edge[0]]['decomp_features'])
            # x_b = np.array(G.node[edge[1]]['decomp_features'])
            
            elewise_product = np.multiply(x_i, x_b)
            if w_e is not None:
                elewise_product = np.multiply(elewise_product, w_e)
            total = np.multiply(elewise_product, unsup_metric)
            external_separability += np.sum(total)

        return external_separability



def suprise_metric(link, i_degree, j_degree, edges):
    return link - (i_degree*j_degree)/(2.0*edges)

def adjancency_array(graph, node_list):
    return nx.adj_matrix(graph, node_list).toarray()

def hadamard_product(a, b): 
    # practically an XOR on arrays with units of 0 and 1
    return np.multiply(a, b)

def internal_consistency(graph, w_i=None):
    node_list = graph.nodes()
    internal_consistency = 0.0
    adj_m = adjancency_array(graph, node_list)
    length = (adj_m.shape)[1]
    total_edges = graph.size()
    
    
    for i in range(length):
        for j in range(length):
            # get index for the "suprise value" between two nodes
            suprise_index = suprise_metric(
                adj_m[i][j],
                graph.degree(node_list[i]),
                graph.degree(node_list[j]),
                total_edges
            )

            # element-wise product of attribute vectors
            # x_i = np.array(graph.node[node_list[i]]['feature_vector'])
            # x_j = np.array(graph.node[node_list[j]]['feature_vector'])
            x_i = np.array(graph.node[node_list[i]]['decomp_features'])
            x_j = np.array(graph.node[node_list[j]]['decomp_features'])
        
            elewise_product = hadamard_product(x_i, x_j)
            if w_i is not None:
                elewise_product = np.multiply(elewise_product, w_i)
            total = np.multiply(elewise_product, suprise_index)
            
            internal_consistency += np.sum(total)
    
    return internal_consistency

def boundary_edges(G):
    E_list = []
    for edge in G.edges():
        if G.node[edge[0]]['subgraphs'] != G.node[edge[1]]['subgraphs']:
            E_list.append(edge)
    return E_list
    
def unsuprising_metric(k_i, k_b, edges):
    return (1 - min(1, (k_i*k_b)/(2.0*edges)))

def external_separability(G, E, w_e=None):
    external_separability = 0.0 
    # ideally it should be low for high quality neighbourhood

    for edge in E:
        unsup_metric = unsuprising_metric(
            G.degree(edge[0]), 
            G.degree(edge[1]), 
            len(G.edges())
        )
        # x_i = np.array(G.node[edge[0]]['feature_vector'])
        # x_b = np.array(G.node[edge[1]]['feature_vector'])
        x_i = np.array(G.node[edge[0]]['decomp_features'])
        x_b = np.array(G.node[edge[1]]['decomp_features'])
        
        elewise_product = hadamard_product(x_i, x_b)
        if w_e is not None:
            elewise_product = np.multiply(elewise_product, w_e)
        total = np.multiply(elewise_product, unsup_metric)
        
        external_separability += np.sum(total)


    return external_separability

def cluster_by_degree(graph): 
    # VERY BAD WAY OF CLUSTERING
    total_degree = 0
    nodes = 0
    for node in graph.nodes():
        nodes += 1
        total_degree += graph.degree(node)
    avg_degree = total_degree/nodes

    for node in graph.nodes():
        if graph.degree(node) >= avg_degree:
            graph.node[node]['subgraph'] = 'C'
        else:
            graph.node[node]['subgraph'] = 'B'


def subgraph_separate(graph):
    subgraphs = []
    count = 0
    remaining_subgraphs = True 
    while remaining_subgraphs:
        subgraph_nodes = []
        for node in graph.nodes():
            if 'subgraphs' in graph.node[node]:
                if count in graph.node[node]['subgraphs']:
                    subgraph_nodes.append(node)
        count += 1
        if len(subgraph_nodes) == 0:
            break
        subgraphs.append(graph.subgraph(subgraph_nodes))
        del subgraph_nodes
    return subgraphs


def calculate_normality(C, graph):
    I = internal_consistency(C)
    print("Internal Consistency: %f"  % I)
    E = external_separability(graph, boundary_edges(graph))
    print("External Separability: %f"  % E)
    # Calculate Normality
    N = I - E
    print("Normality: %f" % N)
    print("Optizmizing weight vector...")
    objective_optimization(graph, C)
    return N

def calculate_imin(C, adj_m):
    node_list = C.nodes()
    minimum = 0.0
    for i in range(len(adj_m.toarray())):
        for j in range(len(adj_m.toarray())):
            minimum += -(C.degree(node_list[i])*C.degree(node_list[i]))/(2.0*len(C.edges()))
    return minimum

def optimize(w_v, C, G, i_max): # taking out one weight vector for simplisiticity
    # TODO: make this parameters more efficient
    # x_i, x_i = weight_vectors
    return internal_consistency(C, w_v[0]) - external_separability(G, boundary_edges(G), w_v[1])

def objective_optimization(graph, C):
    adj_m = nx.adjacency_matrix(C)
    length = sum([len(graph.node[x]['decomp_features']) for x in graph.nodes()])/len(graph.nodes())
    I_max = float(len(adj_m.toarray())**2)
    I_min = calculate_imin(C, adj_m)
    x_i = np.ones(length)
    x_e = np.ones(length)
    # weight vector components are normalised between 0 and 1
    bnds = tuple((0, 1) for x in x_i)
    
    
    # res = sp.optimize.minimize(fun=optimize, method='BFGS', jac=True, args=(x_i, C, graph), options={"maxiter": 5000}, bounds=bounds)
    res = sp.optimize.minimize(
        optimize, # function 
        [x_i, x_e], # weight vector 
        args=(C, graph, I_max, ), # other parameters to be passed in as arguments to the function
        method='L-BFGS-B',
        bounds=bnds, # bounds of the weight vector
        options={"maxiter": 30}
        )
    print "weight vector after optimisation: %s" % res.x
    print "results after optimisation of weight vector==="
    print "Normality: %f" % (optimize(res.x, C, graph, I_max))

def decomposition(graph):
    # using PCA decompose the feature vectors into a smaller feature matrix 
    pca = PCA(n_components=4)
    feature_matrix = []
    for node in graph.nodes():
        feature_matrix.append(graph.node[node]['feature_vector'])
    sk_transf = pca.fit_transform(np.array(feature_matrix))
    print sk_transf
    for i, node in enumerate(graph.nodes()):
        graph.node[node]['decomp_features'] = sk_transf[i]  


def operations(graph):
    decomposition(graph)
    subgraphs = subgraph_separate(graph)
    count = 1
    for subgraph in subgraphs:
        print "subgraph: %d" % count 
        calculate_normality(subgraph, graph)
        count += 1

def main(args):
    # TODO: add argument parser
    # ARGUMENTS 
    # -[l|c] data_directory {--directed} {--pickle|--json}
    # -l load from graph file like JSON or PICKLE
    # -c Create graph from data files (edges, features, etc) 
    # --directed 

    if len(args) < 4:
        print("""Invalid arguments EXITING \n ARGUMENTS 
    # -[l|c] data_directory {--directed} {--pickle|--json}
    # -l load from graph file like JSON or PICKLE
    # -c Create graph from data files (edges, features, etc) 
    # --directed """
    )
        sys.exit()
    _dir = os.path.dirname(os.path.realpath(__file__))

    if args[1] == '-l': #TODO: get loading from file to work
        graph_file = os.path.join(_dir + '/' + args[2])
        if args[3] == '--pickle':
            graph = nx.read_gpickle(graph_file)
        elif args[3] == '--json':
            graph = read_json_file(graph_file)
            
    
    elif args[1] == '-c':
        if args[3] == '--directed':
            graph = wrapper(args[2], "directed")
        else:
            graph = wrapper(args[2])
    
    
        
    else:
        print("No valid arguments given")
    print "Graph size: %d" % graph.size()
    operations(graph)
    
if __name__ == "__main__":
    main(sys.argv)