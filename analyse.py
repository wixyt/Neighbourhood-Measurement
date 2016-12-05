#!/usr/bin/env python
import sys
import argparse
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
import json
import os
from load import wrapper



def read_json_file(path):
    with open(path, 'rb') as f:
        js_graph = json.load(f)
        try:
            graph = json_graph.node_link_graph(js_graph)
        except NetworkXError as e:
            print "%s" % e
    return graph


def suprise_metric(link, i_degree, j_degree, edges):
    return link - (i_degree*j_degree)/(2.0*edges)

def adjancency_array(graph, node_list):
    return nx.adj_matrix(graph, node_list).toarray()

def hadamard_product(a, b): 
    # practically an XOR on arrays with units of 0 and 1
    return np.multiply(a, b)

def internal_consistency(graph):
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
            x_i = np.array(graph.node[node_list[i]]['attributes'])
            x_j = np.array(graph.node[node_list[j]]['attributes'])
            elewise_product = hadamard_product(x_i, x_j)
            total = np.multiply(elewise_product, suprise_index)
            
            internal_consistency += np.sum(total)
    
    return internal_consistency

def boundary_edges(G):
    E_list = []
    for edge in G.edges():
        if G.node[edge[0]]['subgraph'] != G.node[edge[1]]['subgraph']:
            E_list.append(edge)
    return E_list
    
def unsuprising_metric(k_i, k_b, edges):
    return (1 - min(1, (k_i*k_b)/(2.0*edges)))

def external_separability(G, E):
    external_separability = 0.0 
    # ideally it should be low for high quality neighbourhood

    for edge in E:
        # AWKWARD way of ensuring the interior node's degree is the first parameter
        # TODO: make it not so awkward
        if G.node[edge[0]]['subgraph'] == 'C':
            unsup_metric = unsuprising_metric(
                G.degree(edge[0]), 
                G.degree(edge[1]), 
                len(G.edges())
            )
            x_i = np.array(G.node[edge[0]]['attributes'])
            x_b = np.array(G.node[edge[1]]['attributes'])
        else:
            unsup_metric = unsuprising_metric(G.degree(edge[1]), G.degree(edge[0]), len(G.edges()))
            x_i = np.array(G.node[edge[1]]['attributes'])
            x_b = np.array(G.node[edge[0]]['attributes'])
        
        elewise_product = hadamard_product(x_i, x_b)
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


def silly_cluster(graph):
    # assign silly way of getting two subgraphs (useing betweenness_centrality)
    betweeness_dict = nx.betweenness_centrality(graph)
    max_val = max(betweeness_dict.values())
    lim = np.average(betweeness_dict.values())/max_val

    for key in betweeness_dict.keys():
        if betweeness_dict[key]/max_val > lim:
            graph.node[key]['subgraph'] = 'C'
        else:
            graph.node[key]['subgraph'] = 'B'

def subgraph_separate(graph):
    C_list = []
    B_list = []
    for node in graph.nodes():
        if graph.node[node]['subgraph'] == 'C':
            C_list.append(node)
        else:
            B_list.append(node)
    C = graph.subgraph(C_list) 

    return C

def cluster_and_subgraph(graph):
    cluster_by_degree(graph)
    # silly_cluster(graph)
    C = subgraph_separate(graph)
    
    return C

def calculate_normality(C, graph):
    I = internal_consistency(C)
    print("Internal Consistency: %f"  % I)
    E = external_separability(graph, boundary_edges(graph))
    print("External Separability: %f"  % E)
    # Calculat Normality
    N = I - E
    print("Normality: %f" % N)
    
def operations(graph):
    I = cluster_and_subgraph(graph)
    calculate_normality(I, graph)

def main(args):
    # TODO: add argument parser
    # ARGUMENTS 
    # -[l|c] data_directory {--directed} {--pickle|--json}
    # -l load from graph file like JSON or PICKLE
    # -c Create graph from data files (edges, features, etc) 
    # --directed 

    if len(args) < 3:
        print("invalid argumanfoi oa EXITING")
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
        operations(graph)
    
        
    else:
        print("No valid arguments given")

    
if __name__ == "__main__":
    main(sys.argv)