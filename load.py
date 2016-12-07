#!/usr/bin/env python
import sys
import snap
import networkx as nx
import numpy as np
import os, os.path
import matplotlib.pyplot as plt
import glob
import math
import operator
from sklearn.feature_extraction import DictVectorizer
from networkx.readwrite import json_graph
import json


_dir = os.path.dirname(os.path.realpath(__file__))

def edge_list_from_file(path, delim="\t"):
    with open(path, 'rb') as f:
        reader = csv.reader(f, delimiter=delim)
        d = list(reader)

    edges = []
    skip = re.compile('^#')
    for element in d:
        try:
            if skip.search(element[0]):
                print("Caught bad element: ", element[0])
                continue
            # pair = element[0].split(' ')
            # edges.append((int(pair[0]), int(pair[1])))
        except Exception as e:
            print e
    return edges

def load_edges(path, G):
    with open(path, 'r') as edge_file:
        for line in edge_file:
            edges = [int(x) for x in line.split(' ')]
            G.add_edge(edges[0], edges[1])

    return G


def graph_nx(edges):
    G = nx.Draph()
    G.add_edges_from(edges)
    return G

def draw_nx_graph(Graph):
    nx.draw(Graph)
    plt.show()

def load_attributes(feature_file_path):
    # create dictionary of features keys and values
    feature_index = {}
    with open(feature_file_path, 'r') as feature_file:
        for line in feature_file:
            feature = line.split(' ')
            feature_index[feature[0]] = feature[1][:-1]
    
    return feature_index

def apply_attribute_vector(feature_vector, node_id, G, feature_dict):
    key_array = feature_dict.keys()
    # G.node[node_id]['attributes'] = np.zeros(len(feature_dict))
    G.node[node_id]['attributes'] = [0 for x in range(len(key_array))]

    # print("key %d feature vector %s len feature vector %s" %(node, feature_vect, len(feature_vect)))
    for key in range(len(feature_dict)):
        G.node[node_id]['attributes'][key] = feature_vector[int(key)]
        # print( type(G.node[node_id]['attributes']))
        # G.node[node_id]['attributes'].tolist()

        
def apply_attributes(feature_vector, node_id, G, feature_dict):
    key_array = feature_dict.keys()
    # print feature_dict
    G.node[node_id]['named_attributes'] = {}
    # print("key %d feature vector %s len feature vector %s" %(node, feature_vect, len(feature_vect)))
    for key in range(len(feature_dict)):
        G.node[node_id]['named_attributes'][feature_dict[str(key)]] = feature_vector[int(key)] 


def load_nodes(edge_feat_path, feature_index, G):
    key_array = feature_index.keys()

    with open(edge_feat_path, 'r') as file:
        for line in file:
            try:
                features = [int(x) for x in line.split(' ')]
                node = features[0]
                feature_vect = features[1:]
                G.add_node(node)
                
                apply_attribute_vector(feature_vect, node, G, feature_index)
                apply_attributes(feature_vect, node, G, feature_index)
            except Exception as e:
                print "error: %s" % e
            

def load_node_features(node_feat_file, feature_index, G, node_id):
    # print feature_index.values()
    key_array = feature_index.keys()
    with open(node_feat_file, 'r') as file:
        vector_string = file.read()[:-1]
        node_feat_vect = [int(x) for x in vector_string.split(' ')]
        apply_attribute_vector(node_feat_vect, node_id, G, feature_index)
        apply_attributes(node_feat_vect, node_id, G, feature_index)

def named_attributes_to_vector(G):
    
    # extract all named node features into a dictionary containing all known features of the graph 
    # for each node produce a vector that is consistent with the size of the large feature dictionary
    # note: this will result in very large attribute vectors - but feature selection will reduce the size
    total = {}
    for node in G.nodes():
        for element in G.node[node]['named_attributes']:
            if element not in total:
                total[element] = 1
    sorted_dict = sorted(total.items(), key=operator.itemgetter(0))
    
    v = DictVectorizer(sparse=False)
    for node in G.nodes():
        
        X = v.fit_transform([total, G.node[node]['named_attributes']])
        G.node[node]['full_featvector'] = X[1].tolist() # numpy arrays cannot be stored properly
        

def save(path, graph):

    save_file = json_graph.node_link_data(graph)
    json.dump(save_file, open(path, 'w'), indent=2)

def main(args):
    if args[1] == '--create':
        
        graph = nx.DiGraph()
        if len(args) <= 2:
            print("Argument error: --load [data directory] \n")
            print("exiting...")
            sys.exit()

        data_dir = os.path.join(_dir + "/" + args[2])
            
        nodes = [int(x.split("/")[-1].split('.')[0]) for x in glob.iglob(data_dir + "/*.featnames")]
        
        for node in nodes:
            # get file paths
            feature_name_file = os.path.join(data_dir + "/%s.featnames" % node)
            edge_file = os.path.join(data_dir + "/%s.edges" % node)
            edge_feat_file = os.path.join(data_dir + "/%s.feat" % node)
            node_feat_file = os.path.join(data_dir + "/%s.egofeat" % node)
            
            # load nodes/attributes onto graph
            
            graph.add_node(node)
            attribute_dict = load_attributes(feature_name_file)
            load_node_features(node_feat_file, attribute_dict, graph, node)
            load_nodes(edge_feat_file, attribute_dict, graph)
            
            load_edges(edge_file, graph)

    # for node in graph.node.keys():
        # print("Key %s - %s " % (node, graph.node[node]['named_attributes']))
    print graph.size()
    for node in graph.nodes():
        print node

    # nx.write_gpickle(graph, args[3])
    save(args[3], graph)

def wrapper(path, G_type='uni'):
    if G_type == "directed":
        print ("Returning Directed Graph...")
        graph = nx.DiGraph()
    else:
        print ("Returning Undirected Graph...")
        graph = nx.Graph()


    data_dir = os.path.join(_dir + "/" + path)
        
    nodes = [int(x.split("/")[-1].split('.')[0]) for x in glob.iglob(data_dir + "/*.featnames")]


    for node in nodes:
        # get file paths
        feature_name_file = os.path.join(data_dir + "/%s.featnames" % node)
        edge_file = os.path.join(data_dir + "/%s.edges" % node)
        edge_feat_file = os.path.join(data_dir + "/%s.feat" % node)
        node_feat_file = os.path.join(data_dir + "/%s.egofeat" % node)
        
        # load nodes/attributes onto graph
        
        graph.add_node(node)
        attribute_dict = load_attributes(feature_name_file)
        load_node_features(node_feat_file, attribute_dict, graph, node)
        load_nodes(edge_feat_file, attribute_dict, graph)
        load_edges(edge_file, graph)
        named_attributes_to_vector(graph)
    
    return graph

if __name__ == "__main__":
    main(sys.argv)
