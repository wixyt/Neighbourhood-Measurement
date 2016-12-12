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


class GraphLoader(object):
    def __init__(self, args):
        self.directed = args[1]
        # TODO: add elif for multigraphs
        if self.directed:
            print ("Returning Directed Graph...")
            self.graph = nx.DiGraph()
        else:
            print ("Returning Undirected Graph...")
            self.graph = nx.Graph()
        self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)) + "/" + args[0])
        self.file_nodes = [int(x.split("/")[-1].split('.')[0]) for x in glob.iglob(self.path + "/*.featnames")]

    
    def save(self, path):
        save_file = json_graph.node_link_data(self.graph)
        json.dump(save_file, open(path, 'w'), indent=2)

    def load_files(self):
        subgraph_count = 0 # value to keep track on total number of subgraphs
        for node in self.file_nodes: 
            # get file paths
            feature_name_file = os.path.join(self.path + "/%s.featnames" % node)
            edge_file = os.path.join(self.path + "/%s.edges" % node)
            edge_feat_file = os.path.join(self.path + "/%s.feat" % node)
            node_feat_file = os.path.join(self.path + "/%s.egofeat" % node)
            circles_file = os.path.join(self.path + "/%s.circles" % node)
            
            # load nodes/attributes onto graph
            
            self.graph.add_node(node)
            attribute_dict = self.load_attributes(feature_name_file)

            self.load_node_features(node_feat_file, attribute_dict, node)
            self.load_nodes(edge_feat_file, attribute_dict)
            self.load_edges(edge_file)
            self.add_empty_subgraph()
            subgraph_count += self.apply_ground_truths(circles_file, subgraph_count)
        self.named_attributes_to_vector()

    def load_attributes(self, feature_file_path):
        # create dictionary of features keys and values
        feature_index = {}
        try: 
            feature_file_contents = open(feature_file_path)
        except IOError:
            print("Could not open feature file.")
            return False
        for line in feature_file_contents:
            feature = line.split(' ')
            feature_index[feature[0]] = feature[-1][:-1]

        return feature_index


    def load_node_features(self, node_feat_file, feature_index, node_id):
        key_array = feature_index.keys()
        with open(node_feat_file, 'r') as file:
            vector_string = file.read()[:-1]
            node_feat_vect = [int(x) for x in vector_string.split(' ')]
            self.apply_attribute_vector(node_feat_vect, node_id, feature_index)
            self.apply_attributes(node_feat_vect, node_id, feature_index)

    
    def apply_attribute_vector(self, feature_vector, node_id, feature_dict):
        key_array = feature_dict.keys()
        self.graph.node[node_id]['attributes'] = [0 for x in range(len(key_array))]
        for key in range(len(feature_dict)):
            self.graph.node[node_id]['attributes'][key] = feature_vector[int(key)]

    
    def apply_attributes(self, feature_vector, node_id, feature_dict):
        key_array = feature_dict.keys()
        self.graph.node[node_id]['named_attributes'] = {}
        for key in range(len(feature_dict)):
            self.graph.node[node_id]['named_attributes'][feature_dict[str(key)]] = feature_vector[int(key)]

    def load_nodes(self, edge_feat_path, feature_index):
        key_array = feature_index.keys()
        with open(edge_feat_path, 'r') as file:
            for line in file:
                try:
                    features = [int(x) for x in line.split(' ')]
                    node = features[0]
                    feature_vect = features[1:]
                    self.graph.add_node(node)
                    
                    self.apply_attribute_vector(feature_vect, node, feature_index)
                    self.apply_attributes(feature_vect, node, feature_index)
                except Exception as e:
                    print "error: %s" % e

    def load_edges(self, path):
        try:
            edge_file_contents = open(path)
        except IOError:
            print "Error reading edge file contents"
        for line in edge_file_contents:
            edges = [int(x) for x in line.split(' ')]
            self.graph.add_edge(edges[0], edges[1])

    
    def add_empty_subgraph(self):
        for node in self.graph.nodes():
            self.graph.node[node]['subgraphs'] = {}

    
    def apply_ground_truths(self, path, count):
        file = open(path)
        for line in file:
            if line[-1] == '\n':
                line = line[:-1]
            circle = line.split()
            for node in circle[1:]:
                # add an entry to subgraphs of the ground truth set 
                self.graph.node[int(node)]['subgraphs'][count] = 1
            count += 1
        return count


    def named_attributes_to_vector(self):
        # extract all named node features into a dictionary containing all known features of the graph 
        # for each node produce a vector that is consistent with the size of the large feature dictionary
        # note: this will result in very large attribute vectors - but feature selection will reduce the size
        total = {}
        for node in self.graph.nodes():
            for element in self.graph.node[node]['named_attributes']:
                if element not in total:
                    total[element] = 1
        sorted_dict = sorted(total.items(), key=operator.itemgetter(0))
        
        v = DictVectorizer(sparse=False)
        for node in self.graph.nodes():
            X = v.fit_transform([total, self.graph.node[node]['named_attributes']])
            # nx.set_node_attributes(G, 'full_featvector', {node: X[1].tolist()})
            self.graph.node[node]['feature_vector'] = X[1].tolist() # numpy arrays cannot be stored properly
        

    def draw_nx_graph(self):
        nx.draw(self.graph)
        plt.show()


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
