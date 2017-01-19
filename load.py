#!/usr/bin/env python
import sys
import networkx as nx
import numpy as np
import os, os.path
import glob
import math
import operator




class GraphLoader(object):
    def __init__(self):
        """
            parameters:
            directed - whether the graph is directed, e.g. connections in 1 direction
            path - path to file directory
            file_nodes - prefix of filesnames, e.g. 170.edges, 170.circles etc.
        """
        self.directed = bool
        self.path = str
        self.file_nodes = [ int ]


    def read_json_file(self, path):
        """
            If file is passed in with the appropriate arguments and is a JSON file,
            parse to graph structure using inbuilt methods.
        """
        try:
            file = open(path)
            js_graph = json.load(f)
            try:
                graph = json_graph.node_link_graph(js_graph)
            except NetworkXError as e:
                print "Error loading Json graph: %s" % e
            return graph
        except IOError:
            print "error reading JSON graph file"
            return False

    def generate_random(self, node_number, size, attribute_vector_size):
        """
            Generate random graph from node and size values.
        """
        import random
        self.graph = nx.Graph()
        self.graph.add_nodes_from(range(0, node_number))
        while len(self.graph.edges()) < size:
            n1 = random.randrange(0, node_number, 1)
            n2 = random.randrange(0, node_number, 1)
            if n1 != n2:
                self.graph.add_edge(n1, n2)
        for node in self.graph.nodes():
            attr_vec = []
            for i in range(attribute_vector_size):
                attr_vec.append(random.randint(0, 1))
            self.graph.node[node]['feature_vector'] = attr_vec
        return True


    def save(self, path):
        """
            Save graph to specified file locaiton using inbuilt methods.
        """

        try:
            from networkx.readwrite import json_graph
        except ImportError as e:
            print "Import error loading networkx json_graph:\n %s" % e
            return False
        try:
            import json
        except ImportError as e:
            print "Json import error: %s" % e
            return False

        save_file = json_graph.node_link_data(self.graph)
        json.dump(save_file, open(path, 'w'), indent=2)
        return True


    def load_files(self, args):
        """
            Load data files (egonet files) from specified file directory
            location or JSON file.
            Using filename data parse nodes, circles, edges, and feature vectors from
            files within file directory.
        """

        if args[0].endswith('.json'):
            self.graph = read_json_file(args[0])
            return True
        else:
            self.path = os.path.join(os.path.dirname(os.path.realpath(__file__)) + "/" + args[0])
            self.file_nodes = [int(x.split("/")[-1].split('.')[0]) for x in glob.iglob(self.path + "/*.featnames")]
            self.directed = args[1]
            try:
                import networkx as nx
            except ImportError:
                print "Networkx library does not exist"
                return False
            if self.directed:
                self.graph = nx.DiGraph()
            else:
                self.graph = nx.Graph()

            subgraph_count = 0 # value to keep track on total number of subgraphs
            try:
                for node in self.file_nodes:

                    # establish file paths
                    feature_name_file = os.path.join(self.path + "/%s.featnames" % node)
                    edge_file = os.path.join(self.path + "/%s.edges" % node)
                    edge_feat_file = os.path.join(self.path + "/%s.feat" % node)
                    node_feat_file = os.path.join(self.path + "/%s.egofeat" % node)
                    # TODO: try except statement for when circles files does not exist - do not add groundtruths etc
                    circles_file = os.path.join(self.path + "/%s.circles" % node)
                    # load nodes/attributes onto graph

                    self.graph.add_node(node)
                    attribute_dict = self.load_attributes(feature_name_file)

                    self.load_node_features(node_feat_file, attribute_dict, node)
                    self.load_nodes(edge_feat_file, attribute_dict)
                    self.load_edges(edge_file)
                    self.add_empty_subgraph()
                    subgraph_count += self.apply_ground_truths(circles_file, subgraph_count)
            except IOError:
                print "IOError in main load function"
                return False
            except Exception as e:
                print "undefined error occurred %s" % sys.exc_info()[0]
                return False
            else:
                self.named_attributes_to_vector()
                # returns True


    def load_attributes(self, feature_file_path):
        """
        create dictionary of features keys and values
        """
        feature_index = {}
        try:
            feature_file_contents = open(feature_file_path)
            for line in feature_file_contents:
                feature = line.split(' ')
                feature_index[feature[0]] = feature[-1][:-1]
            return feature_index
        except IOError:
            print("Could not open feature file.")
            raise ImportError



    def load_node_features(self, node_feat_file, feature_index, node_id):
        """
            Intermediate function that saves node feature vectors to the relevant nodes
        """
        key_array = feature_index.keys()
        try:
            file = open(node_feat_file)
        except IOError:
            print "Could not open node feature file %s" % node_feat_file
            raise IOError
        vector_string = file.read()[:-1]
        node_feat_vect = [int(x) for x in vector_string.split(' ')]
        self.apply_attribute_vector(node_feat_vect, node_id, feature_index)
        self.apply_attributes(node_feat_vect, node_id, feature_index)
        return True


    def apply_attribute_vector(self, feature_vector, node_id, feature_dict):
        """
            Saves attribute vector (1D array of categorical variables, e.g. [0, 1])
            to attribute key of the node
        """
        key_array = feature_dict.keys()
        self.graph.node[node_id]['attributes'] = [0 for x in range(len(key_array))]
        for key in range(len(feature_dict)):
            self.graph.node[node_id]['attributes'][key] = feature_vector[int(key)]


    def apply_attributes(self, feature_vector, node_id, feature_dict):
        """
            Saves named attributes, e.g. "Friends": 1, as dictionary on the specified node.
            Potentially more flexible than a feature vector, particularly as the
            size of the feature vector increases (large feature vectors are generally sparse).
        """
        key_array = feature_dict.keys()
        self.graph.node[node_id]['named_attributes'] = {}
        for key in range(len(feature_dict)):
            self.graph.node[node_id]['named_attributes'][feature_dict[str(key)]] = feature_vector[int(key)]


    def load_nodes(self, edge_feat_path, feature_index):
        """
            Add nodes from file to graph structure.
        """
        key_array = feature_index.keys()
        try:
            file = open(edge_feat_path)
        except IOError:
            print "Error reading edge feature file"
            return False
        for line in file:
            try:
                features = [int(x) for x in line.split(' ')]
                node = features[0]
                feature_vect = features[1:]
                self.graph.add_node(node)

                self.apply_attribute_vector(feature_vect, node, feature_index)
                self.apply_attributes(feature_vect, node, feature_index)
            except Exception as e:
                print "Error at line %s: %s" % (line, e)
        return True


    def load_edges(self, path):
        """
            Add edges to graph structure from the file location.
        """

        try:
            edge_file_contents = open(path)
        except IOError:
            print "Error reading edge file contents"
            return False
        for line in edge_file_contents:
            edges = [int(x) for x in line.split(' ')]
            self.graph.add_edge(edges[0], edges[1])
        return True


    def add_empty_subgraph(self): #TODO: why is this here (idea?)
        for node in self.graph.nodes():
            self.graph.node[node]['subgraphs'] = {}


    def apply_ground_truths(self, path, count):
        """
            Fancy way of adding known circles from egonet files to the graph structure
            as subgraphs.
        """
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

        """
            extract all named node features into a dictionary containing all known features of the graph
            for each node produce a vector that is consistent with the size of the large feature dictionary
            note: this will result in very large attribute vectors - but feature selection will reduce the size
        """
        total = {}
        for node in self.graph.nodes():
            for element in self.graph.node[node]['named_attributes']:
                if element not in total:
                    total[element] = 1
        sorted_dict = sorted(total.items(), key=operator.itemgetter(0))

        try:
            # function is normally only called once so import is okay here TODO: check this
            from sklearn.feature_extraction import DictVectorizer
        except ImportError:
            print "sklearn module missing"
            return False

        v = DictVectorizer(sparse=False)
        for node in self.graph.nodes():
            X = v.fit_transform([total, self.graph.node[node]['named_attributes']])
            self.graph.node[node]['feature_vector'] = X[1].tolist() # numpy arrays cannot be stored properly

    def draw_nx_graph(self):
        """
        Plot graph using inbuilt methods - helpful for visualising the graph during
        testing.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print "Matplotlib import error"
            return False
        nx.draw(self.graph)
        plt.show()
        return True


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
