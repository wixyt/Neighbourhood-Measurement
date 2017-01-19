#!/usr/bin/env python
import sys
import argparse
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import json
import os
# from sklearn.decomposition import PCA
from networkx.readwrite import json_graph

# TODO: debug edges where subgraph is not connected

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
        self.subgraph_nodes = dict
        self.subgraph_input = [] # for find_edge_nodes
        self.target_node = int
        self.boundary_edges = []


    def calculate(self, total_graph, subgraph, target=None, decompose=None):
        """
            Main method that performs establishment of subgraph objects and weight vector
            Calls inner calculation methods
        """
        assert (type(total_graph) == type(nx.Graph()) or type(total_graph) == (nx.DiGraph())), "Total Graph not type of Nx graph"
        assert type(subgraph) == type(list()), "Subgraph not of type list"
        self.total_graph = total_graph
        self.subgraph = total_graph.subgraph(subgraph)
        self.subgraph_nodes = [
            {x: [y for y in nx.all_neighbors(self.total_graph, x)]} for x in subgraph
        ]
        self.feature_vectors = [self.total_graph.node[x]['feature_vector'] for x in self.total_graph.nodes()]
        # TODO: attach weight vector to graph object class
        self.weight_vector = self.create_weight_vector(self.feature_vectors)

        self.find_boundary_edges(subgraph)
        # self.internal_boundary_nodes = self.find_internal_boundary_nodes(self.edge_nodes)

        # Used if feature vectors are too large (vector size > 1000)
        if decompose is not None:
            self.decomposition(self.total_graph)

        return self.results(self.total_graph, self.subgraph, False)


    def find_internal_boundary_nodes(self, edge_nodes):
        """
            Internal boundary nodes are used to find all nodes that have existing connections
            with nodes or subgraphs outside the current subgraph.
        """
        boundary_nodes = []
        C = set(self.subgraph.nodes())
        for edge in self.edge_nodes:
            for node in edge:
                if node in C:
                    boundary_nodes.append(node)

        return set(boundary_nodes)



    def create_weight_vector(self, feature_vectors):
        """
            Sets up a weight vector based on all node attribute vectors within the graph.
            Weight vector is inversely weighted so more frequently appeared attributes are assigned a lower weight;
            less frequently appearing attributes are assigned a high weight. This weight vector is then used to
            better separate subgraphs based on the similarity and rarity of their attributes.
        """
        import math
        feature_distribution = {}
        for vector in feature_vectors:
            for i in range(len(vector)):
                if i not in feature_distribution:
                    feature_distribution[i] = 0
                if vector[i] == 1:
                    feature_distribution[i] += 1
        total = 0
        for v in feature_distribution.keys():
            total += feature_distribution[v]
        weight_vector = {}
        for key in feature_distribution.keys():
            weight_vector[key] = abs(math.log(((total * 1.0) / feature_distribution[key] + 1.0)))

        return weight_vector

    def features_to_dict(self, fv):
        feature_dict = {}
        for i in range(len(fv)):
            if fv[i] > 0:
                feature_dict[i] = fv[i]

        return feature_dict

    def weighted_jaccard(self, w_v, fv_a, fv_b):
        """
            calculates the jaccard distance (similarity) of two feature vectors.
            |sum of fv1_i in fv2_i * w_v_i| / |sum of fv1_i U fv2_i * w_v_i|
        """
        numerator = 0.0
        denominator = sum([x for x in w_v.values()])

        for i in range(len(fv_a)):
            if fv_a[i] == fv_b[i]:
                numerator += 1.0
        # weight vector components are normalised between 0 and 1

        return numerator / denominator

    def internal_consistency(self, G):
        """
            Internal consistency is the measure of how cohesive nodes are within the current subgraph.
            This is the summation product of the amount of connections there exist between nodes within
            the subgraph multiplied by the similarity of of their attribute vectors.
        """
        node_list = G.nodes()
        internal_consistency = 0.0
        adj_m = nx.adj_matrix(G, node_list).toarray()
        length = (adj_m.shape)[1]
        total_edges = G.size()
        total_weightOfJac = 0.0
        for i in range(length):
            for j in range(length):
                # calculate index for the "suprise value" between two nodes
                suprise_index = adj_m[i][j] - \
                (G.degree(node_list[i])*G.degree(node_list[j]))/(2.0* self.total_graph.size())

                x_i = G.node[node_list[i]]['feature_vector']
                x_j = G.node[node_list[j]]['feature_vector']

                total = np.multiply(self.weighted_jaccard(
                    self.weight_vector,
                    x_i,
                    x_j),
                suprise_index)
                internal_consistency += np.sum(total)

        return internal_consistency

    def external_separability(self, G, E):
        """
            Calculate the difference between edge nodes of the current subgraph and
            external connected nodes. Difference based on degree of edge nodes and the
            difference of their respective feature vectors.
        """
        external_separability = 0.0
        for edge in E:
            unsup_metric = self.unsuprising_metric(
                G.degree(edge[0]),
                G.degree(edge[1]),
                G.size()
            )
            x_i = np.array(G.node[edge[0]]['feature_vector'])
            x_b = np.array(G.node[edge[1]]['feature_vector'])

            total = np.multiply(self.weighted_jaccard(self.weight_vector, x_i, x_b), unsup_metric)
            external_separability += np.sum(total)

        return external_separability


    def results(self, graph, C, optimise=False):
        """
            Calculation of preliminary measurements like adjacency matrix, maximum and minimum values for
            normalisation.
        """
        adj_m = nx.adjacency_matrix(C)
        length = sum([len(graph.node[x]['feature_vector']) for x in graph.nodes()])/len(graph.nodes())
        I_max = float(len(adj_m.toarray())**2)
        I_min = self.calculate_imin(C, adj_m)

        def normalised_summation_products(i_max, i_min, C, G):
            i = self.internal_consistency(C) / len(C.nodes()) * 1.0
            e = self.external_separability(G, self.boundary_edges) / len(C.nodes()) * 1.0
            return (i - e)

        if optimise:
            # TODO: optimisation
            print("In optimisation!")
            # initial_vector = np.ones(length)
            # res = sp.optimize.minimize(
            #     self.q, # function
            #     intial_vector, # weight vector
            #     args=(I_max, I_min, C, graph, ), # other parameters to be passed in as arguments to the function
            #     method='L-BFGS-B',
            #     bounds=tuple((0, 1) for x in initial_vector), # bounds of the weight vector
            #     options={"maxiter": 5}
            # )
            # res = self.q(res.x, I_max, I_min, C, graph, True)

        else:
            res =  normalised_summation_products(I_max, I_min, C, graph)

        # print "Imax: %f" % I_max
        # print "Imin: %f" % I_min
        # print "weight vector after optimisation: %s" % res.x
        # print "results after optimisation of weight vector==="
        # print "Normality: %f" % res

        return res

#### HELPER FUNCTIONS

    def calculate_imin(self, C, adj_m):
        """
            Smallest value for Internal Consistency: no similarity in feature vectors and
            no connections between subgraph nodes.
        """
        node_list = C.nodes()
        minimum = 0.0
        for i in range(len(adj_m.toarray())):
            for j in range(len(adj_m.toarray())):
                minimum -= (\
                    self.total_graph.degree(node_list[i])*self.total_graph.degree(node_list[j]))\
                    /(2.0*self.total_graph.size()\
                )

        return minimum


    def find_boundary_edges(self, subgraph_input):
        for x in self.subgraph_nodes:
            for y in x.values():
                for z in y:
                    if z not in subgraph_input:
                        self.boundary_edges.append((x.keys()[0], z))


    def unsuprising_metric(self, k_i, k_b, edges):
        return (1 - min(1, (k_i*k_b)/(2.0*edges)))


    def decomposition(self, graph):
        """
        using PCA, decompose the feature vectors to a smaller feature matrix size
        """

        pca = PCA(n_components=4)
        feature_matrix = []
        for node in graph.nodes():
            feature_matrix.append(graph.node[node]['feature_vector'])
        sk_transf = pca.fit_transform(np.array(feature_matrix))
        for i, node in enumerate(graph.nodes()):
            graph.node[node]['feature_vector'] = sk_transf[i]
