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
        # Ensure total_graph and subgraph is of the right type
        # assert (type(total_graph) == type(nx.Graph()) or type(total_graph) == (nx.DiGraph())), "Total Graph not type of Nx graph"
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

        if decompose is not None:
            self.decomposition(self.total_graph)

        return self.objective_optimization(self.total_graph, self.subgraph, False)


    def find_internal_boundary_nodes(self, edge_nodes):
        boundary_nodes = []
        C = set(self.subgraph.nodes())
        for edge in self.edge_nodes:
            for node in edge:
                if node in C:
                    boundary_nodes.append(node)
        return set(boundary_nodes)



    def create_weight_vector(self, feature_vectors):
        import math
        feature_distribution = {}
        for vector in feature_vectors:
            for i in range(len(vector)):
                if i not in feature_distribution:
                    feature_distribution[i] = 0
                if vector[i] > 0:
                    feature_distribution[i] += 1
        total = 0
        for v in feature_distribution.keys():
            total += feature_distribution[v]
        weight_vector = {}
        for key in feature_distribution.keys():
            weight_vector[key] = abs(math.log((feature_distribution[key] / (total * 1.0))))
        return weight_vector

    def features_to_dict(self, fv):
        feature_dict = {}
        for i in range(len(fv)):
            if fv[i] > 0:
                feature_dict[i] = fv[i]
        return feature_dict

    def weighted_jaccard(self, w_v, fv_a, fv_b):
        # return |sum of fv1_i in fv2_i * w_v_i| / |sum of fv1_i U fv2_i * w_v_i|
        numerator = 0.0
        for i in range(len(fv_a)):
            if fv_a[i] == fv_b[i]:
                numerator += w_v[i]
        denominator = 0.0
        for i in range(len(fv_a)):
            if fv_a[i] == 1 or fv_b[i] == 1:
                denominator += w_v[i]

        print "num: %f & denominator: %f" % (numerator, denominator)
        return numerator / denominator

    def internal_consistency(self, G):
        node_list = G.nodes()
        internal_consistency = 0.0
        adj_m = nx.adj_matrix(G, node_list).toarray()
        length = (adj_m.shape)[1]
        total_edges = G.size()

        for i in range(length):
            for j in range(length):
                # get index for the "suprise value" between two nodes
                suprise_index = adj_m[i][j] - \
                (G.degree(node_list[i])*G.degree(node_list[j]))/(2.0* self.total_graph.size())

                x_i = G.node[node_list[i]]['feature_vector']
                x_j = G.node[node_list[j]]['feature_vector']

                # elewise_product = np.multiply(x_i, x_j)

                # if W is not None:
                #     elewise_product = np.multiply(elewise_product, W)
                total = np.multiply(self.weighted_jaccard(self.weight_vector, x_i, x_j), suprise_index)
                internal_consistency += np.sum(total)

        return internal_consistency

    def external_separability(self, G, E):
        external_separability = 0.0
        # ideally it should be low for high quality neighbourhood
        for edge in E:
            unsup_metric = self.unsuprising_metric(
                G.degree(edge[0]),
                G.degree(edge[1]),
                G.size()
            )

            x_i = np.array(G.node[edge[0]]['feature_vector'])
            x_b = np.array(G.node[edge[1]]['feature_vector'])

            elewise_product = np.multiply(x_i, x_b)
            # if W is not None:
            #     elewise_product = np.multiply(elewise_product, W)
            total = np.multiply(elewise_product, unsup_metric)
            external_separability += np.sum(total)

        return external_separability


    def q(self, i_max, i_min, C, G):
        # x_e is the summation product of external_separability
        # x_i is the summation product of internal_consistency
        # x_ei is the summation product of the internal separability
        x_i = self.internal_consistency(C)
        x_ei = self.external_separability(G, C.edges())
        x_e = self.external_separability(G, self.boundary_edges)
        print "Xi Summation: %f" % ((x_i - i_min)/(i_max-i_min))
        print "Xe Summation: %f" % (x_e/(x_ei-x_e))
        return 1 - ((x_i)/(i_max-i_min)) + (x_e/(x_ei-x_e))

    def objective_optimization(self, graph, C, optimise=False):
        adj_m = nx.adjacency_matrix(C)
        length = sum([len(graph.node[x]['feature_vector']) for x in graph.nodes()])/len(graph.nodes())
        I_max = float(len(adj_m.toarray())**2)
        I_min = self.calculate_imin(C, adj_m)

        # weight vector components are normalised between 0 and 1

        if optimise:
            print "WARNING: Optimisation functions should not occur"
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
            res = - self.q(I_max, I_min, C, graph)

        print "Imax: %f" % I_max
        print "Imin: %f" % I_min
        # print "weight vector after optimisation: %s" % res.x
        # print "results after optimisation of weight vector==="
        print "Normality: %f" % res
        return res

#### HELPER FUNCTIONS

    def calculate_imin(self, C, adj_m):
        node_list = C.nodes()
        minimum = 0.0
        for i in range(len(adj_m.toarray())):
            for j in range(len(adj_m.toarray())):
                minimum += (\
                    self.total_graph.degree(node_list[i])*self.total_graph.degree(node_list[j]))\
                    /(2.0*self.total_graph.size()\
                )
        return -minimum


    def find_boundary_edges(self, subgraph_input):
        # TODO: Find a better way of doing this
        for x in self.subgraph_nodes:
            for y in x.values():
                for z in y:
                    if z not in subgraph_input:
                        self.boundary_edges.append((x.keys()[0], z))


    def unsuprising_metric(self, k_i, k_b, edges):
        return (1 - min(1, (k_i*k_b)/(2.0*edges)))


    def decomposition(self, graph):
        # using PCA decompose the feature vectors into a smaller feature matrix
        pca = PCA(n_components=4)
        feature_matrix = []
        for node in graph.nodes():
            feature_matrix.append(graph.node[node]['feature_vector'])
        sk_transf = pca.fit_transform(np.array(feature_matrix))
        for i, node in enumerate(graph.nodes()):
            graph.node[node]['feature_vector'] = sk_transf[i]


# def suprise_metric(link, i_degree, j_degree, edges):
#     return link - (i_degree*j_degree)/(2.0*edges)

# def adjancency_array(graph, node_list):
#     return nx.adj_matrix(graph, node_list).toarray()

# def hadamard_product(a, b):
#     # practically an XOR on arrays with units of 0 and 1
#     return np.multiply(a, b)

# def internal_consistency(graph, w_i=None):
#     node_list = graph.nodes()
#     internal_consistency = 0.0
#     adj_m = adjancency_array(graph, node_list)
#     length = (adj_m.shape)[1]
#     total_edges = graph.size()


#     for i in range(length):
#         for j in range(length):
#             # get index for the "suprise value" between two nodes
#             suprise_index = suprise_metric(
#                 adj_m[i][j],
#                 graph.degree(node_list[i]),
#                 graph.degree(node_list[j]),
#                 total_edges
#             )

#             # element-wise product of attribute vectors
#             # x_i = np.array(graph.node[node_list[i]]['feature_vector'])
#             # x_j = np.array(graph.node[node_list[j]]['feature_vector'])
#             x_i = np.array(graph.node[node_list[i]]['decomp_features'])
#             x_j = np.array(graph.node[node_list[j]]['decomp_features'])

#             elewise_product = hadamard_product(x_i, x_j)
#             if w_i is not None:
#                 elewise_product = np.multiply(elewise_product, w_i)
#             total = np.multiply(elewise_product, suprise_index)

#             internal_consistency += np.sum(total)

#     return internal_consistency

# def boundary_edges(G):
#     E_list = []
#     for edge in G.edges():
#         if G.node[edge[0]]['subgraphs'] != G.node[edge[1]]['subgraphs']:
#             E_list.append(edge)
#     return E_list


# def external_separability(G, E, w_e=None):
#     external_separability = 0.0
#     # ideally it should be low for high quality neighbourhood

#     for edge in E:
#         unsup_metric = unsuprising_metric(
#             G.degree(edge[0]),
#             G.degree(edge[1]),
#             len(G.edges())
#         )
#         # x_i = np.array(G.node[edge[0]]['feature_vector'])
#         # x_b = np.array(G.node[edge[1]]['feature_vector'])
#         x_i = np.array(G.node[edge[0]]['decomp_features'])
#         x_b = np.array(G.node[edge[1]]['decomp_features'])

#         elewise_product = hadamard_product(x_i, x_b)
#         if w_e is not None:
#             elewise_product = np.multiply(elewise_product, w_e)
#         total = np.multiply(elewise_product, unsup_metric)

#         external_separability += np.sum(total)


#     return external_separability

# def cluster_by_degree(graph):
#     # VERY BAD WAY OF CLUSTERING
#     total_degree = 0
#     nodes = 0
#     for node in graph.nodes():
#         nodes += 1
#         total_degree += graph.degree(node)
#     avg_degree = total_degree/nodes

#     for node in graph.nodes():
#         if graph.degree(node) >= avg_degree:
#             graph.node[node]['subgraph'] = 'C'
#         else:
#             graph.node[node]['subgraph'] = 'B'


# def subgraph_separate(graph):
#     subgraphs = []
#     count = 0
#     remaining_subgraphs = True
#     while remaining_subgraphs:
#         subgraph_nodes = []
#         for node in graph.nodes():
#             if 'subgraphs' in graph.node[node]:
#                 if count in graph.node[node]['subgraphs']:
#                     subgraph_nodes.append(node)
#         count += 1
#         if len(subgraph_nodes) == 0:
#             break
#         subgraphs.append(graph.subgraph(subgraph_nodes))
#         del subgraph_nodes
#     return subgraphs


# def calculate_normality(C, graph):
#     I = internal_consistency(C)
#     print("Internal Consistency: %f"  % I)
#     E = external_separability(graph, boundary_edges(graph))
#     print("External Separability: %f"  % E)
#     # Calculate Normality
#     N = I - E
#     print("Normality: %f" % N)
#     print("Optizmizing weight vector...")
#     objective_optimization(graph, C)
#     return N

# def calculate_imin(C, adj_m):
#     node_list = C.nodes()
#     minimum = 0.0
#     for i in range(len(adj_m.toarray())):
#         for j in range(len(adj_m.toarray())):
#             minimum += -(C.degree(node_list[i])*C.degree(node_list[i]))/(2.0*len(C.edges()))
#     return minimum

# def optimize(w_v, C, G, i_max): # taking out one weight vector for simplisiticity
#     # TODO: make this parameters more efficient
#     # x_i, x_i = weight_vectors
#     return internal_consistency(C, w_v[0]) - external_separability(G, boundary_edges(G), w_v[1])

# def objective_optimization(graph, C):
#     adj_m = nx.adjacency_matrix(C)
#     length = sum([len(graph.node[x]['decomp_features']) for x in graph.nodes()])/len(graph.nodes())
#     I_max = float(len(adj_m.toarray())**2)
#     I_min = calculate_imin(C, adj_m)
#     x_i = np.ones(length)
#     x_e = np.ones(length)
#     # weight vector components are normalised between 0 and 1
#     bnds = tuple((0, 1) for x in x_i)


#     # res = sp.optimize.minimize(fun=optimize, method='BFGS', jac=True, args=(x_i, C, graph), options={"maxiter": 5000}, bounds=bounds)
#     res = sp.optimize.minimize(
#         optimize, # function
#         [x_i, x_e], # weight vector
#         args=(C, graph, I_max, ), # other parameters to be passed in as arguments to the function
#         method='L-BFGS-B',
#         bounds=bnds, # bounds of the weight vector
#         options={"maxiter": 30}
#         )
#     print "weight vector after optimisation: %s" % res.x
#     print "results after optimisation of weight vector==="
#     print "Normality: %f" % (optimize(res.x, C, graph, I_max))

# def decomposition(graph):
#     # using PCA decompose the feature vectors into a smaller feature matrix
#     pca = PCA(n_components=4)
#     feature_matrix = []
#     for node in graph.nodes():
#         feature_matrix.append(graph.node[node]['feature_vector'])
#     sk_transf = pca.fit_transform(np.array(feature_matrix))
#     print sk_transf
#     for i, node in enumerate(graph.nodes()):
#         graph.node[node]['decomp_features'] = sk_transf[i]


# def operations(graph):
#     decomposition(graph)
#     subgraphs = subgraph_separate(graph)
#     count = 1
#     for subgraph in subgraphs:
#         print "subgraph: %d" % count
#         calculate_normality(subgraph, graph)
#         count += 1

# def main(args):
#     # TODO: add argument parser
#     # ARGUMENTS
#     # -[l|c] data_directory {--directed} {--pickle|--json}
#     # -l load from graph file like JSON or PICKLE
#     # -c Create graph from data files (edges, features, etc)
#     # --directed

#     if len(args) < 4:
#         print("""Invalid arguments EXITING \n ARGUMENTS
#     # -[l|c] data_directory {--directed} {--pickle|--json}
#     # -l load from graph file like JSON or PICKLE
#     # -c Create graph from data files (edges, features, etc)
#     # --directed """
#     )
#         sys.exit()
#     _dir = os.path.dirname(os.path.realpath(__file__))

#     if args[1] == '-l': #TODO: get loading from file to work
#         graph_file = os.path.join(_dir + '/' + args[2])
#         if args[3] == '--pickle':
#             graph = nx.read_gpickle(graph_file)
#         elif args[3] == '--json':
#             graph = read_json_file(graph_file)


#     elif args[1] == '-c':
#         if args[3] == '--directed':
#             graph = wrapper(args[2], "directed")
#         else:
#             graph = wrapper(args[2])



#     else:
#         print("No valid arguments given")
#     print "Graph size: %d" % graph.size()
#     operations(graph)

# if __name__ == "__main__":
#     main(sys.argv)
