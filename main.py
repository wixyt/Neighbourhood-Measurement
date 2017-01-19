#!/usr/bin/env python
import sys
import itertools

def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(description='Normality Test Program')
    parser.add_argument( '-d', '--data', type = str , help = 'path to data folder for graph construction')
    parser.add_argument('-b', '--bigraph', default = False, help = 'Bigraph (directed) edges')

    args = parser.parse_args()

    return args

class TestingError(Exception):
    """
        Error raised if testing fails
    """

def test_load_module(module):
    """
        Test that module contains the appropriate elements
    """
    print "-" * 40
    print "Testing graph loader module"
    print "-" * 40
    try:
        gl_class = module.GraphLoader
        if type(gl_class) != type:
            print "GraphLoader is not a class"
            class tc: pass
            if type(gl_class) == type(tc):
                print "GraphLoader does not inherit from Object Class"
            raise TestingError()
        print "Graph Loader Module Exists"
    except AttributeError:
        print "Graph loader does not exist"
        raise TestingError()

    try:
        graph_loader = module.GraphLoader()
    except Exception as e:
        print "Error calling Graph Loader: %s" % e
        raise TestingError()

    dummy_test = '/test/file/location/'
    try:
        rsf = graph_loader.load_files
        print "-" * 40
        print "Testing load file method..."
        print "-" * 40
        try:
            if rsf(dummy_test):
                print "Inadvertently returned true from loading dummy file"
                raise TestingError()
        except IOError:
            print "Uncaught IOError"
            raise TestingError()
        except Exception as e:
            print "Unexpected error: %s" % e
            raise TestingError()

        try:
            if hasattr(module.GraphLoader, 'path'):
                print "'path' is a class variable and does not belong to the instance"
                raise TestingError()

            if hasattr(module.GraphLoader, 'directed'):
                print "'directed' is class variable and does not belong to the instance"
                raise TestingError()

            if hasattr(module.GraphLoader, 'graph'):
                print "'graph' is a class variable and does not belong to the instance"
                raise TestingError()

            if hasattr(module.GraphLoader, 'file_nodes'):
                print "'file_nodes' is a class variable and does not belong to the instance"
                raise TestingError()

            print "Load File Attributes Verified"
        except TestingError:
            raise TestingError()
        except Exception as e:
            print "unhandled exception: %s" % e
            raise TestingError()
    except AttributeError:
        print "load files does not exist!"
        raise TestingError()

    test_load = module.GraphLoader()
    test_load.load_files(['testdata/', False])

    try:
        import networkx as nx
    except ImportError:
        print "error importing networkx library, make sure this is installed"
        raise TestingError()
    try:
        assert type(test_load.graph) == type(nx.Graph()), "Type of Networkx Graph"
        print "Graph exists and is of correct graph type"
    except AssertionError:
        print "Graph data structure does not exist"
        raise TestingError()

    try:
        assert test_load.graph.size() == 6
        assert len(test_load.graph.nodes()) == 6
        assert len(test_load.graph.node[test_load.graph.nodes()[0]]['feature_vector']) == 5
        print "test graph features loaded correctly"
    except AssertionError:
        print "test graph features did not load correctly - check parameters"
        raise TestingError()
    print "-" * 40
    print "All tests passed"
    print "-" * 40


def calculate_graph_normality(graph):
    """
        Processes:
        1. Calculate normality score for each edge - display to user
        2. Import and assign each node to cluster based on normality score
    """
    try:
        from analyse import Normality
    except ImportError as imp_e:
        print "analyse.py file does not exist! %s " % imp_e
        sys.exit()

    for i, edge in enumerate(graph.edges()):
        normality = Normality()
        print "=" * 5 + "\n"
        print "Subgraph: ", edge
        print "Normality %s: %f" % (edge, normality.calculate(graph, [edge[0], edge[1]]))

    try:
        import search
    except ImportError:
        print("search.py file does not exist")
        sys.exit()

    clusters = search.IncrementalCluster()

    for node in graph.nodes():
        clusters.AddNode(node, graph)

    for cluster in clusters.cluster_sets:
        print cluster['cluster_number']
        for node_info in cluster['node_maps']:
            print node_info


def plot(edges):
    # TODO: implement bar graph for clusters within a graph
    import matplotlib.pyplot as plt
    import numpy as np
    width = 1/1.5
    plt.bar(np.arange(len(graph.edges())), normality_outputs, width, color="green")
    plt.xticks(np.arange(len(graph.edges())), graph.edges())
    plt.ylabel("normality")
    plt.show()


def main():

    args = parse_arguments()

    try:
        import load
    except ImportError:
        print("load.py file does not exist")
        sys.exit()

    if args.data:
        try:
            test_load_module(load)
        except TestingError:
            pass

    graphObj = load.GraphLoader()
    if args.data:
        graphObj.load_files([args.data, args.bigraph])
    else:
        graphObj.generate_random(12, 14, 10)

    print "Size of graph: %d" % graphObj.graph.size()
    print "Number of nodes: %d" % len(graphObj.graph.nodes())
    print "Size of feature vector: %d" % len(graphObj.graph.node[graphObj.graph.nodes()[0]]['feature_vector'])


    calculate_graph_normality(graphObj.graph)

    #Matplot Lib:
    # plot(graphObj.graph)




if __name__ == "__main__":
    main()
