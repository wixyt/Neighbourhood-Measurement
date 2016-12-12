import sys

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
        except TestingError:
            raise TestingError()
        except Exception as e:
            print "unhandled exception: %s" % e
            raise TestingError()
    except AttributeError:
        print "load files does not exist!"
        raise TestingError()
    
    test_load = module.GraphLoader()
    test_load.load_files('testdata/', False)
    print type(test_load.graph)
    assert test_load.graph



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
    graphObj.load_files([args.data, args.bigraph])
    
    print "Size of graph: %d" % graphObj.graph.size()
    print "Number of nodes: %d" % len(graphObj.graph.nodes())
    print "Size of feature vector: %d" % len(graphObj.graph.node[graphObj.graph.nodes()[0]]['feature_vector'])
    

if __name__ == "__main__":
    main()