import sys

def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(description='Normality Test Program')
    parser.add_argument( '-d', '--data', type = str , help = 'path to data folder for graph construction')
    parser.add_argument('-b', '--bigraph', default = False, help = 'Bigraph (directed) edges')
    
    args = parser.parse_args()

    return args

def main():

    args = parse_arguments()

    try:
        import load
    except ImportError:
        print("load.py file does not exist")
        sys.exit()

    graphObj = load.GraphLoader([args.data, args.bigraph])
    graphObj.load_files()
    
    print "Size of graph: %d" % graphObj.graph.size()
    print "Number of nodes: %d" % len(graphObj.graph.nodes())
    print "Size of feature vector: %d" % len(graphObj.graph.node[graphObj.graph.nodes()[0]]['feature_vector'])
    

if __name__ == "__main__":
    main()