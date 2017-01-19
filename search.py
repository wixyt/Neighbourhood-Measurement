from analyse import Normality

class IncrementalCluster(object):
    """
        Class to perform clustering by evaluating the best cluster for each node on a
        node-by-node basis (incremental).
        TODO: establish compound datastructure for existing clusters for more timely
        computation of clusters.
    """
    def __init__(self):
        self.cluster_sets = []
        # assign arbitary normality
        self.cluster_count = 0
        self.norm_threshold = 0.07

    def Connections(self, node):
        return node.edges

    def LargestCluster(self):
        """
            Return the largest cluster after node assignment takes place. Useful for
            marking nodes that are part of the largest cluster in visualisation.
        """
        largest = -1
        current = 1
        for cluster in self.cluster_sets:
            if len(cluster['nodes']) > current:
                largest = cluster['cluster_number']
                current = len(cluster['nodes'])
        if current > 1:
            return self.cluster_sets[largest]['nodes']
        else:
            return "No Cluster"

    def AddNode(self, node, graph):
        """
        node is node object consisting of connected nodes and attributes
        {
            node: id
            connected: [node ids]
            attributes: [attribute vector]
        }
        """
        node_info = {}
        node_info['node'] = node
        node_info['connected'] = graph.neighbors(node)
        node_info['attributes'] = graph.node[node]['feature_vector']
        self.AssignNode(node_info, graph)

    def EvaluateCurrentCluster(self, N, C, T):
        print "Current cluster value: %f" % C
        print "Normality Value %f" % N
        print "Current Cluster Size %f" % T
        before = ((C * 1.0) / (T * 1.0))
        after = N / (T * 1.0)
        print "Derivative of current cluster %f" % before
        print "Derivative after addition of node %f" % after
        if before <= after:
            return True
        else:
            return False

    def AssignNode(self, node, graph):
        """
            Individual node is assigned to cluster based on its similarity to existing
            clusters. If no clusters exist or no cluster is suitable, where the
            current node is dissimilar, assign the node to its own, new cluster.
        """
        print "\n\n" + "+" * 15

        # assign to cluster set if cluster set is empty
        if self.cluster_count == 0:

            cluster = {
                'cluster_number': self.cluster_count,
                'nodes': [node['node']],
                'cluster_value': 0.0,
                'node_maps': [node]
            }
            self.cluster_count += 1
            self.cluster_sets.append(cluster)
        # determine if node is connected to any existing subset, else assign to its own subset
        elif node['connected']:
            # TODO: Assign nodes to more than one cluster if normality is high enough,
            # node can belong in more than one subgraph set.
            highest_value = 0.0
            best_cluster = -1

            for cluster in self.cluster_sets:
                for connected_node in node['connected']:
                    # if a connect node from node is in cluster set's connected nodes
                    # perform normality computation.
                    if connected_node in set(cluster['nodes']):
                        normality = Normality()
                        # calculate normality score with relevant subsets
                        normality_value = normality.calculate(graph, ([node['node']] + cluster['nodes']))

                        if normality_value > highest_value:
                            highest_value = normality_value
                            best_cluster = cluster['cluster_number']
            print "Best cluster: \nNode %s cluster %s \n" % (node['node'], self.cluster_sets[best_cluster]['nodes'])
            # assign node to relevant subset or create new subset consisting of node based on threshold
            if self.EvaluateCurrentCluster(
                highest_value,
                self.cluster_sets[best_cluster]['cluster_value'],
                len(self.cluster_sets[best_cluster]['nodes'])
                ) and highest_value > 0.098:
                self.cluster_sets[best_cluster]['nodes'].append(node['node'])
                self.cluster_sets[best_cluster]['node_maps'].append(node)
                self.cluster_sets[best_cluster]['cluster_value'] = highest_value
            else:
                cluster = {
                'cluster_number': self.cluster_count,
                'nodes': [node['node']],
                'node_maps': [node],
                'cluster_value': 0.0
                }
                self.cluster_count += 1
                self.cluster_sets.append(cluster)
        else:
            cluster = {
                'cluster_number': self.cluster_count,
                'nodes': [node['node']],
                'node_maps': [node],
                'cluster_value': 0.0
            }
            self.cluster_count += 1
            self.cluster_sets.append(cluster)







# TODO: compare incremental node by node clustering with search based clustering
def breadth_first_search(graph, start_node):
    marked, queue = set(), [start_node]
    while queue:
        vertex = queue.pop(0)
        if vertex not in marked:
            marked.add(vertex)
            queue.extend(graph[vertex] - marked)
    return marked

def depth_first_search(graph, start_node, marked=None):
    # Graph = {1: [EDGES], 2: [EDGES], ... }
    if marked is None:
        marked = set()
    marked.add(start_node)
    for next in graph[start_node] - marked:
        depth_first_search(graph, next, marked)
    return marked
