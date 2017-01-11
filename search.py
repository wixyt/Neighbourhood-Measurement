from analyse import Normality

class IncrementalCluster(object):
    def __init__(self):
        self.cluster_sets = []
        # assign arbitary normality
        self.cluster_count = 0
        self.norm_threshold = 0.07

    def Connections(self, node):
        return node.edges

    def AddNode(self, node, graph):
        node_info = {}
        node_info['node'] = node
        node_info['connected'] = graph.neighbors(node)
        node_info['attributes'] = graph.node[node]['feature_vector']
        self.AssignNode(node_info, graph)


    def AssignNode(self, node, graph):
        # node is node object consisting of connected nodes and attributes
        #{
            # node: id
            # connected: [node ids]
            # attributes: [attribute vector]
        #}
        # find if node is connected to any existing subset, else assign to its own subset

        if self.cluster_count == 0:
            cluster = {
                'cluster_number': self.cluster_count,
                'nodes': [node['node']],

                'node_maps': [node]
            }
            self.cluster_count += 1
            self.cluster_sets.append(cluster)
        elif node['connected']:
            # TODO: THIS ASSIGNS THE NODE TO THE BEST CLUSTER - NEED TO CHANGE IF NODE CAN EXIST IN MORE THAN ONE CLUSTER
            highest_value = 0.0
            best_cluster = -1

            for cluster in self.cluster_sets:
                # if a connect node from node is in cluster set's connected nodes perform compute'
                for connected_node in node['connected']:

                    if connected_node in set(cluster['nodes']):
                        normality = Normality()
                        normality_value = normality.calculate(graph, ([node['node']] + cluster['nodes']))

                        if normality_value > highest_value:
                            highest_value = normality_value
                            best_cluster = cluster['cluster_number']
            if highest_value > 0.50:
                self.cluster_sets[best_cluster]['nodes'].append(node['node'])
                self.cluster_sets[best_cluster]['node_maps'].append(node)
            else:
                cluster = {
                'cluster_number': self.cluster_count,
                'nodes': [node['node']],
                'node_maps': [node]
                }
                self.cluster_count += 1
                self.cluster_sets.append(cluster)
        else:
            cluster = {
                'cluster_number': self.cluster_count,
                'nodes': [node['node']],
                'node_maps': [node]
            }
            self.cluster_count += 1
            self.cluster_sets.append(cluster)
        # assign to cluster set if cluster set is empty


        # calculate normality score with relevant subsets

        # assign node to relevant subset or create new subset consisting of node based on threshold
