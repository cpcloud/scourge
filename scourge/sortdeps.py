from pprint import pprint
import os
import pickle as p
import networkx as nx

toposort = nx.topological_sort

path = '/home/phillip/code/py/arrow-build-new2'

with open(os.path.join(path, '.matrices'), mode='rb') as f:
    matrix = p.load(f)

with open(os.path.join(path, '.graph'), mode='rb') as f:
    graph = p.load(f)


# add all nodes to graph

g = nx.DiGraph()

for node in graph.nodes():
    for constraints in matrix[node]:
        g.add_node((node,) + constraints)


for node in nx.topological_sort(graph):
    node_versions = matrix[node]

    for edge in graph.edge[node]:
        edge_versions = matrix[edge]

        version_intersection = node_versions & edge_versions

        for i in version_intersection:
            g.add_edge((node,) + i, (edge,) + i)

        # print(node, edge)
        if any(node_versions) and (not any(edge_versions)):
            for i in node_versions:
                g.add_edge((node,) + i, (edge,))
        elif (not any(node_versions)) and any(edge_versions):
            for i in edge_versions:
                g.add_edge((node,), (edge,) + i)
pprint(g.edge)
