import dgl
import matplotlib.pyplot as plt
import networkx as nx
import torch as th


# create a graph

g_nx = nx.petersen_graph()  # create a classical petersen graph from networkx
g_dgl = dgl.DGLGraph(g_nx)  # convert it into a DGLGraph

plt.subplot(121)  # 1 row 2 column, the first subplot
nx.draw(g_nx, with_labels=True)
plt.subplot(122)
nx.draw(g_dgl.to_networkx(), with_labels=True)  # convert it back to a networkx graph

plt.show()


# build a star graph

g = dgl.DGLGraph(multigraph=False)
g.add_nodes(10)
# four methods to add edges
# first: add one for a time
for i in range(1, 5):
    g.add_edge(i, 0)
# second: add several for a time by mapping lists
src = list(range(5, 8))
dst = [0]*3  # [0, 0, 0]
g.add_edges(src, dst)
# third: add several for a time by mapping tensors
src = th.tensor([8, 9])
dst = th.tensor([0, 0])
g.add_edges(src, dst)
# fourth: Edge broadcasting
# g.clear()
# g.add_nodes(10)
src = th.tensor(list(range(1, 10)))
g.add_edges(9, 0)
nx.draw(g.to_networkx(), with_labels=True)
print(g.nodes(), g.edges())
eid_90 = g.edge_id(9, 0)
print(type(eid_90), eid_90)  # <class 'int'> 8
plt.show()


# assign a feature

# assign node feature
x = th.randn(10, 3)  # 10 rows, 3 columns
g.ndata["x"] = x  # .ndata is a syntax sugar to access the state of all nodes; g.ndata["x"] == g.nodes[:].data["x"]
# Access node set with integer, list, or integer tensor
g.nodes[0].data["x"] = th.zeros(1, 3)  # Access node set with integer
g.nodes[[0, 1, 2]].data["x"] = th.zeros(3, 3)  # with list; g.nodes[[0, 1, 2]].data["x"] == g.nodes[0, 1, 2].data["x"]
g.nodes[th.tensor([0, 1, 2])].data["x"] = th.zeros(3, 3)  # Access node set with integer tensor

# assign edge feature
g.edata["w"] = th.randn(10, 2)  # edata means edge data
# Access edge set with IDs in integer, list, or integer tensor
g.edges[1].data["w"] = th.randn(1, 2)  # Access edge set with IDs in integer
g.edges[[0, 1, 2]].data["w"] = th.zeros(3, 2)  # Access edge set with IDs in list
g.edges[th.tensor([0, 1, 2])].data["w"] = th.zeros(3, 2)  # Access edge set with IDs in integer tensor
# You can also access the edges by giving endpoints, notice that they are directional
g.edges[1, 0].data["w"] = th.ones(1, 2)  # g.edges[[0, 1, 2]].data["w"] differs from g.edges[0, 1, 2].data["w"]
g.edges[[3, 1, 2], [0]*3].data["w"] = th.zeros(3, 2)  # edges [1, 2, 3] -> 0
# show a scheme containing the shape and data type (dtype) of its field value
print(g.node_attr_schemes())  # {'x': Scheme(shape=(3,), dtype=torch.float32)}
g.ndata["x"] = th.zeros((10, 4))
print(g.node_attr_schemes())  # {'x': Scheme(shape=(4,), dtype=torch.float32)}
# remove node or edge states from the graph
g.ndata.pop('x')
g.edata.pop('w')
print(g.node_attr_schemes(), g.edge_attr_schemes())  # {}


# working with multigraphs: construct DGLGraph with multigraph=True
g_multi = dgl.DGLGraph(multigraph=True)
g_multi.add_nodes(10)
g_multi.ndata["x"] = th.randn(10, 2)
g_multi.add_edges(list(range(1, 10)), 0)
g_multi.add_edge(1, 0)  # two edges on 1->0, only works in multigraphs
g_multi.edata["w"] = th.randn(10, 2)
g_multi.edges[1].data['w'] = th.zeros(1, 2)
print(g_multi.edges())

eid_10 = g_multi.edge_id(1, 0)  # get the corresponding IDs
print(type(eid_10), eid_10)  # <class 'torch.Tensor'> tensor([0, 9])
g_multi.edges[eid_10].data["w"] = th.ones(len(eid_10), 2)

