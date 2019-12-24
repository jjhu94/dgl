import dgl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_karate_club_graph():
    g = dgl.DGLGraph()
    # add 34 nodes into the graph; nodes are labeled from 0~33
    g.add_nodes(34)
    # all 78 edges as a list of tuples
    edge_list = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2),
        (4, 0), (5, 0), (6, 0), (6, 4), (6, 5), (7, 0), (7, 1),
        (7, 2), (7, 3), (8, 0), (8, 2), (9, 2), (10, 0), (10, 4),
        (10, 5), (11, 0), (12, 0), (12, 3), (13, 0), (13, 1), (13, 2),
        (13, 3), (16, 5), (16, 6), (17, 0), (17, 1), (19, 0), (19, 1),
        (21, 0), (21, 1), (25, 23), (25, 24), (27, 2), (27, 23),
        (27, 24), (28, 2), (29, 23), (29, 26), (30, 1), (30, 8),
        (31, 0), (31, 24), (31, 25), (31, 28), (32, 2), (32, 8),
        (32, 14), (32, 15), (32, 18), (32, 20), (32, 22), (32, 23),
        (32, 29), (32, 30), (32, 31), (33, 8), (33, 9), (33, 13),
        (33, 14), (33, 15), (33, 18), (33, 19), (33, 20), (33, 22),
        (33, 23), (33, 26), (33, 27), (33, 28), (33, 29), (33, 30),
        (33, 31), (33, 32)]
    # add edges two lists of nodes: src (source node features) and dst (destination node features)
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    g.add_edges(dst, src)  # edges are directional in DGL

    return g


# Create a graph in DGL
G = build_karate_club_graph()
print('We have %d nodes.' % G.number_of_nodes())
print('We have %d edges.' % G.number_of_edges())

# Visualize the graph
nx_G = G.to_networkx().to_undirected()  # the actual graph is undirected
pos = nx.kamada_kawai_layout(nx_G)  # Kamada-Kawaii layout usually looks pretty for arbitrary graphs
nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])

# Add features to nodes or edges
G.ndata["feat"] = torch.eye(34)  # adding the one-hot feature for all nodes


### Define a Graph Convolutional Network (GCN) ###
# Each node will update its own feature with information sent from neighboring nodes.
def gcn_message(edges):  # get original h
    return {"msg": edges.src["h"]}


def gcn_reduce(nodes):  # calculate new h
    return {"h": torch.sum(nodes.mailbox["msg"], dim=1)}


# Define the GCNLayer module
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, inputs):  # g is the graph and the inputs is the input node features
        g.ndata["h"] = inputs
        g.send(g.edges(), gcn_message)
        g.recv(g.nodes(), gcn_reduce)
        h = g.ndata.pop("h")  # get the result node features
        return self.linear(h)  # perform linear transformation


# Define a 2-layer GCN model
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(in_feats, hidden_size)
        self.gcn2 = GCNLayer(hidden_size, num_classes)

    def forward(self, g, inputs):
        h = self.gcn1(g, inputs)
        h = torch.relu(h)
        h = self.gcn2(g, h)
        return h


net = GCN(34, 5, 2)

# Data preparation and initialization
inputs = torch.eye(34)
labeled_nodes = torch.tensor([0, 33])
labels = torch.tensor([0, 1])

# Train and visualize
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
all_logits = []
for epoch in range(50):
    logits = net(G, inputs)
    all_logits.append(logits.detach())  # detach from the autograd graph
    logp = F.log_softmax(logits, dim=1)  # Apply a softmax followed by a logarithm.
    loss = F.nll_loss(logp[labeled_nodes], labels)  # negative log likelihood loss; only compute loss for labeled nodes

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Epoch {} | Loss: {}".format(epoch, loss.item()))


def draw(i):
    cls1color = '#00FFFF'
    cls2color = '#FF00FF'
    pos = {}
    colors = []
    for v in range(34):
        pos[v] = all_logits[i][v].numpy()
        cls = pos[v].argmax()
        colors.append(cls1color if cls else cls2color)
    ax.cla()
    ax.axis("off")
    ax.set_title("Epoch: {}".format(i))
    nx.draw_networkx(nx_G.to_undirected(), pos, node_color=colors, with_labels=True, node_size=300, ax=ax)


fig = plt.figure(dpi=150)
fig.clf()
ax = fig.subplots()
draw(0)

ani = animation.FuncAnimation(fig, draw, frames=len(all_logits), interval=100)

plt.show()
plt.close()


