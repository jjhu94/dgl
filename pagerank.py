import dgl
import dgl.function as fn
import matplotlib.pyplot as plt
import networkx as nx
import time
import torch

# create an erdos renyi graph
N = 100  # number of nodes
DAMP = 0.85  # weight
K = 10
g = nx.nx.erdos_renyi_graph(N, 0.1)  # erdos_renyi_graph(n, p), n - number of nodes, p - Probability for edge creation.
g = dgl.DGLGraph(g)
nx.draw(g.to_networkx(), node_size=50, node_color=[[.5, .5, .5]])
plt.show()

g.ndata['pv'] = torch.ones(N) / N  # Initialize the PageRank value of each node to 1/N
g.ndata['deg'] = g.out_degrees(g.nodes()).float()  # store each node’s out-degree as a node feature
# The node out_degree is the number of edges pointing out of the node.



# two UDFs
def pagerank_message_func(edges):  # the message functions are expressed as Edge UDFs
    return {'pv': edges.src['pv'] / edges.src['deg']}  # the function computes messages only from source node features


def pagerank_reduce_func(nodes):  # The reduce functions are Node UDFs which have a single argument nodes
    msgs = torch.sum(nodes.mailbox['pv'], dim=1)  # mailbox contains all incoming message features
    pv = (1 - DAMP) / N + DAMP * msgs  # computes its new PageRank value
    return {'pv': pv}


# register the 2 functions
g.register_message_func(pagerank_message_func)
g.register_reduce_func(pagerank_reduce_func)


# iterate over all the nodes
def pagerank_naive(g):
    for u, v in zip(*g.edges()):
        g.send((u, v))
    for v in g.nodes():
        g.recv(v)


# compute on a batch of nodes or edges
def pagerank_batch(g):
    g.send(g.edges())
    g.recv(g.nodes())


def pagerank_level2(g):
    g.update_all()


# use DGL builtin functions instead, faster and more efficient
def pagerank_builtin(g):
    g.ndata['pv'] = g.ndata['pv'] / g.ndata['deg']
    g.update_all(message_func=fn.copy_src(src='pv', out='m'),  # compute the output using the source node feature data
                 reduce_func=fn.sum(msg='m', out='m_sum'))  # sum the messages in the node’s mailbox
    g.ndata['pv'] = (1 - DAMP) / N + DAMP * g.ndata['m_sum']  # update


for i in range(2):
    if i == 0:
        start1 = time.time()
        for k in range(K):
            pagerank_naive(g)
            pagerank_batch(g)
            pagerank_level2(g)
        end1 = time.time()
        print(g.ndata['pv'], end1 - start1)  # 2.843s
    else:
        start2 = time.time()
        for k in range(K):
            pagerank_builtin(g)
        end2 = time.time()
        print(g.ndata['pv'], end2 - start2)  # 0.004s





