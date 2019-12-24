import dgl
import dgl.function as fn
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dgl.data import MiniGCDataset
from torch.utils.data import DataLoader


dataset = MiniGCDataset(80, 10, 20)  # 80 graphs with 10 ~ 20 nodes

graph, label = dataset[0]

fig, ax = plt.subplots()  # axis
nx.draw(graph.to_networkx(), ax=ax)
ax.set_title('Class: {:d}'.format(label))
plt.show()


# form a mini-batch: batch multiple samples together
def collate(samples):
    graphs, labels = map(list, zip(*samples))  # `samples` is a list of pairs (graph, label); graphs, labels: lists
    batched_graph = dgl.batch(graphs)  # view as a whole graph
    return batched_graph, torch.tensor(labels)  # graph, tensor


# graph convolution
msg = fn.copy_src(src='h', out='m')  # message function


def reduce(nodes):  # reduce function
    accum = torch.mean(nodes.mailbox['m'], dim=1)  # Take an average over all neighbor node features
    return {'h': accum}


class NodeApplyModule(nn.Module):  # calculate the new weight
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h': h}


class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        g.ndata['h'] = feature  # initialize the node features
        g.update_all(msg, reduce)  # message_func, reduce_func
        g.apply_nodes(func=self.apply_mod)  # apply the function on the nodes to update their features
        return g.ndata.pop('h')  # return updated features


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()

        self.layers = nn.ModuleList([
            GCN(in_dim, hidden_dim, F.relu),
            GCN(hidden_dim, hidden_dim, F.relu)])
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        # g: DGLGraph(num_nodes=432, num_edges=3040, ndata_schemes={}, edata_schemes={})
        h = g.in_degrees().view(-1, 1).float()  # in_degree == out_degree for undirected graphs
        # h.size == torch.Size([432, 1])
        for conv in self.layers:
            h = conv(g, h)  # g, feature: 'BatchedDGLGraph' object, tensor(number of in_degree for each node)
            # h.size() == torch.Size([432, 256], 256 weights for each node
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')  # (graph, feat, weight=None): averages all the values of node field [feat] in graph
        # Average over node attribute h without weighting for each graph in a batched graph
        # hg.size() == torch.Size([32, 256]), mean feature for each graph
        # self.classify(hg).size() == torch.Size([32, 8])
        return self.classify(hg)


# setup and training
trainset = MiniGCDataset(320, 10, 20)
testset = MiniGCDataset(80, 10, 20)

data_loader = DataLoader(trainset, batch_size=32, shuffle=True,
                         collate_fn=collate)  # use customized collate_fn to achieve custom batching
model = Classifier(1, 256, trainset.num_classes)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.train()

epoch_losses = []
for epoch in range(80):
    epoch_loss = 0
    for iter, (bg, label) in enumerate(data_loader):  # 320 samples, 32 in one batch, 10 batches in total
        prediction = model(bg)  # 'BatchedDGLGraph' object
        loss = loss_func(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= (iter + 1)  # iter == 9
    print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
    epoch_losses.append(epoch_loss)

plt.title('cross entropy averaged over minibatches')
plt.plot(epoch_losses)
plt.show()


model.eval()
test_X, test_Y = map(list, zip(*testset))
test_bg = dgl.batch(test_X)
test_Y = torch.tensor(test_Y).float().view(-1, 1)
probs_Y = torch.softmax(model(test_bg), 1)
# softmax is applied to all slices along dim, and will re-scale them
# so that the elements lie in the range [0, 1] and sum to 1.
sampled_Y = torch.multinomial(probs_Y, 1)
# (input, num_samples, replacement=False, out=None)
# Returns a tensor where each row contains num_samples indices sampled (non-zero and not negative ones)
argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)
# torch.max(input, dim, keepdim=False, out=None), when dim=1, return the maximum value of each row
print('Accuracy of sampled predictions on the test set: {:.4f}%'.format(
    (test_Y == sampled_Y.float()).sum().item() / len(test_Y) * 100))
print('Accuracy of argmax predictions on the test set: {:4f}%'.format(
    (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100))







