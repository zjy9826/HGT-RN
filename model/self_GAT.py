import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd

class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]

        # res_att = a.to_dense

        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)
    

    
class slefGAT_Layer(nn.Module):

    def __init__(self, in_features,r_in_features, out_features, dropout, alpha, concat=True):
        super(slefGAT_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.Wr = nn.Parameter(torch.zeros(size=(r_in_features, out_features)))
        nn.init.xavier_normal_(self.Wr.data, gain=1.414)
                
        # self.a = nn.Parameter(torch.zeros(size=(1, 2*out_features)))
        self.a = nn.Parameter(torch.zeros(size=(1, 3*out_features+1)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, edge_index,edge_weight_road,edge_r):
        '''
        input:node_feature N*N
        edge_index: 2*E
        edge_r: E*8
        '''

        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        # edge = adj.nonzero().t()
        edge = edge_index


        h = torch.mm(input, self.W)

        # h: N x out
        assert not torch.isnan(h).any()

        r = torch.mm(edge_r, self.Wr)
        # r: E x out
        assert not torch.isnan(r).any()

        '''
        在每一对起点和终点之后加入方向
        '''
        edge_weight_road = edge_weight_road.unsqueeze(1)
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :], r,edge_weight_road), dim=1).t()
        # edge: 3*D+1 x E 

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out
        
        h_prime = h_prime.div(e_rowsum + torch.Tensor([9e-15]).cuda())
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class W_D_GAT(nn.Module):
    def __init__(self, nfeat,rfeat, nhid, nclass, dropout, alpha, nheads):
        super(W_D_GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [slefGAT_Layer(nfeat,
                                         rfeat,
                                         nhid,
                                         dropout=dropout,
                                         alpha=alpha,
                                         concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = slefGAT_Layer(nhid * nheads,
                                     rfeat,
                                     nclass,
                                     dropout=dropout,
                                     alpha=alpha,
                                     concat=False)

        # self.Eto8 = nn.Linear(nfeat,8)

    def forward(self, x, edge_index,edge_weight_road,edge_r):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x,edge_index,edge_weight_road,edge_r) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_att(x, edge_index,edge_weight_road,edge_r)
        x = F.elu(x)
        # res_att_map = self.Eto8(res_att_map)
        return x






# # 加载 node文件
# def load_graph_node_features(path, feature1='FID', feature2='y',feature3='x', feature4='street_cou'):
#     """X.shape: (num_node, 4), four features: FID, y x, street_cou"""
#     df = pd.read_csv(path)
#     # rlt_df = df[[feature1, feature2, feature3, feature4]]
#     rlt_df = df[[feature1]]
#     X = rlt_df.to_numpy()

#     return X   

# def get_onehot(lis,num):
#     onehot_encoded = []
#     for value in lis:
#         letter = [0 for _ in range(num)]
#         letter[value[0]] = 1
#         onehot_encoded.append(letter)
#     return np.array(onehot_encoded) 

# def get_onehot_r(lis,num):
#     onehot_encoded = []
#     for value in lis:
#         letter = [0 for _ in range(num)]
#         letter[value] = 1
#         onehot_encoded.append(letter)
#     return np.array(onehot_encoded)

# # node_onehot = load_graph_node_features("./data/node.csv")
# # node_onehot = get_onehot(node_onehot,37486)
# node_onehot = np.load("./data/node_onehot_1.npy",allow_pickle=True)
# edge_index_road = np.load("./data/edge_index_road.npy",allow_pickle=True)
# edge_r_road = np.load("./data/edge_r_road.npy",allow_pickle=True)
# print(node_onehot.shape)
# print(edge_index_road)
# print(edge_r_road)

# node_onehot = torch.tensor(node_onehot,dtype=torch.float).to(device="cuda")
# edge_index_road = torch.tensor(edge_index_road).to("cuda")
# edge_r_road_onehot = get_onehot_r(edge_r_road,8)
# edge_r_road_onehot = torch.tensor(edge_r_road_onehot,dtype=torch.float).to("cuda")

# model = W_D_GAT(37486,8,256,256,0.5,0.2,8).to("cuda")
# # 优化器
# optimizer = torch.optim.Adam(model.parameters(), 
#                        lr=0.005, 
#                        weight_decay=5e-4)
# model.train()
# optimizer.zero_grad()
# y_pre = model(node_onehot,edge_index_road,edge_r_road_onehot)
# print(y_pre.shape)
# print(y_pre)
# # loss_train.backward()
# optimizer.step()
