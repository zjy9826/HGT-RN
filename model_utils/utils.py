import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm 
import torch.nn as nn
import torch.nn.functional as F
import Levenshtein

def fix_positions(x, indices):
    result = torch.zeros_like(x)
    result[indices] = x[indices]
    return result

#   topology-aware loss: self_softmax
def _logsoftmax_self_cd(x,mask):

    for index,posi in enumerate(mask):
        x[index,0,:] = fix_positions(x[index,0,:],posi[0])

    # 计算每行的最大值
    row_max = torch.max(x,dim=-1,keepdim=True).values
    # print(row_max.shape)
    # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    x = x - row_max
    x_exp = x.exp()  # m * n
    partition = x_exp.sum(dim=-1, keepdim=True)  # 按列累加, m * 1
    # return torch.log((x_exp / partition))  # 广播机制, [m * n] / [m * 1] = [m * n]
    return (x_exp / partition)

# topology-aware loss: self_softmax
def _softmax_self_bj(x,mask):

    for index,posi in enumerate(mask):
        x[index,0,:] = fix_positions(x[index,0,:],posi[0])

    # 计算每行的最大值
    row_max = torch.max(x,dim=-1,keepdim=True).values
    # print(row_max.shape)
    # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    x = x - row_max
    x_exp = x.exp()  # m * n
    partition = x_exp.sum(dim=-1, keepdim=True)  # 按列累加, m * 1
    # return torch.log((x_exp / partition))  # 广播机制, [m * n] / [m * 1] = [m * n]
    return (x_exp / partition)

# topology-aware loss: self_softmax
def _softmax_self(x,mask):

    # 计算每行的最大值
    row_max = torch.max(x,dim=-1,keepdim=True).values
    # print(row_max.shape)
    # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    x = x - row_max
    x_exp = x.exp()  # m * n
    # print(f"x_exp :{x_exp}")
    x_exp_mask = x_exp*mask
    partition = x_exp_mask.sum(dim=-1, keepdim=True)  # 按列累加, m * 1
    # return torch.log((x_exp / partition))  # 广播机制, [m * n] / [m * 1] = [m * n]
    return (x_exp_mask / partition)


# topology-aware loss
def self_cross_entropy_loss(outputs, targets,mask,task):

    loss = torch.zeros(0).to("cuda")

    predicted_probs = 0
    # 计算softmax预测值
    if task=='cd':
        predicted_probs = _logsoftmax_self_cd(outputs,mask)

    if task=='bj':
        predicted_probs = _softmax_self_bj(outputs,mask)

    if task=='porto':
        predicted_probs = _softmax_self(outputs,mask)

    # 去除outputs的多余维度
    predicted_probs = torch.squeeze(predicted_probs, dim=1)
    
    # # 计算softmax预测值
    # softmax = nn.Softmax(dim=1)
    # predicted_probs = softmax(outputs)
    
    # 将targets调整为与predicted_probs相同的形状，以便索引操作
    targets = torch.squeeze(targets, dim=1)
    
    # 计算交叉熵损失
    loss = -torch.log(predicted_probs[range(targets.shape[0]), targets])
    
    # 计算平均损失
    loss = torch.mean(loss)
    
    return loss

def handle_data(inputData, train_len=None):
    # reverse the sequence
    # us_pois = [list(reversed(upois)) for upois in inputData]
    us_pois = [list(upois) for upois in inputData]
    return us_pois

class Data_modify_target(Dataset):
    def __init__(self, trainlist,node_mask, train_len=None):
        # data_node = np.array(data)[:,:,0].tolist()
        # inputs = handle_data(data_node)
        self.df = trainlist

        self.inputs = []
        self.trainX = []  # traj id: user id + traj no.
        self.train_segment_id = []
        self.train_eid = []
        self.mask = []
        for row in tqdm(trainlist):
            if(len(np.array(row)[:,2].tolist())>20):
                self.inputs.append(np.array(row)[:19,2].tolist())
                self.trainX.append(np.array(row)[:19,(2,4,5)].tolist())
                self.train_segment_id.append(np.array(row)[19:20,2].tolist())
                self.train_eid.append(np.array(row)[19:20,0].tolist())
                self.mask.append(node_mask[np.array(row)[18:19,2][0]])
            else:

                # print(np.array(row)[-1:,0].tolist())
                # print(np.array(row)[-2:-1,0].tolist())
                self.inputs.append(np.array(row)[:-1,2].tolist())
                self.trainX.append(np.array(row)[:-1,(2,4,5)].tolist())
                self.train_segment_id.append(np.array(row)[-1:,2].tolist())
                self.train_eid.append(np.array(row)[-1:,0].tolist())
                self.mask.append(node_mask[np.array(row)[-2:-1,2][0]])

        self.length = len(trainlist)
        # self.n_node = len(data_node[0])
        self.n_node = 0
    def __getitem__(self, index):
        u_input = self.inputs[index]
        node = np.unique(u_input)
        self.n_node = len(u_input)
        items = node.tolist() + (self.n_node - len(node)) * [0]
        adj = np.zeros((self.n_node, self.n_node))
        for i in np.arange(len(u_input) - 1):
            u = np.where(node == u_input[i])[0][0]
            adj[u][u] = 1
            if u_input[i + 1] == 0:
                break
            v = np.where(node == u_input[i + 1])[0][0]
            if u == v or adj[u][v] == 4:
                continue
            adj[v][v] = 1
            if adj[v][u] == 2:
                adj[u][v] = 4
                adj[v][u] = 4
            else:
                adj[u][v] = 2
                adj[v][u] = 3
        alias_inputs = [np.where(node == i)[0][0] for i in u_input]
        return (self.trainX[index], self.train_segment_id[index], self.train_eid[index],
            torch.tensor(alias_inputs), torch.tensor(adj), torch.tensor(items),self.mask[index])
    def __len__(self):
        return self.length
    


# class Data(Dataset):
#     def __init__(self, trainlist,node_mask, train_len=None):
#         # data_node = np.array(data)[:,:,0].tolist()
#         # inputs = handle_data(data_node)
#         self.df = trainlist
#
#         self.inputs = []
#         self.trainX = []  # traj id: user id + traj no.
#         self.train_segment_id = []
#         self.train_eid = []
#         self.mask = []
#         for row in tqdm(trainlist):
#             self.inputs.append(np.array(row)[:-5,2].tolist())
#             self.trainX.append(np.array(row)[:-1,(2,4,5)].tolist())
#             self.train_segment_id.append(np.array(row)[-5:-4,4].tolist())
#             self.train_eid.append(np.array(row)[-5:-4,0].tolist())
#             # print(np.array(row)[-6:-5,2][0])
#             # print(node_mask[np.array(row)[-6:-5,2][0]])
#             self.mask.append(node_mask[np.array(row)[-6:-5,2][0]])
#         # self.inputs = np.asarray(inputs)  node
#         # self.direction = np.asarray(np.array(data)[:,:,1].tolist())  dir
#         # self.taxi_id = np.asarray(np.array(data)[:,:,2].tolist())  taxi
#         # self.targets = np.asarray(targets)  dir_target
#         # self.target_eid = np.asarray(target_eid)  e_target
#         self.length = len(trainlist)
#         # self.n_node = len(data_node[0])
#         self.n_node = 0
#     def __getitem__(self, index):
#         u_input = self.inputs[index]
#         node = np.unique(u_input)
#         self.n_node = len(u_input)
#         items = node.tolist() + (self.n_node - len(node)) * [0]
#         adj = np.zeros((self.n_node, self.n_node))
#         for i in np.arange(len(u_input) - 1):
#             u = np.where(node == u_input[i])[0][0]
#             adj[u][u] = 1
#             if u_input[i + 1] == 0:
#                 break
#             v = np.where(node == u_input[i + 1])[0][0]
#             if u == v or adj[u][v] == 4:
#                 continue
#             adj[v][v] = 1
#             if adj[v][u] == 2:
#                 adj[u][v] = 4
#                 adj[v][u] = 4
#             else:
#                 adj[u][v] = 2
#                 adj[v][u] = 3
#         alias_inputs = [np.where(node == i)[0][0] for i in u_input]
#         return (self.trainX[index], self.train_segment_id[index], self.train_eid[index],
#             torch.tensor(alias_inputs), torch.tensor(adj), torch.tensor(items),self.mask[index])
#     def __len__(self):
#         return self.length
#
# # class CosineSimilarity(nn.Module):
#
# #     def forward(self, tensor_1, tensor_2):
# #         normalized_tensor_1 = tensor_1 / tensor_1.norm(dim=-1, keepdim=True)
# #         normalized_tensor_2 = tensor_2 / tensor_2.norm(dim=-1, keepdim=True)
# #         return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)


# def get_node_in_neighbor(pre_nodeid,pre_fea,node_spatial_emb,edge_dir_emb):
#     edgeid = torch.zeros(1).to(device="cuda")
#     rows = torch.where(edge_dir_emb[:, 1] == pre_nodeid)[0]
#     pre_nodeid_neighbor = edge_dir_emb[rows][:,2]
#     # print(pre_nodeid)
#     # print(edge_dir_emb[rows])
#     # print(pre_nodeid_neighbor)
#     pre_nodeid_neighbor_fea = torch.index_select(node_spatial_emb, dim=0, index=pre_nodeid_neighbor)
#
#     # print(pre_nodeid_neighbor_fea.shape)
#     # print(pre_fea.shape)
#     pre_fea = pre_fea.repeat(pre_nodeid_neighbor_fea.shape[0], 1)
#     # print(pre_fea.shape)
#
#     similarity = torch.nn.functional.cosine_similarity(pre_fea,pre_nodeid_neighbor_fea)
#     # print(similarity)
#     # print(torch.argmax(similarity))
#     edgeid = edge_dir_emb[rows][torch.argmax(similarity),0]
#     # print(edgeid)
#     # import sys
#     # sys.exit()
#
#     return edgeid

# def get_eid(pre_nodeid,direction,edge_dir_emb):
#     edgeid = torch.zeros(1).to(device="cuda")
#     for i, r in enumerate(direction):
#
#         rows = torch.where(edge_dir_emb[:, 1] == pre_nodeid)[0]
#         edge_sample = edge_dir_emb[rows]
#
#         res_shun = torch.remainder(edge_sample[:,4]-r,8)
#         res_ni = torch.remainder(r - edge_sample[:,4],8)
#         res = torch.min(res_shun, res_ni)
#
#         if(res.shape[0]==0):
#             # edgeid.append(edgeid[-1])
#             edgeid[i] = edgeid[i-1]
#         # print(res)
#         else:
#             min_index = torch.argmin(res)  # 判断方向是否符合，取最小的
#             # print(f"min_index:{min_index}")
#             pre_nodeid = edge_sample[min_index][2]
#
#             edgeid[i] = edge_sample[min_index][0]
#             # print(f"edge_sample[min_index][0]:{edge_sample[min_index][0]}")
#     return edgeid

## 计算编辑距离
# def get_DE(s1,s2):
#     # de = 0
#     str1 = [str(elem) for elem in s1.tolist()]
#     str2 = [str(elem) for elem in s2.tolist()]
#     return Levenshtein.distance(str1,str2)

# def get_MRK(y,y_pre,k):
#     # print(y)
#     # print(y_pre)
#     total = 0
#     for a,b in zip(y,y_pre):
#         temp = 0
#         for i in range(y.shape[1]):
#             if a[i]==b[i]:
#                 temp+=1
#         if temp >= k:
#             total +=5
#     return total



def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable
    

#################### topK ########## 
def top_k(loc_pred, loc_true, topk):
    """
    count the hit numbers of loc_true in topK of loc_pred, used to calculate Precision, Recall and F1-score,
    calculate the reciprocal rank, used to calcualte MRR,
    calculate the sum of DCG@K of the batch, used to calculate NDCG

    Args:
        loc_pred: (batch_size * output_dim)
        loc_true: (batch_size * 1)
        topk:

    Returns:
        tuple: tuple contains:
            hit (int): the hit numbers \n
            rank (float): the sum of the reciprocal rank of input batch \n
            dcg (float): dcg
    """
    assert topk > 0, "top-k ACC评估方法：k值应不小于1"
    # loc_pred = torch.FloatTensor(loc_pred)
    val, index = torch.topk(loc_pred, topk, 1)
    # print(index)
    # index = index.cpu().numpy()
    # print(index)
    hit = 0
    rank = 0.0
    dcg = 0.0
    # print(loc_true)
    for i, p in enumerate(index):
        target = loc_true[i]
        # print(target)
        # print(p)
        # print(target in p)
        if target in p:
            hit += 1
            rank_list = list(p)
            rank_index = rank_list.index(target)
            # rank_index is start from 0, so need plus 1
            rank += 1.0 / (rank_index + 1)
            dcg += 1.0 / np.log2(rank_index + 2)
    return hit, rank, dcg

def calculate_score(pres_raw, labels, topk):
    #print(f"the shape of top k is {pres_raw.shape}, {labels.shape}")
    hit, rank, dcg = top_k(pres_raw, labels, topk)
    total = labels.shape[0]
    recall = hit / total
    precision = hit / (total  * topk)
    if hit == 0:
        f1_score = 0
    else:
        f1_score = (2 * precision * recall) / (precision + recall)
    mrr = rank / total
    ndcg = dcg / total
    # print(recall,precision, f1_score, mrr)
    return (recall,precision, f1_score, mrr,ndcg)


# data=[
#     [
#         [2,3,1],
#         [4,1,4],
#         [2,3,5]
#     ],
#     [
#         [3,2,3],
#         [2,3,1],
#         [6,1,7]
#     ],
#     [
#         [5,3,5],
#         [1,5,6],
#         [7,6,2]
#     ],
#     [
#         [7,6,2],
#         [3,2,3],
#         [2,3,1]
#     ]
# ]
# # data=[
# #     [
# #         [1],
# #         [4],
# #         [1]
# #     ],
# #     [
# #         [3],
# #         [1],
# #         [7]
# #     ],
# #     [
# #         [5],
# #         [6],
# #         [5]
# #     ],
# #     [
# #         [2],
# #         [3],
# #         [1]
# #     ]
# # ]
# targets = [
#     [1,7],
#     [3,5],
#     [3,5],
#     [6,3]
# ]

# train_data = Data(data=data,targets=targets)
# train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=2,
#                                                shuffle=True, pin_memory=True)
# for data in tqdm(train_loader):
#     alias_inputs, adj, items, targets, inputs,r,taxi_id = data
#     alias_inputs = trans_to_cuda(alias_inputs).long()
#     items = trans_to_cuda(items).long()
#     adj = trans_to_cuda(adj).float()
#     inputs = trans_to_cuda(inputs).long()

#     print(f"alias_inputs:{alias_inputs},shape:{alias_inputs.shape}")
#     print(f"items:{items},shape:{items.shape}")
#     print(f"adj:{adj},shape:{adj.shape}")
#     print(f"inputs:{inputs},shape:{inputs.shape}")
#     print(f"r:{r},r.shape:{r.shape}")

#     local_agg = trans_to_cuda(LocalAggregator(128,0.2, dropout=0.0))
#     embedding = trans_to_cuda(nn.Embedding(8, 128))
#     h = embedding(items)
#     h_local = local_agg(h, adj)
#     h_local = F.dropout(h_local, 0)
#     # print(h_local)