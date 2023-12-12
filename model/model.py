import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import GATConv
from torch.nn.utils.rnn import pad_sequence
import numpy as np

# node
# class NodeEmbeddings(nn.Module):
#     def __init__(self, num_users, embedding_dim):
#         super(NodeEmbeddings, self).__init__()
#
#         self.node_embedding = nn.Embedding(
#             num_embeddings=num_users,
#             embedding_dim=embedding_dim,
#         )
#
#     def forward(self, node_num):
#         embed = self.node_embedding(node_num)
#         return embed
#
# # taxi_id--user
# class UserEmbeddings(nn.Module):
#     def __init__(self, num_users, embedding_dim):
#         super(UserEmbeddings, self).__init__()
#
#         self.user_embedding = nn.Embedding(
#             num_embeddings=num_users,
#             embedding_dim=embedding_dim,
#         )
#
#     def forward(self, user_idx):
#         embed = self.user_embedding(user_idx)
#         return embed
# # direction embedding
# class DirectionEmbeddings(nn.Module):
#     def __init__(self, num_cats, embedding_dim):
#         super(DirectionEmbeddings, self).__init__()
#
#         self.dir_embedding = nn.Embedding(
#             num_embeddings=num_cats,
#             embedding_dim=embedding_dim,
#         )
#
#     def forward(self, dir_idx):
#         embed = self.dir_embedding(dir_idx)
#         return embed




class LocalAggregator(nn.Module):
    def __init__(self, dim, alpha, dropout=0., name=None):
        super(LocalAggregator, self).__init__()
        self.dim = dim
        self.dropout = dropout

        self.a_0 = nn.Linear(self.dim, 1)
        self.a_1 = nn.Linear(self.dim, 1)
        self.a_2 = nn.Linear(self.dim, 1)
        self.a_3 = nn.Linear(self.dim, 1)

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, hidden, adj, mask_item=None):
        h = hidden
        # batch_size = h.shape[0]
        N = h.shape[0]

        a_input = (h.repeat(1, 1, N).view( N * N, self.dim)
                   * h.repeat(1, N, 1)).view( N, N, self.dim)

        e_0 = self.a_0(a_input)
        e_1 = self.a_1(a_input)
        e_2 = self.a_2(a_input)
        e_3 = self.a_3(a_input)
        

        e_0 = self.leakyrelu(e_0).squeeze(-1).view( N, N)
        e_1 = self.leakyrelu(e_1).squeeze(-1).view( N, N)
        e_2 = self.leakyrelu(e_2).squeeze(-1).view(N, N)
        e_3 = self.leakyrelu(e_3).squeeze(-1).view( N, N)

        mask = -9e15 * torch.ones_like(e_0)
        alpha = torch.where(adj.eq(1), e_0, mask)
        alpha = torch.where(adj.eq(2), e_1, alpha)
        alpha = torch.where(adj.eq(3), e_2, alpha)
        alpha = torch.where(adj.eq(4), e_3, alpha)
        alpha = torch.softmax(alpha, dim=-1)

        output = torch.matmul(alpha, h)
        return output


# 嵌入表征融合 
class FuseEmbeddings(nn.Module):
    def __init__(self, global_embed_dim, s_embed_dim):
        super(FuseEmbeddings, self).__init__()
        embed_dim = global_embed_dim + s_embed_dim
        self.fuse_embed = nn.Linear(embed_dim, global_embed_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, wh, s):
        # print(f"user_embed:{user_embed.shape}")
        # print(f"poi_embed:{poi_embed.shape}")
        x = self.fuse_embed(torch.cat((wh, s), -1))
        x = self.leaky_relu(x)
        return x

def get_dir_taxi_onehot(lis,num):
    res = torch.zeros((len(lis),num),device="cuda",dtype=torch.float32)
    for i in range(res.shape[0]):
        res[i][lis[i]] = 1
            
    
    return res


#  位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)  


# HGT-RN model
class hgt_rn(nn.Module):
    def __init__(self,args):
        super().__init__()

        # 嵌入层
        # self.node_emb = nn.Linear(37486,M)
        # self.dir_emb = nn.Linear(8,D)
        # self.taxiid_emb = nn.Linear(10356,32)

        self.node_emb = nn.Parameter(torch.zeros(size=(args.node_num, args.node_embed_dim)))
        nn.init.xavier_normal_(self.node_emb.data, gain=1.414)

        self.dir_emb = nn.Parameter(torch.zeros(size=(args.num_dir, args.dir_embed_dim)))
        nn.init.xavier_normal_(self.dir_emb.data, gain=1.414)

        self.taxiid_emb = nn.Parameter(torch.zeros(size=(args.num_taxi, args.taxi_embed_dim)))
        nn.init.xavier_normal_(self.taxiid_emb.data, gain=1.414)

        M = args.node_embed_dim + args.dir_embed_dim + args.local_embed_dim + args.global_embed_dim
        # M = args.node_embed_dim + args.dir_embed_dim + args.local_embed_dim

        self.FuseEmbeddings = FuseEmbeddings(args.global_embed_dim,args.node_embed_dim)

         # 定义位置编码器
        self.positional_encoding = PositionalEncoding(M,dropout=0)

        # 定义Transformer
        self.transformer = nn.Transformer(M,nhead=8 ,num_encoder_layers=2,num_decoder_layers=1,dim_feedforward=2048,dropout=0.3,batch_first=True)

        self.fc = nn.Linear(M+args.taxi_embed_dim, args.node_num)
        # self.crition = nn.CrossEntropyLoss()

    def forward(self, batch,node_spatial_emb,local_node,transition_probability_map,args):
        # alias_inputs, adj, items, targets, inputs
        batch_seq_embeds_enc = []
        batch_seq_embeds_dec = []
        batch_taxi_embeds = []
        batch_seq_len = []  # 每个的真实长度

        batch_tar_r = []
        batch_tar_e = []
        batch_pre_node = []
        batch_mask = []

        batch_seq_lens = []
        # print(f"batch:{batch}")
        batch.sort(key=lambda item:len(item[0]),reverse=True)
        # print(f"batch:{batch}")

        for sample in batch:
            # samlle[0]: len*3（节点，方向，出租车）
            # samplep[1]: target 方向/node
            # sample[2]: target 路段
            # sample[3] :alias_inputs, 
            # sample[4] : adj, 
            # sample[5] : items 
            # sample[6] : inputs  node_traj
            # print(sample[0])
            # print([each[0] for each in sample[0]])

            # 得到路网约束mask
            batch_mask.append(sample[6])

            batch_tar_r.append(sample[1])
            batch_tar_e.append(sample[2])
            
            node_traj = [each[0] for each in sample[0]]
            r_traj = [each[1] for each in sample[0]]
            taxiid = sample[0][0][2]-1

            # print(f"node_traj:{node_traj}")
            # print(f"r_traj:{r_traj}")
            # print(f"taxiid:{taxiid}")

            # 获取倒数第5个
            batch_pre_node.append(node_traj[-1])
            # 从 p map 中找 概率
            batch_seq_lens.append(node_traj[-1])

            
            # 得到反向位置编码
            # input_sequence = torch.flip(torch.tensor(node_traj[:-4],device=args.device), [0])
            # reverse_pos_emb = self.reverse_pos(input_sequence)

            ## 嵌入层 进行node，方向dir嵌入
            node_one_hot = get_dir_taxi_onehot(node_traj,args.node_num)
            # node_emb = self.node_emb(node_one_hot)
            node_emb = torch.matmul(node_one_hot, self.node_emb)
        

            dir_one_hot = get_dir_taxi_onehot(r_traj,args.num_dir)
            # dir_emb = self.dir_emb(dir_one_hot)
            dir_emb = torch.matmul(dir_one_hot, self.dir_emb)


            local_input_one_hot = get_dir_taxi_onehot(sample[5],args.node_num)
            local_input = torch.matmul(local_input_one_hot, self.node_emb)
            adj = sample[4].to("cuda")
            local_node_embedding = local_node(local_input,adj)
            local_embedding = local_node_embedding[sample[3]]
            # print(f"local_embedding.shape:{local_embedding.shape}")

            '''# spgat'''
            wh = torch.index_select(node_spatial_emb, dim=0, index=torch.tensor(node_traj,device=args.device))
            s = torch.mean(node_emb,dim=0).repeat(wh.shape[0],1)
            # print(f"wh.shape:{wh.shape}")
            # print(f"s.shape:{s.shape}")
            wh = self.FuseEmbeddings(wh,s)
            
            '''# spgat'''

            """
            global local graph embedbing +
            """
            # wh = wh + local_embedding

            node_dir_wh_emb = torch.cat((node_emb,dir_emb,local_embedding,wh),dim=-1)
            # node_dir_wh_emb = torch.cat((node_emb,dir_emb,local_embedding),dim=-1)

            seq_len_truth = node_dir_wh_emb.shape[0]
            batch_seq_len.append(seq_len_truth)

            batch_seq_embeds_enc.append(node_dir_wh_emb[:,:]) 
            batch_seq_embeds_dec.append(node_dir_wh_emb[-1,:])

            # 出租车id
            taxiid_onehot = torch.zeros(args.num_taxi,device=args.device)
            taxiid_onehot[taxiid] = 1
            taxiid_onehot = torch.tensor(taxiid_onehot,device=args.device,dtype=torch.float32)
            # taxiid_emb = self.taxiid_emb(taxiid_onehot)
            taxiid_emb = torch.matmul(taxiid_onehot,self.taxiid_emb)
            # taxiid_emb = self.taxiid_emb(torch.tensor(taxiid,dtype=torch.long,device=device))
            batch_taxi_embeds.append(taxiid_emb)
       
        # print(batch_seq_len)
        batch_seq_embeds_enc_padded = pad_sequence(batch_seq_embeds_enc, batch_first=True, padding_value=-1)
        batch_seq_embeds_dec_padded = pad_sequence(batch_seq_embeds_dec, batch_first=True, padding_value=-1).unsqueeze(1)
        batch_taxi_embeds_padded = pad_sequence(batch_taxi_embeds, batch_first=True, padding_value=-1).unsqueeze(1)

        # Feedforward
        batch_seq_embeds_enc_padded = batch_seq_embeds_enc_padded.to(device=args.device, dtype=torch.float)
        batch_seq_embeds_dec_padded = batch_seq_embeds_dec_padded.to(device=args.device, dtype=torch.float)

        batch_tar_r = torch.tensor(batch_tar_r).to(device=args.device, dtype=torch.long)
        batch_tar_e = torch.tensor(batch_tar_e).to(device=args.device, dtype=torch.long)
        batch_pre_node = torch.tensor(batch_pre_node).to(device=args.device, dtype=torch.long)
        # print(batch_mask)
        # batch_mask = torch.tensor(batch_mask).to(device=args.device, dtype=torch.long)


        src = batch_seq_embeds_enc_padded
        tgt = batch_seq_embeds_dec_padded

        # 给src和tgt的token增加位置信息
        src = self.positional_encoding(src).to(device=args.device,dtype=torch.float32) 
        tgt = self.positional_encoding(tgt).to(device=args.device,dtype=torch.float32)

        def get_key_padding_mask(src_mask):

            src_mask[src_mask == -1] = -torch.inf
            return src_mask


        tgt_mask = nn.Transformer.generate_square_subsequent_mask(1).to(args.device,dtype=torch.float32)
 
        maxlen = max(batch_seq_len)
        src_mask = torch.full((len(batch_seq_len),maxlen),-1).to(device=args.device,dtype=torch.float32)

        for index,item in enumerate(batch_seq_len):
            src_mask[index][:item] = 0
       


        src_mask = get_key_padding_mask(src_mask)
        # print(f"src_mask:{src_mask}")
        # import sys
        # sys.exit()

        # 将准备好的数据送给transformer
        out = self.transformer(src, tgt,
                               tgt_mask=tgt_mask,
                               src_key_padding_mask=src_mask,
                               tgt_key_padding_mask=None)
        
        outputs = torch.cat((out,batch_taxi_embeds_padded),dim=-1)
        # print(f"output :{outputs.shape}")   # torch.Size([256, 5, 8])  bs/len/outdim

        outputs = self.fc(outputs)
        outputs = F.dropout(outputs,0.2,training=self.training)
        # print(outputs.shape)
        if(type(transition_probability_map)==dict):
            outputs = self._adjust_pred_prob_by_graph_usedict(batch_seq_lens, outputs,transition_probability_map)
        else:
            outputs = self._adjust_pred_prob_by_graph(batch_seq_lens, outputs,transition_probability_map)
        # print(outputs.shape)

        return outputs,batch_tar_r,batch_tar_e,batch_pre_node,batch_mask
    
    def init_weights(self):
        initrange = 0.1
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

    
    ###########加上GAT中的注意力分数  node2dir ########################
    def _adjust_pred_prob_by_graph(self,batch_seq_lens,y_pred_poi,transition_probability_map):
            # y_pred_poi_adjusted = torch.zeros_like(y_pred_poi)
            for i in range(len(batch_seq_lens)): # batch level
                traj_i_input = batch_seq_lens[i]  # list of input check-in pois each batch （5）
                # print(traj_i_input)
                # for j in range(len(traj_i_input)): #
                #     traj_node = traj_i_input[j]
                #     # print(traj_node)
                if traj_i_input in transition_probability_map:
                    traj_node_p = torch.tensor(transition_probability_map[traj_i_input]).to("cuda")
                    y_pred_poi[i, 0, :] = y_pred_poi[i, 0, :] + traj_node_p
            return y_pred_poi
    
    def _adjust_pred_prob_by_graph_usedict(self,batch_seq_lens,y_pred_poi,transition_probability_map):
            for i in range(len(batch_seq_lens)): # batch level
                traj_i_input = batch_seq_lens[i]  # list of input check-in pois each batch （5）

                if traj_i_input in transition_probability_map:
                    # print(transition_probability_map[traj_i_input])     
                    for (key,value) in transition_probability_map[traj_i_input].items():
                        y_pred_poi[i, 0, key] = y_pred_poi[i, 0, key] + value


            return y_pred_poi
    