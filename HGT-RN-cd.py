from model_utils.utils import Data_modify_target,self_cross_entropy_loss,calculate_score,_logsoftmax_self_cd
import argparse
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter
from model_utils.dataload import get_data_roadsegment_cd
from model import LocalAggregator,hgt_rn
from sklearn.metrics import accuracy_score
from tqdm import tqdm 
from torch.optim.lr_scheduler import ExponentialLR
from model.self_GAT import W_D_GAT

import warnings
warnings.filterwarnings('ignore')

def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable
    
def get_onehot_neighbor(lis,num):
    # onehot_encoded = []
    res = torch.zeros((lis.shape[0], num)).to(device=args.device)
    for i in range(lis.shape[0]):
        res[i][lis[i]] = 1

    return res

def get_onehot(lis,num):
    # onehot_encoded = []
    res = torch.zeros((lis.shape[0], num)).to(device=args.device)
    for i in range(lis.shape[0]):
        res[i][lis[i]] = 1

    return res

def train(args):

    # % ====================== Load data ======================
    # Read  train data    
    trainlist, testlist = get_data_roadsegment_cd(file_path)

    ## build Global Graph

    ## load Weighted-Directional GAT data : X , edge_index,edge_weight, direction_node2node  ##
    node_onehot = np.load(args.node_X,allow_pickle=True)
    edge_index_road = np.load(args.edge_index,allow_pickle=True)
    edge_weight_road = np.load(args.edge_weight,allow_pickle=True)
    edge_r_road = np.load(args.edge_r,allow_pickle=True)

    # to tensor
    node_onehot = torch.tensor(node_onehot,dtype=torch.float).to(device=args.device)
    edge_index_road = torch.tensor(edge_index_road).to(args.device)
    edge_weight_road = torch.tensor(edge_weight_road,dtype=torch.float).to(args.device)
    # to one-hot vector
    edge_r_road_onehot = get_onehot(edge_r_road,8)
    edge_r_road_onehot = torch.tensor(edge_r_road_onehot,dtype=torch.float).to(args.device)

    '''
    node_onehot,edge_index_road,edge_weight_road,edge_r_road_onehot
    '''

    transition_probability_map = np.load(args.transition_probability_map,allow_pickle=True).tolist()

    node_mask = np.load(args.node_mask,allow_pickle=True).tolist()

    
    # % ====================== Define dataloader ======================
    print('Prepare dataloader...')
    train_dataset = Data_modify_target(trainlist,node_mask)
    test_dataset = Data_modify_target(testlist,node_mask)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True, drop_last=False,
                              pin_memory=True, num_workers=args.workers,
                              collate_fn=lambda x: x)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                            batch_size=args.batch_size,
                            shuffle=False, drop_last=False,
                            pin_memory=True, num_workers=args.workers,
                            collate_fn=lambda x: x)


    # % ====================== Build Models ======================
    # Model1: Global& Local node embedding model
    global_node = W_D_GAT(args.node_num, 8, 256, args.global_embed_dim, 0.5, 0.2, 2).to(args.device)
    
    local_node = LocalAggregator(dim = args.node_embed_dim, alpha = args.alpha)
    
    # % Model2: Sequence model
    seq_model = hgt_rn(args).to(device=args.device)


    # Define overall loss and optimizer
    optimizer = optim.Adam(params=list(global_node.parameters()) +
                                  list(local_node.parameters()) +
                                  list(seq_model.parameters()),
                           lr=args.lr)
    
    criterion_fun = nn.CrossEntropyLoss(ignore_index=-1).to(device=args.device)


    # % ====================== Train ======================
    global_node = global_node.to(device=args.device)
    local_node = local_node.to(device=args.device)
    seq_model = seq_model.to(device=args.device)


    for epoch in range(args.epochs): 
        global_node.train()
        local_node.train()
        seq_model.train()

        result1_train = []
        result2_train = []
        result3_train = []
        result5_train = []
        result10_train = []
        result20_train = []
        
        result1_test = []
        result2_test = []
        result3_test = []
        result5_test = []
        result10_test = []
        result20_test = []

        train_batches_loss_list = []

        # scheduler.step()

        # for step, data in enumerate(tqdm(train_loader)):
        for step, data in enumerate(train_loader):
            # alias_inputs, adj, items, inputs = data  # bachsieze,seq_len,dim 
            # global node embedding
            #  spgat
            node_spatial_emb = global_node(node_onehot,edge_index_road,edge_weight_road,edge_r_road_onehot)
            # spgat

            ############################
            y_pred,batch_tar_r,batch_tar_e,batch_pre_node,batch_mask = seq_model(data,node_spatial_emb,local_node,transition_probability_map,args)

            # 计算损失，反向传播梯度以及更新模型参数
            # 训练过程中，正向传播生成网络的输出，计算输出和实际值之间的损失值

            y_pred = y_pred.transpose(2,1)

            # print(y_pred.shape)
            # print(batch_tar_r.shape)
            # import sys
            # sys.exit()

            #计算loss
            # single_loss = criterion_fun(y_pred,batch_tar_r)
            single_loss = self_cross_entropy_loss(y_pred,batch_tar_r,batch_mask,'cd')
            # print(single_loss)
            # 清除网络先前的梯度值
            optimizer.zero_grad()
            single_loss.backward()  # 调用backward()自动生成梯度
            optimizer.step()  # 使用optimizer.step()执行优化器，把梯度传播回每个网络

            train_batches_loss_list.append(single_loss.detach().cpu().numpy())

            y_pred = y_pred.transpose(2,1)
          

            # y_pred = F.softmax(y_pred,dim=2)
            # print(batch_pre_node)
            # print(batch_mask)
            # import sys
            # sys.exit()
            # batch_mask = batch_mask.unsqueeze(1)
            y_pred = _logsoftmax_self_cd(y_pred,batch_mask)

            Predict_segmentID = torch.argmax(y_pred, axis=-1) # one-hot解码
            
            loc_pred = y_pred.squeeze(1)
            loc_true = batch_tar_r
            # print(loc_pred.shape)
            # print(loc_true.shape)
            top1 = list(calculate_score(loc_pred,loc_true,1))
            top2 = list(calculate_score(loc_pred,loc_true,2))
            top3 = list(calculate_score(loc_pred,loc_true,3))
            top5 = list(calculate_score(loc_pred,loc_true,5))
            top10 = list(calculate_score(loc_pred,loc_true,10))
            top20 = list(calculate_score(loc_pred,loc_true,20))

            result1_train.append(top1) 
            result2_train.append(top2)
            result3_train.append(top3)
            result5_train.append(top5) 
            result10_train.append(top10)
            result20_train.append(top20)

           

            # y_list_train = y_list_train + y_eid.cpu().numpy().flatten().tolist()
            # pre_list_train = pre_list_train + e.flatten().tolist()

            if(step==0):
                y_list_train = batch_tar_r.reshape(-1)
                pre_list_train = Predict_segmentID.reshape(-1)
            else:
                y_list_train = torch.cat([y_list_train,batch_tar_r.reshape(-1)])
                pre_list_train = torch.cat([pre_list_train,Predict_segmentID.reshape(-1)])

        
        # 计算正确率
        # rate = rightNum / all_num  # 训练集
        rate = accuracy_score(y_list_train.cpu(),pre_list_train.cpu())


        
        # train end --------------------------------------------------------------------------------------------------------

        val_batches_loss_list = []

        global_node.eval()
        local_node.eval()
        seq_model.eval()

        # for step, data in enumerate(tqdm(test_loader)):
        for step, data in enumerate(test_loader):

            # global node embedding
            #  spgat
            node_spatial_emb = global_node(node_onehot,edge_index_road,edge_weight_road,edge_r_road_onehot)
            # spgat
            ############################
            y_pred,batch_tar_r,batch_tar_e,batch_pre_node,batch_mask = seq_model(data,node_spatial_emb,local_node,transition_probability_map,args)

            # 计算 test loss 
            y_pred = y_pred.transpose(2,1)
            # single_loss = criterion_fun(y_pred,batch_tar_r)
            single_loss = self_cross_entropy_loss(y_pred,batch_tar_r,batch_mask,'cd')
            val_batches_loss_list.append(single_loss.detach().cpu().numpy())
            y_pred = y_pred.transpose(2,1)


            
            # y_pred = F.softmax(y_pred,dim=2)
            y_pred = _logsoftmax_self_cd(y_pred,batch_mask)


            testYPredict_segmentID = torch.argmax(y_pred, axis=-1) # one-hot解码
            # print(testYPredict_segmentID.shape)

            loc_pred = y_pred.squeeze(1)
            loc_true = batch_tar_r
            # print(loc_pred.shape)
            # print(loc_true.shape)
            top1 = list(calculate_score(loc_pred,loc_true,1))
            top2 = list(calculate_score(loc_pred,loc_true,2))
            top3 = list(calculate_score(loc_pred,loc_true,3))
            top5 = list(calculate_score(loc_pred,loc_true,5))
            top10 = list(calculate_score(loc_pred,loc_true,10))
            top20 = list(calculate_score(loc_pred,loc_true,20))

            result1_test.append(top1) 
            result2_test.append(top2)
            result3_test.append(top3)
            result5_test.append(top5) 
            result10_test.append(top10)
            result20_test.append(top20)
            
            # print(f"x_batch :{X_batch[:,9,0].shape}")  # torch.Size([20])
            # print(f"Predict_segmentID :{testYPredict_segmentID.shape}")  # torch.Size([20, 5])
    

            if(step==0):
                y_list_test = batch_tar_r.reshape(-1)
                pre_list_test = testYPredict_segmentID.reshape(-1)
            else:
                y_list_test = torch.cat([y_list_test,batch_tar_r.reshape(-1)])
                pre_list_test = torch.cat([pre_list_test,testYPredict_segmentID.reshape(-1)])

        # 计算正确率
    
        acc = accuracy_score(y_list_test.cpu(),pre_list_test.cpu())  # 准确率相当于 AMR  

        # valid end --------------------------------------------------------------------------------------------------------

        # epoch_val_loss = np.mean(val_batches_loss_list)
        # # Monitor loss and score
        # monitor_loss = epoch_val_loss
        # # Learning rate schuduler
        # lr_scheduler.step(monitor_loss)

        result1_train = np.array(result1_train)
        result2_train = np.array(result2_train)
        result3_train = np.array(result3_train)
        result5_train = np.array(result5_train)
        result10_train = np.array(result10_train)
        result20_train = np.array(result20_train)

        result1_test = np.array(result1_test)
        result2_test = np.array(result2_test)
        result3_test = np.array(result3_test)
        result5_test = np.array(result5_test)
        result10_test = np.array(result10_test)
        result20_test = np.array(result20_test)
        
        print(f'==============================epoch：{epoch} start =============================================')
        print(f"train_loss : {np.mean(train_batches_loss_list)},acc_train：{rate}")
        print(f"recall,precision, f1_score, mrr,ndcg @1:{round(np.mean(result1_train[:,0]),5),round(np.mean(result1_train[:,1]),5),round(np.mean(result1_train[:,2]),5),round(np.mean(result1_train[:,3]),5),round(np.mean(result1_train[:,4]),5)}")
        print(f"recall,precision, f1_score, mrr,ndcg @2:{round(np.mean(result2_train[:,0]),5),round(np.mean(result2_train[:,1]),5),round(np.mean(result2_train[:,2]),5),round(np.mean(result2_train[:,3]),5),round(np.mean(result2_train[:,4]),5)}")
        print(f"recall,precision, f1_score, mrr,ndcg @3:{round(np.mean(result3_train[:,0]),5),round(np.mean(result3_train[:,1]),5),round(np.mean(result3_train[:,2]),5),round(np.mean(result3_train[:,3]),5),round(np.mean(result3_train[:,4]),5)}")
        print(f"recall,precision, f1_score, mrr,ndcg @5:{round(np.mean(result5_train[:,0]),5),round(np.mean(result5_train[:,1]),5),round(np.mean(result5_train[:,2]),5),round(np.mean(result5_train[:,3]),5),round(np.mean(result5_train[:,4]),5)}")
        print(f"recall,precision, f1_score, mrr,ndcg @10:{round(np.mean(result10_train[:,0]),5),round(np.mean(result10_train[:,1]),5),round(np.mean(result10_train[:,2]),5),round(np.mean(result10_train[:,3]),5),round(np.mean(result10_train[:,4]),5)}")
        print(f"recall,precision, f1_score, mrr,ndcg @20:{round(np.mean(result20_train[:,0]),5),round(np.mean(result20_train[:,1]),5),round(np.mean(result20_train[:,2]),5),round(np.mean(result20_train[:,3]),5),round(np.mean(result20_train[:,4]),5)}")
        print(f"test_loss:{np.mean(val_batches_loss_list)},test_acc:{acc}")
        print(f"recall,precision, f1_score, mrr,ndcg @1:{round(np.mean(result1_test[:,0]),5),round(np.mean(result1_test[:,1]),5),round(np.mean(result1_test[:,2]),5),round(np.mean(result1_test[:,3]),5),round(np.mean(result1_test[:,4]),5)}")
        print(f"recall,precision, f1_score, mrr,ndcg @2:{round(np.mean(result2_test[:,0]),5),round(np.mean(result2_test[:,1]),5),round(np.mean(result2_test[:,2]),5),round(np.mean(result2_test[:,3]),5),round(np.mean(result2_test[:,4]),5)}")
        print(f"recall,precision, f1_score, mrr,ndcg @3:{round(np.mean(result3_test[:,0]),5),round(np.mean(result3_test[:,1]),5),round(np.mean(result3_test[:,2]),5),round(np.mean(result3_test[:,3]),5),round(np.mean(result3_test[:,4]),5)}")
        print(f"recall,precision, f1_score, mrr,ndcg @5:{round(np.mean(result5_test[:,0]),5),round(np.mean(result5_test[:,1]),5),round(np.mean(result5_test[:,2]),5),round(np.mean(result5_test[:,3]),5),round(np.mean(result5_test[:,4]),5)}")
        print(f"recall,precision, f1_score, mrr,ndcg @10:{round(np.mean(result10_test[:,0]),5),round(np.mean(result10_test[:,1]),5),round(np.mean(result10_test[:,2]),5),round(np.mean(result10_test[:,3]),5),round(np.mean(result10_test[:,4]),5)}")
        print(f"recall,precision, f1_score, mrr,ndcg @20:{round(np.mean(result20_test[:,0]),5),round(np.mean(result20_test[:,1]),5),round(np.mean(result20_test[:,2]),5),round(np.mean(result20_test[:,3]),5),round(np.mean(result20_test[:,4]),5)}")
        print(f'==============================epoch：{epoch}  end =============================================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    file_path = "./CD-data"
    # GAT model input: X / edge list / edge weight / Orientation feature between two nodes
    parser.add_argument('--node_X',default= file_path + '/x_onehot.npy')
    parser.add_argument('--edge_index', default= file_path +'/edge_list_train.npy')
    parser.add_argument('--edge_weight',default= file_path +'/edge_weight_train_min-max-cd.npy')
    parser.add_argument('--edge_r',default= file_path +'/edge_r_train.npy')
    #  probability transition vector
    parser.add_argument('--transition_probability_map',default= file_path + '/transition_probability_map_node2node_cd_dic.npy')
    # Road network constraints: node mask,  indicating whether the next intersection at the current intersection exists in the road network
    parser.add_argument('--node_mask',default= file_path + '/node_mask_node2node_cd_dic.npy')

    # embedding
    parser.add_argument('--node_num', type=int, default=11382)
    parser.add_argument('--num_taxi', type=int, default=2001)
    parser.add_argument('--num_dir', type=int, default=8)

    parser.add_argument('--node_embed_dim', type=int, default=256)
    parser.add_argument('--taxi_embed_dim', type=int, default=32)
    parser.add_argument('--dir_embed_dim', type=int, default=256)
    parser.add_argument('--global_embed_dim', type=int, default=256)
    parser.add_argument('--local_embed_dim', type=int, default=256)

    # model
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--seq_encoder_input_embed', type=int, default=0)
    parser.add_argument('--device',type=str,default="cuda",help='')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr',type=float,default=0.0001,help='Initial learning rate.')
    parser.add_argument('--lr-scheduler-factor',type=float,default=0.1,help='Learning rate scheduler factor')
    parser.add_argument('--dropout_global', type=float, default=0.5, help='Dropout rate.')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')

    args = parser.parse_args(args=[])

    train(args)
