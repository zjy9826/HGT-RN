import torch.utils.data as data
import networkx as nx
import numpy as np
import pandas as pd 


def get_data_roadsegment_Porto():
    # 加载数据

    trainlist = np.load("./Porto-data/Porto_train.npy",allow_pickle=True)
    testlist = np.load("./Porto-data/Porot_test.npy",allow_pickle=True)

    return trainlist, testlist

def get_data_roadsegment_bj(file_path):
    # 加载数据
    trainlist = np.load(file_path + "/id_ndoe_dir_traj_train.npy",allow_pickle=True)
    testlist = np.load(file_path + "/id_ndoe_dir_traj_test.npy",allow_pickle=True)

    return trainlist, testlist

def get_data_roadsegment_cd(file_path):
    # 加载数据
    trainlist = np.load(file_path + "/id_ndoe_dir_traj_train.npy",allow_pickle=True)
    testlist = np.load(file_path + "/id_ndoe_dir_traj_test.npy",allow_pickle=True)

    return trainlist, testlist
