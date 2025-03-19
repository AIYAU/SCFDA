import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

class Task(object):
    def __init__(self, data, num_classes, shot_num, query_num):
        self.data = data
        self.num_classes = num_classes
        self.support_num = shot_num
        self.query_num = query_num

        class_folders = sorted(list(data))
        class_list = random.sample(class_folders, self.num_classes)
        labels = np.array(range(len(class_list)))
        labels = dict(zip(class_list, labels))
        samples = dict()

        self.support_datas = []
        self.query_datas = []
        self.support_labels = []
        self.query_labels = []

        self.support_real_labels = []
        self.query_real_labels = []
        for c in class_list:
            temp = self.data[c]  # list
            samples[c] = random.sample(temp, len(temp))
            random.shuffle(samples[c])

            self.support_datas += samples[c][:shot_num]
            self.query_datas += samples[c][shot_num:shot_num + query_num]

            self.support_labels += [labels[c] for i in range(shot_num)]
            self.query_labels += [labels[c] for i in range(query_num)]

            self.support_real_labels += [c for i in range(shot_num)]
            self.query_real_labels += [c for i in range(query_num)]

class FewShotDataset(Dataset):
    def __init__(self, task, split='train'):
        self.task = task
        self.split = split
        self.image_datas = self.task.support_datas if self.split == 'train' else self.task.query_datas
        self.labels = self.task.support_labels if self.split == 'train' else self.task.query_labels

    def __len__(self):
        return len(self.image_datas)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")

class HBKC_dataset(FewShotDataset):
    def __init__(self, *args, **kwargs):
        super(HBKC_dataset, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image = self.image_datas[idx]
        label = self.labels[idx]
        return image, label

class ClassBalancedSampler(Sampler):
    def __init__(self, num_per_class, num_cl, num_inst,shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        else:
            batch = [[i+j*self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1

def get_HBKC_data_loader(task, num_per_class=1, split='train',shuffle = False):
    dataset = HBKC_dataset(task, split=split)
    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.support_num, shuffle=shuffle) # support set
    else:
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.query_num, shuffle=shuffle) # query set
    loader = DataLoader(dataset, batch_size=num_per_class*task.num_classes, sampler=sampler)
    return loader

class MetaTrainLabeledDataset(Dataset):
    def __init__(self, image_datas, image_labels):
        self.image_datas = image_datas
        self.image_labels = image_labels

    def __len__(self):
        return len(self.image_datas)

    def __getitem__(self, idx):
        image = self.image_datas[idx]
        label = self.image_labels[idx]
        return image, label



from . import utils, data_augment
import math

def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, tar_lsample_num_per_class, shot_num_per_class, HalfWidth):
    """
    Data_Band_Scaler:光谱带数据
    GroundTruth:光谱带标签
    class_num:类别数量
    tar_lsample_num_per_class:每个类别采样的数量
    shot_num_per_class:       每个类别采样的数量
    HalfWidth
    """
    print(Data_Band_Scaler.shape) # 打印光谱shape
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape # 解包光谱数据

    '''label start'''
    num_class = int(np.max(GroundTruth)) # 获取类别数量
    data_band_scaler = utils.flip(Data_Band_Scaler) # 增强高光谱数据
    groundtruth = utils.flip(GroundTruth)
    del Data_Band_Scaler
    del GroundTruth
    # 从翻转或填充后的数据中，提取原始数据位置的中心区域，考虑到窗口的半宽度。
    G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth]
    data = data_band_scaler[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth, :]

    [Row, Column] = np.nonzero(G) #获取非零标签的位置索引
    del data_band_scaler
    del groundtruth

    nSample = np.size(Row) # 有标签样本的总数量
    print('number of sample', nSample)

    # Sampling samples 初始化字典
    train = {}
    test = {}
    da_train = {}  # Data Augmentation 数据增强训练样本
    m = int(np.max(G)) # 获取最大类别标签索引，类别数量
    nlabeled = tar_lsample_num_per_class
    print('labeled number per class:', nlabeled)
    print((200 - nlabeled) / nlabeled + 1) #计算数据增强的重复次数
    print(math.ceil((200 - nlabeled) / nlabeled) + 1)

    for i in range(m): # 对于每个类别都分配数据
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if G[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices) # 打乱样本的采样顺序
        nb_val = shot_num_per_class # 每类别实际采样的训练样本数量
        train[i] = indices[:nb_val]  #训练样本索引
        da_train[i] = []
        for j in range(math.ceil((200 - nlabeled) / nlabeled) + 1): # 数据增强，重复采样
            da_train[i] += indices[:nb_val]
        test[i] = indices[nb_val:] #调试样本索引
    # 将所有类别的样本索引合并为单个列表，用于后续处理
    train_indices = []
    test_indices = []
    da_train_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
        da_train_indices += da_train[i]
    np.random.shuffle(test_indices) # 打乱测试样本索引

    print('the number of train_indices:', len(train_indices))
    print('the number of test_indices:', len(test_indices))
    print('the number of train_indices after data argumentation:', len(da_train_indices))  # 520
    print('labeled sample indices:', train_indices)

    nTrain = len(train_indices)
    nTest = len(test_indices)
    da_nTrain = len(da_train_indices)

    imdb = {}
    imdb['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTrain + nTest],
                            dtype=np.float32)
    imdb['Labels'] = np.zeros([nTrain + nTest], dtype=np.int64)
    imdb['set'] = np.zeros([nTrain + nTest], dtype=np.int64)

    RandPerm = train_indices + test_indices # 合并训练和测试样本索引

    RandPerm = np.array(RandPerm)

    for iSample in range(nTrain + nTest):
        # 提取以样本为中心的窗口数据
        imdb['data'][:, :, :, iSample] = data[
                                         Row[RandPerm[iSample]] - HalfWidth : Row[RandPerm[iSample]] + HalfWidth + 1,
                                         Column[RandPerm[iSample]] - HalfWidth : Column[RandPerm[iSample]] + HalfWidth + 1,
                                         :]
        # 获取样本的标签
        imdb['Labels'][iSample] = G[Row[RandPerm[iSample]], Column[RandPerm[iSample]]].astype(np.int64)

    imdb['Labels'] = imdb['Labels'] - 1
    imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)
    print('Data is OK.')

    train_dataset = utils.matcifar(imdb, train=True, d=3, medicinal=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=class_num * shot_num_per_class, shuffle=False)
    del train_dataset

    test_dataset = utils.matcifar(imdb, train=False, d=3, medicinal=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)
    del test_dataset
    del imdb

    # Data Augmentation for target domain for training
    # 目标域训练数据的数据增强
    imdb_da_train = {}
    imdb_da_train['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, da_nTrain],
                                     dtype=np.float32)
    imdb_da_train['Labels'] = np.zeros([da_nTrain], dtype=np.int64)
    imdb_da_train['set'] = np.zeros([da_nTrain], dtype=np.int64)

    da_RandPerm = np.array(da_train_indices)
    for iSample in range(da_nTrain):
        # 对样本进行辐射噪声的数据增强
        imdb_da_train['data'][:, :, :, iSample] = data_augment.radiation_noise(
            data[Row[da_RandPerm[iSample]] - HalfWidth: Row[da_RandPerm[iSample]] + HalfWidth + 1,
            Column[da_RandPerm[iSample]] - HalfWidth: Column[da_RandPerm[iSample]] + HalfWidth + 1, :])
        imdb_da_train['Labels'][iSample] = G[Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]].astype(np.int64)
    imdb_da_train['Labels'] = imdb_da_train['Labels'] - 1 # 调整标签
    imdb_da_train['set'] = np.ones([da_nTrain]).astype(np.int64) # 标记为训练集
    print('ok')

    return train_loader, test_loader, imdb_da_train, G, RandPerm, Row, Column, nTrain


def get_target_dataset(Data_Band_Scaler, GroundTruth, class_num, tar_lsample_num_per_class, shot_num_per_class, patch_size):
    train_loader, test_loader, imdb_da_train, G, RandPerm, Row, Column, nTrain = get_train_test_loader(
        Data_Band_Scaler=Data_Band_Scaler,
        GroundTruth=GroundTruth,
        class_num=class_num,
        tar_lsample_num_per_class=tar_lsample_num_per_class,
        shot_num_per_class=shot_num_per_class,
        HalfWidth=patch_size // 2)
    train_datas, train_labels = next(iter(train_loader))
    print('train labels:', train_labels)
    print('size of train datas:', train_datas.shape)

    print(imdb_da_train.keys())
    print(imdb_da_train['data'].shape)
    print(imdb_da_train['Labels'])
    del Data_Band_Scaler, GroundTruth

    # target data with data augmentation 对目标数据做增强
    target_da_datas = np.transpose(imdb_da_train['data'], (3, 2, 0, 1))  # 换坐标轴 (9,9,103, 1800)->(1800, 103, 9, 9)
    print(target_da_datas.shape)
    target_da_labels = imdb_da_train['Labels']
    print('target data augmentation label:', target_da_labels)

    target_da_train_set = {}
    target_aug_data_ssl = []
    target_aug_label_ssl = []

    for class_, path in zip(target_da_labels, target_da_datas):
        if class_ not in target_da_train_set:
            target_da_train_set[class_] = []
        target_da_train_set[class_].append(path)
        target_aug_data_ssl.append(path)
        target_aug_label_ssl.append(class_)
    target_da_metatrain_data = target_da_train_set
    print(target_da_metatrain_data.keys())

    return train_loader, test_loader, target_da_metatrain_data, G, RandPerm, Row, Column, nTrain, target_aug_data_ssl, target_aug_label_ssl

def get_target_dataset_houston(Data_Band_Scaler, GroundTruth_train, GroundTruth_test, class_num, tar_lsample_num_per_class, shot_num_per_class, patch_size):
    train_loader, _, imdb_da_train, _, _, _, _, _ = get_train_test_loader(
        Data_Band_Scaler=Data_Band_Scaler,
        GroundTruth=GroundTruth_train,
        class_num=class_num,
        tar_lsample_num_per_class=tar_lsample_num_per_class,
        shot_num_per_class=shot_num_per_class,
        HalfWidth=patch_size // 2)
    test_loader, G, RandPerm, Row, Column, nTrain = get_alltest_loader(
        Data_Band_Scaler=Data_Band_Scaler,
        GroundTruth=GroundTruth_test,
        class_num=class_num,
        shot_num_per_class=0,
        HalfWidth=patch_size // 2)


    train_datas, train_labels = next(iter(train_loader))
    print('train labels:', train_labels)
    print('size of train datas:', train_datas.shape)

    print(imdb_da_train.keys())
    print(imdb_da_train['data'].shape)
    print(imdb_da_train['Labels'])
    del Data_Band_Scaler, GroundTruth_train, GroundTruth_test

    # target data with data augmentation
    target_da_datas = np.transpose(imdb_da_train['data'], (3, 2, 0, 1))  # 换坐标轴 (9,9,103, 1800)->(1800, 103, 9, 9)
    print(target_da_datas.shape)
    target_da_labels = imdb_da_train['Labels']
    print('target data augmentation label:', target_da_labels)

    # metatrain data for few-shot classification 和之前的区别就是，把多维数组按类别划分为字典。
    target_da_train_set = {}
    target_aug_data_ssl = []
    target_aug_label_ssl = []

    for class_, path in zip(target_da_labels, target_da_datas):
        if class_ not in target_da_train_set:
            target_da_train_set[class_] = []
        target_da_train_set[class_].append(path)
        target_aug_data_ssl.append(path)
        target_aug_label_ssl.append(class_)
    target_da_metatrain_data = target_da_train_set
    print(target_da_metatrain_data.keys())

    return train_loader, test_loader, target_da_metatrain_data, G, RandPerm, Row, Column, nTrain, target_aug_data_ssl, target_aug_label_ssl


def get_alltest_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class, HalfWidth):

    print(Data_Band_Scaler.shape)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    '''label start'''
    num_class = int(np.max(GroundTruth))
    data_band_scaler = utils.flip(Data_Band_Scaler)
    groundtruth = utils.flip(GroundTruth)

    G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth,nColumn - HalfWidth:2 * nColumn + HalfWidth]
    data = data_band_scaler[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth,:]

    [Row, Column] = np.nonzero(G)
    del data_band_scaler
    del groundtruth

    nSample = np.size(Row)
    max_Row = np.max(Row)
    print('number of sample', nSample)

    train = {}

    m = int(np.max(G))

    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if G[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        nb_val = int(len(indices))
        train[i] = indices[:nb_val]

    train_indices = []

    for i in range(m):
        train_indices += train[i]
    np.random.shuffle(train_indices)

    print('the number of target:', len(train_indices))

    nTrain = len(train_indices)

    trainX = np.zeros([nTrain,  2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand], dtype=np.float32)
    trainY = np.zeros([nTrain], dtype=np.int64)

    RandPerm = train_indices
    RandPerm = np.array(RandPerm)
    for i in range(nTrain):
        trainX[i, :, :, :] = data[Row[RandPerm[i]] - HalfWidth: Row[RandPerm[i]] + HalfWidth + 1, Column[RandPerm[i]] - HalfWidth: Column[RandPerm[i]] + HalfWidth + 1, :] # 7 7 144
        trainY[i] = G[Row[RandPerm[i]], Column[RandPerm[i]]].astype(np.int64)
    trainX = np.transpose(trainX, (0, 3, 1, 2))
    trainY = trainY - 1

    print('all data shape', trainX.shape)
    print('all label shape', trainY.shape)

    test_dataset = MetaTrainLabeledDataset(image_datas=trainX, image_labels=trainY)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)
    return test_loader, G, RandPerm, Row, Column, nTrain

class tagetSSLDataset(Dataset):
    def __init__(self, image_datas, image_labels):
        self.image_datas = image_datas
        self.image_labels = image_labels

    def __len__(self):
        return len(self.image_datas)

    def __getitem__(self, idx):
        image = self.image_datas[idx]
        label = self.image_labels[idx]
        return image, label






