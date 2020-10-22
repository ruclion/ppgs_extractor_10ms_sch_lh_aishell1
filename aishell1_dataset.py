import os
import numpy as np
import time

#生成数据的代码
#train/test每一行都只是一个文件名
TRAIN_FILE = './AiShell1/train_aishell1.txt'#'/media/luhui/experiments_data/librispeech/train.txt'
TEST_FILE = './AiShell1/test_aishell1.txt'#'/media/luhui/experiments_data/librispeech/dev.txt'
MFCC_DIR =  './AiShell1/MFCCs'      #'/media/luhui/experiments_data/librispeech/mfcc_hop12.5'#生成MFCC的目录
PPG_DIR =   './AiShell1/PPGs'   #'/media/luhui/experiments_data/librispeech/phone_labels_hop12.5'
MFCC_DIM = 39
AiShell1_PPG_DIM = 218


def text2list(file):
    f = open(file, 'r').readlines()
    f = [i.strip() for i in f]
    print(f[:3])
    return f


def onehot(arr, depth, dtype=np.float32):
    assert len(arr.shape) == 1 #不为1则异常
    onehots = np.zeros(shape=[len(arr), depth], dtype=dtype)
    arr=arr.astype(np.int64)

    arr = arr-1  #下标从0开始
    arr = arr.tolist()

    onehots[np.arange(len(arr)), arr] = 1
    return onehots


def get_single_data_pair(fname, mfcc_dir, ppg_dir):
    assert os.path.isdir(mfcc_dir) and os.path.isdir(ppg_dir)

    mfcc_f = os.path.join(mfcc_dir ,fname+'.npy')
    ppg_f = os.path.join(ppg_dir ,fname+'.npy')

    mfcc = np.load(mfcc_f)
    ppg = np.load(ppg_f)
    ppg = onehot(ppg, depth=AiShell1_PPG_DIM)
    assert mfcc.shape[0] == ppg.shape[0],fname+' 维度不相等'
    return mfcc, ppg


def train_generator():
    file_list = text2list(file=TRAIN_FILE)
    for f in file_list:
        mfcc, ppg = get_single_data_pair(f, mfcc_dir=MFCC_DIR, ppg_dir=PPG_DIR)
        yield mfcc, ppg, mfcc.shape[0]


def test_generator():
    file_list = text2list(file=TEST_FILE)
    for f in file_list:
        mfcc, ppg = get_single_data_pair(f, mfcc_dir=MFCC_DIR, ppg_dir=PPG_DIR)

