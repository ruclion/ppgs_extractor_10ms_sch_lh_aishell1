import os
import numpy as np
from tqdm import tqdm


AiShell1_PPG_DIM = 218

# in
mfccs_dir = 'MFCCs'
ppgs_dir = 'PPGs'

# out
meta_path = 'meta_aishell1.txt'
train_path = 'train_aishell1.txt'
test_path = 'test_aishell1.txt'


def onehot(arr, depth, dtype=np.float32):
    assert len(arr.shape) == 1 #不为1则异常
    onehots = np.zeros(shape=[len(arr), depth], dtype=dtype)
    arr=arr.astype(np.int64)

    arr = arr-1  #下标从0开始
    arr = arr.tolist()

    onehots[np.arange(len(arr)), arr] = 1
    return onehots


def same_frames(fname, ppgs_dir, mfccs_dir):
    ppg_f = os.path.join(ppgs_dir, fname + '.npy')
    mfcc_f = os.path.join(mfccs_dir, fname + '.npy')
    ppg = np.load(ppg_f)
    mfcc = np.load(mfcc_f)
    
    ppg = onehot(ppg, depth=AiShell1_PPG_DIM)
    if mfcc.shape[0] != ppg.shape[0]:
        return False
    return True


def main():
    meta_list_fromPPGs = []
    meta_list_fromMFCCs = []
    mfcc_dict = {}
    meta_list_final = []

    meta_list_fromPPGs = [f[:-4] for f in os.listdir(ppgs_dir) if f.endswith('.npy')]
    print('PPGs:', len(meta_list_fromPPGs))
    meta_list_fromMFCCs = [f[:-4] for f in os.listdir(mfccs_dir) if f.endswith('.npy')]
    for x in meta_list_fromMFCCs:
        mfcc_dict[x] = 1
    print('MFCCs:', len(meta_list_fromMFCCs))


    for x in tqdm(meta_list_fromPPGs):
        if mfcc_dict.get(x, 0) == 1:
            if same_frames(x, ppgs_dir, mfccs_dir):
                meta_list_final.append(x)
            else:
                print('frames of ppg is diff from mfcc:', x)
        else:
            print('only ppg, no mfcc:', x)
    print('Final Lists:', len(meta_list_final))


    # 写入文件
    f = open(meta_path, 'w')
    for idx in meta_list_final:
        f.write(idx + '\n')
    
    np.random.shuffle(meta_list_final)
    len_all = len(meta_list_final)
    len_train = int(len_all * 0.8)
    f_train = open(train_path, 'w')
    f_test = open(test_path, 'w')
    for i, x in enumerate(meta_list_final):
        if i < len_train:
            f_train.write(x + '\n')
        else:
            f_test.write(x + '\n')

    return


if __name__ == '__main__':
    main()
