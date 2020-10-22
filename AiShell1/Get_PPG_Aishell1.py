import os
import numpy as np
from tqdm import tqdm


# in
alignments_f = 'ali_all.txt'

# out
ppgs_dir = 'PPGs'
if not os.path.exists(ppgs_dir):
    os.makedirs(ppgs_dir)

def save_ppg_as_mfcc_path(idx, ppg):
    save_name = idx + '.npy'
    save_name = os.path.join(ppgs_dir, save_name)
    np.save(save_name, ppg)

def main():
    cnt = 0
    f = open(alignments_f, 'r')
    a = f.readlines()
    for line in tqdm(a):
        line = line.strip().split(' ')
        idx = line[0]
        if 'sp' in idx:
            continue

        ppg = np.asarray(line[1:])
        save_ppg_as_mfcc_path(idx, ppg)
        cnt += 1
        print(cnt)
        # break

    # print(cnt)
    return


if __name__ == '__main__':
    main()
