import os
import numpy as np
from tqdm import tqdm
from datetime import datetime
from audio_hjk2 import hparams as audio_hparams
from audio_hjk2 import load_wav, wav2unnormalized_mfcc, wav2normalized_db_mel, wav2normalized_db_spec
from audio_hjk2 import write_wav, normalized_db_mel2wav, normalized_db_spec2wav

import tensorflow as tf
from models import CNNBLSTMCalssifier


# 超参数个数：16
hparams = {
    'sample_rate': 16000,
    'preemphasis': 0.97,
    'n_fft': 400,
    'hop_length': 160,
    'win_length': 400,
    'num_mels': 80,
    'n_mfcc': 13,
    'window': 'hann',
    'fmin': 30.,
    'fmax': 7600.,
    'ref_db': 20,  
    'min_db': -80.0,  
    'griffin_lim_power': 1.5,
    'griffin_lim_iterations': 60,  
    'silence_db': -28.0,
    'center': True,
}


assert hparams == audio_hparams


MFCC_DIM = 39
PPG_DIM = 218


# in 
meta_path = '/ceph/dataset/BZNSYP/ProsodyLabeling/000001-010000.txt'
# 超参数会指定采样率16k, 外部磁盘就不用降了
wav_dir = '/ceph/dataset/BZNSYP/Wave'

# out1
ppg_dir = './CN-PPG_databaker/ppg_ForcePPG'
mfcc_dir = './CN-PPG_databaker/mfcc_ForcePPG'
mel_dir = './CN-PPG_databaker/mel_ForcePPG'
spec_dir = './CN-PPG_databaker/spec_ForcePPG'
rec_wav_dir = './CN-PPG_databaker/rec_wavs_ForcePPG'
os.makedirs(ppg_dir, exist_ok=True)
os.makedirs(mfcc_dir, exist_ok=True)
os.makedirs(mel_dir, exist_ok=True)
os.makedirs(spec_dir, exist_ok=True)
os.makedirs(rec_wav_dir, exist_ok=True)
# out2
out_meta_path = './CN-PPG_databaker/meta_list_ForcePPG.txt'


# NN->PPG
ckpt_path = './aishell1_ckpt_model_dir/aishell1ASR.ckpt-128000'


def check_ppg(ppg):
    print('max and min:', ppg.max(), ppg.min())
    print(ppg.shape)



def main():
    #这一部分用于处理DataBaker格式的数据集
    a = open(meta_path, 'r').readlines()
    b = []
    i = 0
    while i < len(a):
        b.append(a[i].strip()[:6])
        i += 2
    a = b
    print(a[:3])

    # a = PPG_get_restore(a, ppg_dir, ppg_dir, mel_dir, spec_dir)

    # NN->PPG
    # Set up network
    mfcc_pl = tf.placeholder(dtype=tf.float32, shape=[None, None, MFCC_DIM], name='mfcc_pl')
    classifier = CNNBLSTMCalssifier(out_dims=PPG_DIM, n_cnn=3, cnn_hidden=256, cnn_kernel=3, n_blstm=2, lstm_hidden=128)
    predicted_ppgs = tf.nn.softmax(classifier(inputs=mfcc_pl)['logits'])
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    print('Restoring model from {}'.format(ckpt_path))
    saver.restore(sess, ckpt_path)


    meta_list_str = ''
    cnt = 0
    bad_list = []
    for fname in tqdm(a):
        try:
            # 提取声学参数
            wav_f = os.path.join(wav_dir, fname + '.wav')
            wav_arr = load_wav(wav_f)
            mfcc_feats = wav2unnormalized_mfcc(wav_arr)
            ppgs = sess.run(predicted_ppgs, feed_dict={mfcc_pl: np.expand_dims(mfcc_feats, axis=0)})
            ppgs = np.squeeze(ppgs)
            mel_feats = wav2normalized_db_mel(wav_arr)
            spec_feats = wav2normalized_db_spec(wav_arr)

            # 验证声学参数提取的对
            save_name = fname + '.npy'
            save_mel_rec_name = fname + '_mel_rec.wav'
            save_spec_rec_name = fname + '_spec_rec.wav'
            assert ppgs.shape[0] == mfcc_feats.shape[0]
            assert mfcc_feats.shape[0] == mel_feats.shape[0] and mel_feats.shape[0] == spec_feats.shape[0]
            write_wav(os.path.join(rec_wav_dir, save_mel_rec_name), normalized_db_mel2wav(mel_feats))
            write_wav(os.path.join(rec_wav_dir, save_spec_rec_name), normalized_db_spec2wav(spec_feats))
            check_ppg(ppgs)
            
            # 存储声学参数
            mfcc_save_name = os.path.join(mfcc_dir, save_name)
            ppg_save_name = os.path.join(ppg_dir, save_name)
            mel_save_name = os.path.join(mel_dir, save_name)
            spec_save_name = os.path.join(spec_dir, save_name)
            np.save(mfcc_save_name, mfcc_feats)
            np.save(ppg_save_name, ppgs)
            np.save(mel_save_name, mel_feats)
            np.save(spec_save_name, spec_feats)

            meta_list_str += fname + '\n'
            cnt += 1
            print(fname + 'success...')
        except Exception as e:
            bad_list.append(fname)
            print(str(e))
        
        # break

    print('good:', cnt)
    print('bad:', len(bad_list))
    print(bad_list)

    f_meta = open(out_meta_path, 'w')
    f_meta.write(meta_list_str)

    return


if __name__ == '__main__':
    main()
