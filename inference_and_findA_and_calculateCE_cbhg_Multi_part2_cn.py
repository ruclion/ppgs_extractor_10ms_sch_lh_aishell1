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

# in1
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
meta_path = 'wavs_inference_and_findA_and_calculateCE_list.txt'

# in2: NN->PPG
ckpt_path = './aishell1_ckpt_model_dir/aishell1ASR.ckpt-128000'

# out
log_dir = os.path.join('inference_findA_CE_PPG_log_dir', STARTED_DATESTRING)
ppg_dir = os.path.join(log_dir, 'ppg')
rec_wav_dir = os.path.join(log_dir, 'rec_wavs_16000')
os.makedirs(ppg_dir, exist_ok=True)
os.makedirs(rec_wav_dir, exist_ok=True)



def check_ppg(ppg):
    print('max and min:', ppg.max(), ppg.min())
    print(ppg.shape)







def main():
    print('start...')
    a = open(meta_path, 'r').readlines()

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

    
    cnt = 0
    bad_list = []
    for wav_f in tqdm(a):
        try:
            wav_f = wav_f.strip()
            print('process:', wav_f)
            # 提取声学参数
            wav_arr = load_wav(wav_f)
            mfcc_feats = wav2unnormalized_mfcc(wav_arr)
            ppgs = sess.run(predicted_ppgs, feed_dict={mfcc_pl: np.expand_dims(mfcc_feats, axis=0)})
            ppgs = np.squeeze(ppgs)
            mel_feats = wav2normalized_db_mel(wav_arr)
            spec_feats = wav2normalized_db_spec(wav_arr)

            # /datapool/home/hujk17/ppg_decode_spec_10ms_sch_Multi/inference_findA_Multi_log_dir/2020-11-09T11-09-00/0_sample_spec.wav
            fname = wav_f.split('/')[-1].split('.')[0]
            save_mel_rec_name = fname + '_mel_rec.wav'
            save_spec_rec_name = fname + '_spec_rec.wav'
            assert ppgs.shape[0] == mfcc_feats.shape[0]
            assert mfcc_feats.shape[0] == mel_feats.shape[0] and mel_feats.shape[0] == spec_feats.shape[0]
            write_wav(os.path.join(rec_wav_dir, save_mel_rec_name), normalized_db_mel2wav(mel_feats))
            write_wav(os.path.join(rec_wav_dir, save_spec_rec_name), normalized_db_spec2wav(spec_feats))
            check_ppg(ppgs)
            
            # 存储ppg参数
            ppg_save_name = os.path.join(ppg_dir, fname + '.npy')
            np.save(ppg_save_name, ppgs)

            cnt += 1
        except Exception as e:
            bad_list.append(wav_f)
            print(str(e))
        
        # break

    print('good:', cnt)
    print('bad:', len(bad_list))
    print(bad_list)

    return


if __name__ == '__main__':
    main()
