import os
import numpy as np
from audio import wav2mfcc_v2, load_wav


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
    'ref_db': 20,  #
    'min_db': -80.0,  # restrict the dynamic range of log power
    'iterations': 100,  # griffin_lim #iterations
    'silence_db': -28.0,
    'center': False,
}


# in
wav_dir = 'wavs_16000_960'
server_common_data_list = ['/datapool/dataset/data_aishell/wav/train', '/datapool/home/hujk17/ppgs_extractor_10ms_sch_lh_aishell1/AiShell1/dev', '/datapool/home/hujk17/ppgs_extractor_10ms_sch_lh_aishell1/AiShell1/test']

# out
mfcc_dir = 'MFCCs'

if not os.path.exists(mfcc_dir):
    os.makedirs(mfcc_dir)

def main():
    #这一部分用于处理LibriSpeech格式的数据集。
    for first_dir in server_common_data_list:
        for second_dir in os.listdir(os.path.join(wav_dir, first_dir)):
            second_wav_dir = os.path.join(os.path.join(wav_dir,first_dir),second_dir)
            wav_files = [os.path.join(second_wav_dir, f) for f in os.listdir(second_wav_dir) if f.endswith('.wav')]
            cnt = 0
            for wav_f in wav_files:
                try:
                    wav_arr = load_wav(wav_f, sr=hparams['sample_rate'])
                    mfcc_feats = wav2mfcc_v2(wav_arr, sr=hparams['sample_rate'],
                                            n_mfcc=hparams['n_mfcc'], n_fft=hparams['n_fft'],
                                            hop_len=hparams['hop_length'], win_len=hparams['win_length'],
                                            window=hparams['window'], num_mels=hparams['num_mels'],
                                            center=hparams['center'])
                    save_name = wav_f.split('/')[-1].split('.')[0] + '.npy'
                    save_name = os.path.join(mfcc_dir, save_name)
                    np.save(save_name, mfcc_feats)
                    cnt += 1
                    print(cnt)
                except:
                    print(wav_f)
                # break
            # break
        # break
        # 提取完毕以后，需要手动将3个文件夹的东西mv到同一个，和ppg一样的2338个文件夹
    return


if __name__ == '__main__':
    main()
