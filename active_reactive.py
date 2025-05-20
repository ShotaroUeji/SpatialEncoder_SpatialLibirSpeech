import numpy as np
import soundfile as sf
import librosa

def foa_active_reactive(path,
                        n_fft=512, hop=256,
                        channel_order=('W','Y','Z','X')):
    """
    Parameters
    ----------
    path : str
        4-ch FOA (wav, AmbiX/Loud-FMT) ファイルへのパス
    Returns
    -------
    I_a, I_r : ndarray  shape=(3, T, F)
        active / reactive 強度ベクトル [x,y,z] 成分
    """
    # --- 1. 4 ch 読み込み ------------------------
    y, sr = sf.read(path, always_2d=True)          # (samples, 4)
    W, Y, Z, X = y.T if channel_order == ('W','Y','Z','X') else y.T[::-1]

    # --- 2. STFT -------------------------------
    def stft(ch):  # shape=(frames, freq, complex)
        return librosa.stft(ch, n_fft=n_fft, hop_length=hop, window='hann').T
    A00, A1m1, A10, A11 = map(stft, (W, Y, Z, X))

    # --- 3. 強度ベクトル計算 ---------------------
    conj_W = np.conj(A00)
    stack_dip = np.stack([A1m1, A10, A11], axis=-1)  # (..., 3)

    I_complex = conj_W[..., None] * stack_dip         # (..., 3)
    I_active  =  np.real(I_complex)                  # active (energy flow)
    I_reactive = np.imag(I_complex)                  # reactive (stored)

    # 軸を [3, time, freq] にそろえる
    I_a = I_active.transpose(2, 0, 1)
    I_r = I_reactive.transpose(2, 0, 1)
    return I_a, I_r

I_a, I_r = foa_active_reactive("/home/takamichi-lab-pc09/SpatialLibriSpeech/preSpatialLibriSpeech/000000.flac")
print(I_a.shape)   # → (3, time_frames, freq_bins)
