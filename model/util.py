import torch
import numpy as np
import random
from scipy import signal
import librosa
import sys

class STOP:
    FLAG = False

def stop_handler(signum, frame):
    STOP.FLAG = True

def get_partition(filename):
    fid = int(filename.stem)
    return (fid * 4129 + 173) % 97

def pick_false_audio_samples(audio_samples):
    batch_size = audio_samples.size()[0]
    weights = torch.ones((batch_size, batch_size), device=audio_samples.device)
    weights.sub_(torch.eye(batch_size, device=weights.device))
    false_indices = torch.multinomial(weights, 1).view(-1)
    false_audio_samples = torch.index_select(audio_samples, 0, false_indices)
    return false_audio_samples

def audio_transform(audio):
    amp = random.uniform(1.0, 1.1)
    audio *= amp
    try:
        audio = signal.resample(audio, 48000)
    except:
        print("all zero audio", file=sys.stderr)
        audio = np.zeros(48000, dtype=np.float32)
    _, _, audio = signal.spectrogram(audio, nperseg=480, noverlap=240, nfft=512, mode='psd')
    #print(np.min(audio), np.max(audio))
    audio = torch.from_numpy(np.log(audio + 1e-16)).unsqueeze(0)
    return audio

def audio_transform_mel(audio):
    amp = random.uniform(1.0, 1.1)
    audio *= amp
    sample_rate = 48000
    try:
        audio = signal.resample(audio, sample_rate)
    except:
        print("all zero audio", file=sys.stderr)
        audio = np.zeros(48000, dtype=np.float32)
    audio = librosa.feature.melspectrogram(y=audio, sr=sample_rate, hop_length=239, n_fft=512, win_length=480, n_mels=257, center=False)
    audio = librosa.power_to_db(audio)
    audio = torch.from_numpy(audio).unsqueeze(0)
    return audio

def audio_transform_raw(audio):
    amp = random.uniform(1.0, 1.1)
    audio *= amp
    sample_rate = 48000
    try:
        audio = signal.resample(audio, sample_rate)
    except:
        print("all zero audio", file=sys.stderr)
        audio = np.zeros(48000, dtype=np.float32)
    audio = torch.from_numpy(audio).unsqueeze(0)
    return audio

def audio_transform_mfcc(audio):
    amp = random.uniform(1.0, 1.1)
    audio *= amp
    sample_rate = 48000
    try:
        audio = signal.resample(audio, sample_rate)
    except:
        print("all zero audio", file=sys.stderr)
        audio = np.zeros(48000, dtype=np.float32)
    audio = librosa.feature.mfcc(y=audio, sr=sample_rate, hop_length=239, n_fft=512, win_length=480, center=False)
    audio = torch.from_numpy(audio).view(-1)
    audio_mean = audio.mean()
    audio_std =audio.std()
    audio.sub_(audio_mean).div_(audio_std)
    return audio