import os
import torch
from torch.utils.data import Dataset
import librosa
import numpy as np


class StemDataset(Dataset):
    def __init__(self, root_dir, sample_rate=44100, n_fft=1024, hop_length=512):
        self.root_dir = root_dir
        self.file_pairs = []
        self.sr = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

        for song in os.listdir(root_dir):
            mix = os.path.join(root_dir, song, "mixture.wav")
            vocal = os.path.join(root_dir, song, "vocals.wav")
            if os.path.exists(mix) and os.path.exists(vocal):
                self.file_pairs.append((mix, vocal))

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        mix, vocal = self.file_pairs[idx]
        mix_audio, _ = librosa.load(mix, sr=self.sr)
        vocal_audio, _ = librosa.load(vocal, sr=self.sr)

        mix_spec = librosa.stft(mix_audio, n_fft=self.n_fft, hop_length=self.hop_length)
        vocal_spec = librosa.stft(vocal_audio, n_fft=self.n_fft, hop_length=self.hop_length)

        mag_mix = np.abs(mix_spec)
        mag_vocal = np.abs(vocal_spec)

        return torch.tensor(mag_mix).float(), torch.tensor(mag_vocal).float()