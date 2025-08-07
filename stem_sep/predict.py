import sys
import librosa
import numpy as np
import torch
from model import UNet

def separate(path_to_wav, output_vocal="vocal.wav", output_inst="karaoke.wav"):
    audio, sr = librosa.load(path_to_wav, sr=44100)
    spec = librosa.stft(audio)
    mag = np.abs(spec)
    phase = np.angle(spec)

    model = UNet(in_channels=1, out_channels=1)
    model.load_state_dict(torch.load("stem_sep/model.pth", map_location="cpu"))
    model.eval()

    with torch.no_grad():
        input_tensor = torch.tensor(mag).unsqueeze(0).unsqueeze(0).float()
        mask = model(input_tensor).squeeze().numpy()
        vocal_mag = mask * mag
        inst_mag = (1 - mask) * mag

    vocal_spec = vocal_mag * np.exp(1j * phase)
    inst_spec = inst_mag * np.exp(1j * phase)
    vocal = librosa.istft(vocal_spec)
    inst = librosa.istft(inst_spec)

    librosa.output.write_wav(output_vocal, vocal, sr)
    librosa.output.write_wav(output_inst, inst, sr)

if __name__ == "__main__":
    separate(sys.argv[1])