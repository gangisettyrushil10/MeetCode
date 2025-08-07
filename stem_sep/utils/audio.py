import numpy as np
import librosa
import torch
import warnings
warnings.filterwarnings('ignore')

class AudioProcessor:
    """
    Advanced audio processing class for stem separation.
    Handles loading, processing, and converting audio data.
    """
    def __init__(
        self,
        sample_rate=44100,
        n_fft=2048,
        hop_length=512,
        win_length=None,
        window='hann',
        center=True,
        pad_mode='reflect'
    ):
        """
        Initialize audio processor with specific parameters.
        
        Args:
            sample_rate: Audio sample rate (default: 44100 Hz)
            n_fft: FFT window size
            hop_length: Number of samples between successive FFT windows
            win_length: Window size (default: n_fft)
            window: Window type for FFT
            center: Whether to pad signal on both sides
            pad_mode: Signal padding mode
        """
        self.sr = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.window = window
        self.center = center
        self.pad_mode = pad_mode

    def load_audio(self, file_path, duration=None, offset=0.0):
        """
        Load audio file with optional duration and offset.
        
        Args:
            file_path: Path to audio file
            duration: Duration to load in seconds (optional)
            offset: Start reading after this time (seconds)
        
        Returns:
            audio: Audio signal
            sr: Sample rate
        """
        try:
            audio, sr = librosa.load(
                file_path,
                sr=self.sr,
                duration=duration,
                offset=offset
            )
            return audio, sr
        except Exception as e:
            print(f"Error loading audio file {file_path}: {str(e)}")
            return None, None

    def to_spectrogram(self, audio, log=False, normalize=True):
        """
        Convert audio to spectrogram representation.
        
        Args:
            audio: Audio signal
            log: Whether to convert to log scale
            normalize: Whether to normalize the spectrogram
        
        Returns:
            magnitude: Magnitude spectrogram
            phase: Phase spectrogram
        """
        # Compute STFT
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            pad_mode=self.pad_mode
        )
        
        # Separate magnitude and phase
        magnitude, phase = librosa.magphase(stft)
        
        if log:
            magnitude = np.log1p(magnitude)
            
        if normalize:
            magnitude = self.normalize(magnitude)
            
        return magnitude, phase

    def to_audio(self, magnitude, phase):
        """
        Convert spectrogram back to audio signal.
        
        Args:
            magnitude: Magnitude spectrogram
            phase: Phase spectrogram
        
        Returns:
            audio: Reconstructed audio signal
        """
        # Reconstruct complex STFT
        stft = magnitude * phase
        
        # Convert back to audio
        audio = librosa.istft(
            stft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=self.center,
            length=None
        )
        
        return audio

    def normalize(self, spectrogram):
        """
        Normalize spectrogram to [0, 1] range.
        
        Args:
            spectrogram: Input spectrogram
        
        Returns:
            normalized: Normalized spectrogram
        """
        return (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min() + 1e-8)

    def apply_mask(self, mixture_spec, mask):
        """
        Apply separation mask to mixture spectrogram.
        
        Args:
            mixture_spec: Complex mixture spectrogram
            mask: Separation mask (real-valued between 0 and 1)
        
        Returns:
            separated: Masked spectrogram
        """
        return mixture_spec * mask

    def compute_features(self, audio):
        """
        Compute additional audio features for better separation.
        
        Args:
            audio: Audio signal
        
        Returns:
            features: Dictionary of audio features
        """
        features = {
            'mfcc': librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=13),
            'spectral_centroid': librosa.feature.spectral_centroid(y=audio, sr=self.sr),
            'spectral_bandwidth': librosa.feature.spectral_bandwidth(y=audio, sr=self.sr),
            'spectral_rolloff': librosa.feature.spectral_rolloff(y=audio, sr=self.sr)
        }
        return features

    def preprocess_for_model(self, audio):
        """
        Prepare audio data for model input.
        
        Args:
            audio: Audio signal
        
        Returns:
            tensor: Processed tensor ready for model input
        """
        # Convert to spectrogram
        magnitude, _ = self.to_spectrogram(audio)
        
        # Add batch and channel dimensions
        tensor = torch.from_numpy(magnitude).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        
        return tensor

    def postprocess_from_model(self, tensor, original_phase):
        """
        Convert model output back to audio.
        
        Args:
            tensor: Model output tensor
            original_phase: Phase from original mixture
        
        Returns:
            audio: Reconstructed audio signal
        """
        # Remove batch and channel dimensions
        magnitude = tensor.squeeze().cpu().numpy()
        
        # Convert back to audio using original phase
        return self.to_audio(magnitude, original_phase)

    def save_audio(self, audio, file_path):
        """
        Save audio to file.
        
        Args:
            audio: Audio signal
            file_path: Output file path
        """
        try:
            librosa.output.write_wav(file_path, audio, self.sr)
            print(f"Audio saved successfully to {file_path}")
        except Exception as e:
            print(f"Error saving audio file: {str(e)}")