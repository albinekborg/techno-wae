import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import Resample

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

## FÖR ATT ÅTERSKAPA: test_output = Resample(resample_rate,48000)(next(iter(dataset)))
## FÖR ATT SPARA: torchaudio.save("./test_output.wav",test_output.unsqueeze(0),48000)


class MP3Dataset(Dataset):
    def __init__(self, file_paths, resample_rate):
        self.file_paths = file_paths
        self.resample_rate = resample_rate
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        waveform, sample_rate = torchaudio.load(self.file_paths[idx],normalize=True)
        #waveform = librosa.util.normalize(waveform)

        waveform = torch.Tensor(waveform)#.unsqueeze(0)
        waveform = Resample(sample_rate, self.resample_rate)(waveform)
        waveform = waveform.to(device)
        return waveform



