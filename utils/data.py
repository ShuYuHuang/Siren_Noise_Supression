import platform
os_name = platform.system()
import numpy as np
import torch
import torch.utils.data as tud
import torchaudio
from torch_audiomentations import Compose, Gain,PitchShift,Shift,AddBackgroundNoise
from julius.resample import ResampleFrac

if os_name=="Linux":
    torchaudio.set_audio_backend("sox_io") # for Linux system
    
###################################### Parameters and settings ######################################
SEED=28
SAMPLE_RATE=16000
not_signal_kwargs={"sample_rate":SAMPLE_RATE,
            "mode":"per_example",
            "p":1.}
signal_kwargs={"sample_rate":SAMPLE_RATE,
            "mode":"per_example",
            "p":0.5}
artifact_noise_transform=Compose(
    transforms=[
    PitchShift(# Pitch shift (frequency) without changinng meaning
                    min_transpose_semitones= -4.0, 
                    max_transpose_semitones= +1.0,
                    **not_signal_kwargs),
    Shift( # Waveform time shift (ratio) without changing shape
        min_shift=-0.5,
        max_shift=0.5,
        **not_signal_kwargs),
    Gain( # louder or quieter
        min_gain_in_db=-20.0,
        max_gain_in_db=-5.0,
        **not_signal_kwargs),
    ]
)
transform = Compose(
    transforms=[
        PitchShift(# Pitch shift (frequency) without changinng meaning
            min_transpose_semitones= -4.0, 
            max_transpose_semitones= +4.0, 
            **signal_kwargs),
        Shift( # Waveform time shift (ratio) without changing shape
            min_shift=-0.8,
            max_shift=0.8,
            **signal_kwargs),
        Gain( # louder or quieter
            min_gain_in_db=-5.0,
            max_gain_in_db=10.0,
            **signal_kwargs),
    ]
)

###################################### Data Synthesis Dataset  ######################################
class SyntheticCallDataset(tud.IterableDataset):
    def __init__(self,
                 signal_files,
                 artifact_files,
                 noise_files,
                 signal_len,
                 transform=None,
                 artifact_transform=None,
                 noise_transform=None):
        super().__init__()
        self.device=torch.device("cuda")
        self.signal_files=signal_files
        self.artifact_files=artifact_files
        self.noise_files=noise_files
        self.signal_len=signal_len
        # Initialize augmentation callable
        self.transform=transform
        self.artifact_transform = artifact_transform
        self.noise_transform = noise_transform
        # Resamplers for Frequency alignment
        self.resamp_32Kto16K=ResampleFrac(32000,16000).to(self.device)
        self.resamp_48Kto16K=ResampleFrac(48000,16000).to(self.device)
        self.resamp_44Kto16K=ResampleFrac(44100,16000).to(self.device)
        # Resamplers for synthetic signal generation
        self.resamp_16Kto8K=ResampleFrac(16000,8000).to(self.device)
    def __len__(self):
        return len(self.signal_files)
    def load_waveform(self,fname,transform=None):
        # Load data
        x_,sr_orig=torchaudio.load(fname,normalize=True)
        # Extend channel
        if x_.shape[0]==1:
            x_=torch.cat((x_,x_),dim=0)
        # Resample
        if sr_orig==16000:
            x_=x_.to(self.device)
        elif sr_orig==48000:
            x_=self.resamp_48Kto16K(x_.to(self.device))
        elif sr_orig==44100:
            x_=x_.to(torch.float32)
            x_=self.resamp_44Kto16K(x_.to(self.device))
        elif sr_orig==32000:
            x_=self.resamp_32Kto16K(x_.to(self.device))
        else:
            assert sr_orig in [48000,44100,32000], "Unsupported Sample Rate"
        ## Augmentation    
        if transform:
            x_=transform(x_[None,...])[0]
        ## Padding if needed
        if len(x_[0])<self.signal_len:
            x=torch.zeros((2,self.signal_len),dtype=torch.float32,device=self.device)
            start_point=np.random.randint(0,self.signal_len-len(x_[0]))
            end_point=start_point+len(x_[0])
            x[:,start_point:end_point]=x_
        else:
            x=x_[:,:self.signal_len]
        return x
        
    def __iter__(self):
        # Random choice of signal, artifact, noise combinations
        L=len(self)
        signal_files=np.random.permutation(self.signal_files)[:L]
        artifact_files=np.random.choice(self.artifact_files, size=L,replace=True)
        noise_files=np.random.choice(self.noise_files, size=L,replace=True)
        
        for fname_s,fname_a,fname_n in zip(signal_files, artifact_files,noise_files):
            ##### Loading waveforms #####
            # Load Signal
            signal=self.load_waveform(fname_s,self.transform)
            # Load artifact
            artifact=self.load_waveform(fname_a,self.artifact_transform)
            # Load Noise
            noise=self.load_waveform(fname_n,self.noise_transform)
            
            yield signal,artifact,noise
    def collate_fn(self,batch):
        signal,artifact,noise=[torch.stack([b[i] for b in batch],dim=0) for i in range(3)] 
        # Synthesize
        x=self.resamp_16Kto8K(signal+artifact+noise)
        return x,signal,artifact,noise