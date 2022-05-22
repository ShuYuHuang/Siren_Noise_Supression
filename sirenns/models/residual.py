import torch
import torch.nn as nn
# import torch.nn.functional as F
from torchaudio.models import ConvTasNet

from julius.resample import ResampleFrac


class ResidualConvTas(nn.Module):
    def __init__(self,enc_num_feats=128,
                 msk_num_hidden_feats=64,
                 device=torch.device("cuda")):
        super().__init__()
        self.device=device
#         self.num_sources=num_sources
        self.netargs=dict(
            enc_num_feats=enc_num_feats,
            msk_num_hidden_feats=msk_num_hidden_feats
        )
        
        self.noise_net=ConvTasNet(num_sources=2,**self.netargs).to(self.device)
        self.signal_net=ConvTasNet(num_sources=1,**self.netargs).to(self.device)
        # Resampler for upsampling
        self.resamp_8Kto16K=ResampleFrac(8000,16000).to(self.device)
    def forward(self, x):
        err=self.noise_net(x)
        res=x-torch.sum(err,dim=1,keepdim=True)
        sig=self.signal_net(res)
        return torch.cat((sig,err),dim=1)