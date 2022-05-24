import sounddevice as sd
import argparse
import numpy as np

import torch
from sirenns.models.residual import ResidualConvTas
device=torch.device('cuda:0')
net=ResidualConvTas(enc_num_feats=128,
                       msk_num_hidden_feats=64,
                       device=device)
block_shift=int(8000*0.2)
signal_len=int(block_shift*8)

in_buffer=np.zeros((signal_len),dtype=np.float32)
out_buffer=np.zeros((signal_len),dtype=np.float32)

def callback(indata,outdata,frames, time, status=True):
    global in_buffer,out_buffer
    if status:
        print(status)
    in_buffer[:-block_shift] = in_buffer[block_shift:]
    in_buffer[-block_shift:] = np.squeeze(indata)
    
    x=torch.tensor(in_buffer[None,None,:],device=device)
    x=net.resamp_8Kto16K(x)
    pred=np.squeeze(net(x))[0,::2]
    pred=pred.cpu().detach().numpy()*0.005
    
    out_buffer[:-block_shift] = out_buffer[block_shift:]
    out_buffer[-block_shift:] = pred[-block_shift:]
    outdata[:] = out_buffer[-block_shift*3:-block_shift*2,None]
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    net.load_state_dict(torch.load("snapshots/2convtas/best.pth",map_location=device))
    print(sd.query_devices())
    try:
        with sd.Stream(device=("Microphone (Realtek High Defini",
                               "VoiceMeeter Input (VB-Audio Vir"),
                       samplerate=8000, blocksize=block_shift,
                       dtype=np.float32, latency=0.2,
                       channels=1, callback=callback):
            print('#' * 80)
            print('press Return to quit')
            print('#' * 80)
            input()
    except KeyboardInterrupt:
        parser.exit('')
    except Exception as e:
        parser.exit(type(e).__name__ + ': ' + str(e))