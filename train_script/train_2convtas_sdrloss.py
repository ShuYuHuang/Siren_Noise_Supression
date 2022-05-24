#!/usr/bin/env python
# coding: utf-8
# General
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Calculation
import librosa

# Torch related
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud

# Self-Defined functions
import sirenns.utils.losses as lsf 
import sirenns.datasets.loader as ldr
from sirenns.models.residual import ResidualConvTas

# Simple functions
def set_seed(seed=2022):
    np.random.seed(seed)
    torch.manual_seed(seed)
def split(x,test_val_ratio):
    x0,x1=train_test_split(x,test_size=sum(test_val_ratio))
    x1,x2=train_test_split(x1,test_size=test_val_ratio[-1])
    return x0,x1,x2
lamb=[0.8,0.1,0.1]
def one_batch(i_iter,log,sample_batched,model,criterion,optimizer):
    # Prep input
    x,signal,artifact,noise=[_ for _ in sample_batched]
    
    # Predict output
    x=model.resamp_8Kto16K(x)
    pred=model(x)
    
    # Calculate losses
    loss_sig=criterion(pred[:,0:1], signal)
    loss_art=criterion(pred[:,1:2], artifact)
    loss_noise=criterion(pred[:,2:3], noise)
    loss = loss_sig*  lamb[0]+\
           loss_art*  lamb[1]+\
           loss_noise*lamb[2]
    
    # Back-propagate and update model if training
    if model.training:
        #Update
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Moving average loss record
    with torch.no_grad():
        loss_rec = loss.item()
        loss_sig=loss_sig.item()
        log['loss'] = (loss_rec+log['loss']*i_iter)/(i_iter + 1)
        log['loss_sig'] = (loss_sig+log['loss_sig']*i_iter)/(i_iter + 1)
        
    return pred,loss
    
if __name__ == "__main__":

    with torch.cuda.device(1):
        ####### DATA #######
        # List data
        speak_files=glob("../../datas/cv-corpus-9.0-2022-04-27/zh-TW/clips/*.mp3")
        firetruck_files=glob("../../datas/sounds/firetruck/*.wav")
        construction_files=glob("../../datas/sounds/splited_construction/*.wav")
        print(
            "#人聲:",len(speak_files),
            "\n#消防車聲:",len(firetruck_files),
            "\n#工地聲:",len(construction_files)
        )

        # Split data
        set_seed(2022)
        sig_train,sig_val,sig_test=split(speak_files,(0.1,0.1))
        art_train,art_val,art_test=split(firetruck_files,(0.1,0.1))
        noise_train,noise_val,noise_test=split(construction_files,(0.1,0.1))

        # Parameters for dataset
        # SEED=28
        SAMPLE_RATE=16000
        BATCH_SIZE=2
        L=16000*4
        # Common keword arguments
        common_kwargs=dict(
            signal_len=L,
            n_partitions=20,
            transform=ldr.transform,
            artifact_transform=ldr.artifact_noise_transform,
            noise_transform=ldr.artifact_noise_transform
        )

        # Dataset, Loader
        train_ds=ldr.SyntheticCallDataset(sig_train,
                                      artifact_files=art_train,
                                      noise_files=noise_train,
                                      **common_kwargs)
        val_ds=ldr.SyntheticCallDataset(sig_val,
                                      artifact_files=art_val,
                                      noise_files=noise_val,
                                      **common_kwargs)
        train_dl=tud.DataLoader(train_ds,
                                batch_size=BATCH_SIZE,
                                collate_fn=train_ds.collate_fn)
        val_dl=tud.DataLoader(val_ds,
                              batch_size=BATCH_SIZE,
                              collate_fn=val_ds.collate_fn)

        ####### Network #######
        # Form Network
        network=ResidualConvTas(enc_num_feats=128,
                       msk_num_hidden_feats=64,
                       device=torch.device("cuda"))
        # Loss cocmbo 1: MSE+SGD
        # criterion = nn.MSELoss()
        # optimizer = torch.optim.SGD(network.parameters(),lr=1e-3,momentum=0.9,weight_decay=0.0005)
        # Loss cocmbo 2: SDR + clip norm + Adam
        criterion = lambda y,pred: lsf.cal_loss(y,pred,float(L))
        # Clip Norm
        for p in network.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -5, 5))

        optimizer = torch.optim.Adam(network.parameters(),lr=1e-3)

        ####### Training #######


        os.makedirs("../snapshots/2convtas",exist_ok=True)
        os.system("echo training started")
        EPOCH=100
        network.train()
        best_loss=np.inf
        PAITIENCE=5
        count=0
        try:
            for e in range(EPOCH):
                log= {'epoch':e,'step':0,'loss': 0, 'loss_sig': 0}
                network.train()
    #             session=tqdm(enumerate(train_dl))
                session=enumerate(train_dl)
                for i_iter, sample_batched in session:
                    pred,loss=one_batch(i_iter,
                                        log,
                                        sample_batched,
                                        network,
                                        criterion,
                                        optimizer)
                    # print loss and take snapshots
                    if (i_iter + 1) % 5000 == 0:
                        log['step']=i_iter+1
                        os.system(f"echo {log}")
                os.system(f"echo {log}")
                # validate
                log = {'epoch':e,'step':'val','loss': 0, 'loss_sig': 0}
                network.eval()
                with torch.no_grad():
    #                 session=tqdm(enumerate(val_dl))
                    session=enumerate(val_dl)
                    for i_iter,sample_batched in session:

                        pred,loss=one_batch(i_iter,
                                            log,
                                            sample_batched,
                                            network,
                                            criterion,
                                            optimizer)
                        if (i_iter + 1) % 5000 == 0:
                            os.system(f"echo {log}")
                os.system(f"echo {log}")
                # early stop        
                if log["loss_sig"]<best_loss:
                    best_loss=log["loss_sig"]
                    torch.save(network.state_dict(),
                               '../snapshots/2convtas/best.pth')
                elif count<=PAITIENCE: count+=1
                else:
                    count=0
                    best_loss=np.inf
                    break
        except KeyboardInterrupt:
            os.system("echo Human Interrupted")
            os.system(f"echo {log}")
        torch.save(network.state_dict(),
                   '../snapshots/2convtas/latest.pth')
