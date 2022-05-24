import sounddevice as sd
import argparse
import numpy as np

   
def dummy(indata,outdata,frames, time, status=True):
    outdata[:] =indata
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    global in_buffer,out_buffer
    block_shift=int(8000*0.2)
    signal_len=int(block_shift*5)

    print(sd.query_devices())
    
    try:
        with sd.Stream(device=("Microphone (Realtek High Defini",
                               "VoiceMeeter Input (VB-Audio Vir"),
                       samplerate=8000, blocksize=block_shift,
                       dtype=np.float32, latency=0,
                       channels=1, callback=dummy):
            print('#' * 80)
            print('press Return to quit')
            print('#' * 80)
            input()
    except KeyboardInterrupt:
        parser.exit('')
    except Exception as e:
        parser.exit(type(e).__name__ + ': ' + str(e))