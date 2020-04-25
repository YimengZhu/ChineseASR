import numpy as np
import random
from torchvision import transforms
from pdb import set_trace as bp

class TimeStretch:
    def __init__(self, w, low=0.8, high=1.25):
        self.w = w
        self.low = low
        self.high = high

    def __call__(self, spect):
        idx = None
        time_len = spect.shape[1]
        for i in range(time_len//self.w+1):
            stretch = random.uniform(self.low, self.high)
            win_end = min(time_len , self.w*(i+1))
            reidx = np.arange(self.w*i, win_end-1, stretch)
            reidx = np.round(reidx).astype(int)
            idx = np.concatenate((idx, reidx)) if idx is not None else reidx
        spect =  spect[:,idx]
        return spect


class SpectAugment():
    def __init__(self, freq_msk_num=1, time_msk_num=1):
        self.freq_msk_num = freq_msk_num
        self.time_msk_num = time_msk_num

    def __call__(self, spect):
        f, t = spect.shape

        for i in range(self.freq_msk_num):
            msk_size = np.random.uniform(low=0.0, high=f*0.1)
            msk_size = int(msk_size)
            if f -  msk_size< 0:
                continue
            msk_start = random.randint(0, f-msk_size)
            spect[msk_start:msk_start+msk_size, :] = 0

        for i in range(self.time_msk_num):
            msk_size = np.random.uniform(low=0.0, high=t*0.1)
            msk_size = int(msk_size)
            if t - msk_size < 0:
                continue
            msk_start = random.randint(0, t-msk_size)
            spect[:,msk_start:msk_start+msk_size] = 0

        return spect

