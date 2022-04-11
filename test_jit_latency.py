import os
import numpy as np
import time

import torch

if __name__ == "__main__":
    start_load = time.time()
    m = torch.jit.load('./models/fft_test.pt', map_location='cpu')
    end_load = time.time()
    print("Loading : ", end_load - start_load)
    inp = torch.rand((1, 4, 720, 800))
    
    start_infer = time.time()
    for i in range(50):
        out = m(inp)
        
    end_infer = time.time()

    print("infer : " , (end_infer - start_infer) / 50)
