import os
import numpy as np
import time

import torch

from torch.autograd import Variable

from saicinpainting.training.modules import ffc
from saicinpainting.training.modules import ffc_original

if __name__ == "__main__":

    init_conv_kwargs = {
        "ratio_gin": 0,
        "ratio_gout": 0,
        "enable_lfu": False}

    downsample_conv_kwargs = {
        "ratio_gin": 0,
        "ratio_gout": 0,
        "enable_lfu": False}

    resnet_conv_kwargs = {
        "ratio_gin": 0.75,
        "ratio_gout": 0.75,
        "enable_lfu": False}

    model1 = ffc.FFCResNetGenerator(input_nc=4, output_nc=3, ngf=24, n_blocks=6, n_downsampling=3,
                                   init_conv_kwargs=init_conv_kwargs,
                                   downsample_conv_kwargs=downsample_conv_kwargs,
                                   resnet_conv_kwargs=resnet_conv_kwargs)
 
    model2 = ffc_original.FFCResNetGenerator2(input_nc=4, output_nc=3, ngf=24, n_blocks=6, n_downsampling=3,
                                   init_conv_kwargs=init_conv_kwargs,
                                   downsample_conv_kwargs=downsample_conv_kwargs,
                                   resnet_conv_kwargs=resnet_conv_kwargs)
       
    model1.load_state_dict(torch.load('ffcresnet.pt'), strict=True)
    model2.load_state_dict(torch.load('ffcresnet.pt'), strict=True)

    height = 720
    width = 800
    
    model1.eval()
    model2.eval()

    inp = Variable(torch.randn((1, 4, 720, 800)))

    with torch.no_grad():
        torch.onnx.export(model1, inp, 'fft_test.onnx',
                          input_names=['input'],
                          output_names=['output'],
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH)

    m = torch.jit.trace(model2, inp)
    m.save('./fft_test.pt')

    ref = model1(inp)
    ref2 =m(inp)
    print(np.max(np.abs(ref.detach().numpy() - ref2.detach().numpy())))
    np.save('inp', inp.detach().numpy())
    np.save('ref', ref.detach().numpy())

