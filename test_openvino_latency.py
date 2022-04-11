from openvino_extensions import get_extensions_path
from openvino.inference_engine import IECore

import sys
import subprocess
import pytest
from pathlib import Path

import numpy as np
import time

shapes = {}
shapes['input'] = (1, 4, 720, 800)
start_load = time.time()
ie = IECore()
ie.add_extension(get_extensions_path(), 'CPU')
ie.set_config({'CONFIG_FILE': 'user_ie_extensions/gpu_extensions.xml'}, 'GPU')

net = ie.read_network('fft_test.xml')
net.reshape(shapes)
exec_net = ie.load_network(net, 'CPU')
end_load = time.time()

print("Loading : ", end_load - start_load)

inputs = {}
inputs['input'] = np.random.randn(1, 4, 720, 800)

start_infer = time.time()

for i in range(50):
    #inputs['input'] = np.random.randn(1, 4, 720, 800)
    out = exec_net.infer(inputs)
    out = next(iter(out.values()))


end_infer = time.time()

print("infer : " , (end_infer - start_infer) / 50)
