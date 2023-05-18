import torch
import torch.nn as nn

from pgportfolio.learn.inspect import *
from pgportfolio.tools.configprocess import load_config

from .nnagent import NNAgent

import numpy as np

config = load_config()
agent = NNAgent(config, None)

agent.network.conv1

vdict = {
    'ConvLayer/W': agent.network.conv1.weight,
    'ConvLayer/b': agent.network.conv1.bias,
    'EIIE_Dense/W': agent.network.conv2.weight,
    'EIIE_Dense/b': agent.network.conv2.bias,
    'EIIE_Output_WithW/W': agent.network.conv3.weight,
    'EIIE_Output_WithW/b': agent.network.conv3.bias,
    'btc_bias': agent.network.bias
}

vshapedict = {name: tuple(v.shape) for name, v in vdict.items()}
{'ConvLayer/W': (3, 3, 1, 2),
 'ConvLayer/b': (3,),
 'EIIE_Dense/W': (10, 3, 1, 30),
 'EIIE_Dense/b': (10,),
 'EIIE_Output_WithW/W': (1, 11, 1, 1),
 'EIIE_Output_WithW/b': (1,),
 'btc_bias': (1,)}

def assign_data(vname, np_array):
    v: torch.autograd.Variable = vdict[vname]
    assert tuple(v.shape) == tuple(np_array.shape)
    with torch.no_grad():
        v.set_(torch.from_numpy(np_array).to(v))

iw = get_default_initial_weights()
for name, np_array in iw.items():
    assign_data(name, np_array)




i = get_inputs()
x = i['x']
y = i['y']
prev_w = i['prev_w']
setw = lambda *a, **ka: None

agent.network.recording = True
agent.evaluate_tensors(x, y, prev_w, setw, ['output'])
agent.network.recording = False
outs = dict((k, v.cpu().numpy()) for k, v in agent.network.recorded.items())


from IPython import embed; import sys; embed(); sys.exit(0)
