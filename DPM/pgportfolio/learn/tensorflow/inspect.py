import sys
from IPython import embed
import numpy as np
from .nnagent import NNAgent
from pgportfolio.tools.configprocess import load_config
from pgportfolio.learn.inspect import *
import tensorflow as tf
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


config = load_config()
agent = NNAgent(config, None)

for name, graph_node in agent.net.layers_dict.items():
    print(name, graph_node.name)

ndict = {
    'ConvLayer': agent.layers_dict['ConvLayer_0_activation'],
    'EIIE_Dense': agent.layers_dict['EIIE_Dense_1_activation'],
    'EIIE_Output_WithW': agent.layers_dict['EIIE_Output_WithW_2_activation'],
    'voting': agent.layers_dict['voting_3_activation'],
    'softmax_layer': agent.layers_dict['softmax_layer_4_activation'],
}

vlist = agent.net.session.graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
vdict = {v.name.split(':')[0]: v for v in vlist}
vshapedict = {name: tuple(v.shape.as_list()) for name, v in vdict.items()}
{'ConvLayer/W': (1, 2, 3, 3),
 'ConvLayer/b': (3,),
 'EIIE_Dense/W': (1, 30, 3, 10),
 'EIIE_Dense/b': (10,),
 'EIIE_Output_WithW/W': (1, 1, 11, 1),
 'EIIE_Output_WithW/b': (1,),
 'btc_bias': (1, 1)}


def assign_data(vname, np_array):
    v: tf.Variable = vdict[vname]
    assert tuple(v.shape.as_list()) == tuple(np_array.shape)
    agent.session.run(v.assign(np_array))


iw = get_default_initial_weights()
iw = {'ConvLayer/W': iw['ConvLayer/W'].transpose(2, 3, 1, 0),
      'ConvLayer/b': iw['ConvLayer/b'],
      'EIIE_Dense/W': iw['EIIE_Dense/W'].transpose(2, 3, 1, 0),
      'EIIE_Dense/b': iw['EIIE_Dense/b'],
      'EIIE_Output_WithW/W': iw['EIIE_Output_WithW/W'].transpose(2, 3, 1, 0),
      'EIIE_Output_WithW/b': iw['EIIE_Output_WithW/b'],
      'btc_bias': iw['btc_bias'][:, None]}
for name, w in iw.items():
    assign_data(name, w)


output = agent.net.output

i = get_inputs()
x = i['x']
y = i['y']
prev_w = i['prev_w']
setw = lambda *a, **ka: None

# outs = {'ConvLayer/W': iw['ConvLayer/W'].transpose(2, 3, 1, 0),
#       'ConvLayer/b': iw['ConvLayer/b'],
#       'EIIE_Dense/W': iw['EIIE_Dense/W'].transpose(2, 3, 1, 0),
#       'EIIE_Dense/b': iw['EIIE_Dense/b'],
#       'EIIE_Output_WithW/W': iw['EIIE_Output_WithW/W'].transpose(2, 3, 1, 0),
#       'EIIE_Output_WithW/b': iw['EIIE_Output_WithW/b'],
#       'btc_bias': iw['btc_bias'][:, None]}



outs = dict((k, a) for k, a in zip(agent.net.layers_dict.keys(), agent.evaluate_tensors(x, y, prev_w, setw, list(agent.net.layers_dict.values()))))
outs = {
    'ConvLayer': outs['ConvLayer_0_activation'].transpose(0, 3, 1, 2),
    'EIIE_Dense': outs['EIIE_Dense_1_activation'].transpose(0, 3, 1, 2),
    'EIIE_Output_WithW': outs['EIIE_Output_WithW_2_activation'].transpose(0, 3, 1, 2),
    'voting': outs['voting_3_activation'],
    'softmax_layer': outs['softmax_layer_4_activation'],
}

embed()
sys.exit(0)
