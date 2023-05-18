import numpy as np


def get_default_initial_weights():
    np.random.seed(87)
    d = {'ConvLayer/W': (3, 3, 1, 2),
         'ConvLayer/b': (3,),
         'EIIE_Dense/W': (10, 3, 1, 30),
         'EIIE_Dense/b': (10,),
         'EIIE_Output_WithW/W': (1, 11, 1, 1),
         'EIIE_Output_WithW/b': (1,),
         'btc_bias': (1,)}
#     w = {name: np.random.randn(*shape)
#          for name, shape in d.items()}
    w = {name: np.ones(shape, dtype=np.float32) / 100000000
         for name, shape in d.items()}
    return w


def get_inputs(B=109, F=3, C=11, W=31):
     np.random.seed(87)
     i = {'x': np.zeros([B, F, C, W], dtype=np.float32) / 100,
          'y': np.zeros([B, F, C], dtype=np.float32) / 100,
          'prev_w': np.zeros([B, C], dtype=np.float32) / 100}
     # i = {'x': np.random.randn(B, F, C, W),
     #      'y': np.random.randn(B, F, C),
     #      'prev_w': np.random.randn(B, C)}
     return i