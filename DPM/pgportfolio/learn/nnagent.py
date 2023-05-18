import pgportfolio.learn.config as config

if config.backend == 'tensorflow':
    from .tensorflow.nnagent import *
elif config.backend == 'torch':
    from .torch.nnagent import *
else:
    raise Exception('expect "tensorflow" or "torch"')
