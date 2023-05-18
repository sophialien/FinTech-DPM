import pgportfolio.learn.config as config

if config.backend == 'tensorflow':
    from .tensorflow.rollingtrainer import *
elif config.backend == 'torch':
    from .torch.rollingtrainer import *
else:
    raise Exception('expect "tensorflow" or "torch"')
