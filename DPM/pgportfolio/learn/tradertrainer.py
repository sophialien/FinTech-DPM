import pgportfolio.learn.config as config

print(config.backend)

if config.backend == 'tensorflow':
    from .tensorflow.tradertrainer import *
elif config.backend == 'torch':
    from .torch.tradertrainer import *
else:
    raise Exception('expect "tensorflow" or "torch"')
