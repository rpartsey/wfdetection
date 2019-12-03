import torch.optim as optim

OPTIMIZERS = dict(
    adam=optim.Adam,
    sgd=optim.SGD,
    adadelta=optim.Adadelta,
)
