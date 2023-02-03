import numpy as np
import torch.nn as nn


class Classifier(nn.Module):
    """ domain classifier """

    def __init__(self, input_size=512):
        super(Classifier, self).__init__()
        self.layer = nn.Sequential(nn.Linear(input_size, 256), nn.ReLU(inplace=True),
                                   nn.Linear(256, 1), nn.Sigmoid())

    def forward(self, h):
        c = self.layer(h)
        return c


def get_lambda(epoch, max_epoch):
    p = epoch / max_epoch
    return 2. / (1 + np.exp(-10. * p)) - 1.
