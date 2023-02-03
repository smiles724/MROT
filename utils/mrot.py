""" reference from group lasso regularization
    https://pythonot.github.io/_modules/ot/da.html#SinkhornLpl1Transport """
import torch
import ot
from ot.backend import get_backend


def pot_mrot(a, b, M, reg1, reg2, y, Fro=False):
    def f(G):
        """ the regularization term """
        # property-sparsity regularization
        reg = torch.sum((G + G ** 2) * y.unsqueeze(-1) ** 2)

        if Fro:
            # Frobenius norm
            reg += 0.5 * torch.sum(G ** 2)

        return reg

    def df(G):
        """ gradient of the regularization term. ignore np.sum，这里可以直接用pytorch来计算给定G的梯度 """
        return G + y.unsqueeze(-1) ** 2 + torch.mm(y.unsqueeze(-1),
                                                   torch.sum(y.unsqueeze(-1) * G, dim=-2, keepdim=True))

    nx = get_backend(a, b, M)
    ot_G = ot.optim.gcg(a, b, M, reg1, reg2, f, df, verbose=False)
    loss = nx.sum(M * ot_G) + reg1 * nx.sum(ot_G * nx.log(ot_G)) + reg2 * f(ot_G)
    return loss


def sinkhorn_baseline(a, b, M, reg):
    """ simple OT with only cross entropy term """
    loss = ot.sinkhorn2(a, b, M, reg)
    return loss
