import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from rdkit import Chem
atom_d = {'H': 1, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Na': 11, 'P': 15, 'S': 16, 'Cl': 17, 'Br': 18, 'I': 19,
          'Ca': 20, 'Tc': 21, 'Se': 22, 'Fe': 23, 'Al': 24, 'Pt': 25, 'Bi': 26, 'Au': 27, 'Tl': 27, 'Cr': 28, 'Cu': 29,
          'Mn': 30, 'Zn': 31, 'Si': 32, 'Hg': 33, 'As': 34, 'Ti': 35}


def process(data):
    x, pos, miss = [], [], []
    for i, s in enumerate(data['smiles']):
        mol = Chem.MolFromSmiles(s)
        try:
            mol = Chem.MolToMolBlock(mol).splitlines()[4:]
        except:
            miss.append(i)
            continue
        x_tmp, pos_tmp = [], []
        for line in mol:
            if len(line) < 20:
                break
            items = line.split()
            try:
                x_tmp.append(atom_d[items[3]])
            except KeyError:
                print(items[3])
            pos_tmp.append([float(j) for j in items[:3]])

        x.append(torch.tensor(x_tmp))
        pos.append(torch.tensor(pos_tmp))
    x = pad_sequence(x, batch_first=True, padding_value=0)
    pos = pad_sequence(pos, batch_first=True, padding_value=0)
    return x, pos, miss


def esol():
    data = pd.read_csv('../../delaney-processed.csv')
    y = torch.tensor(data['ESOL predicted log solubility in mols per litre'].to_numpy()).float()
    mask = torch.BoolTensor([True] * len(y))
    x, pos, miss = process(data)
    mask[miss] = 0
    y = y[mask]
    torch.save([x, pos, y], '../data/esol.pt')


def freesolv():
    data = pd.read_csv('../../SAMPL.csv')
    y = torch.tensor(data['calc'].to_numpy()).float()
    mask = torch.BoolTensor([True] * len(y))
    x, pos, miss = process(data)
    mask[miss] = 0
    y = y[mask]
    torch.save([x, pos, y], '../data/freesolv.pt')


def lipo():
    data = pd.read_csv('../../Lipophilicity.csv')
    y = torch.tensor(data['exp'].to_numpy()).float()
    mask = torch.BoolTensor([True] * len(y))
    x, pos, miss = process(data)
    mask[miss] = 0
    y = y[mask]
    torch.save([x, pos, y], '../data/lipo.pt')


if __name__ == '__main__':
    esol()
