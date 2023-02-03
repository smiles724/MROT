import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.manifold import TSNE
from model import build_model
from utils import parse_args, set_seed
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(show=True):
    fontsize = 24
    fig, ax1 = plt.subplots(figsize=(8, 4))
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.rcParams.update({'font.size': fontsize})

    erm = [83.010, 0.040]
    ot = [82.752, 0.033]
    ot_e = [82.459, 0.031]
    ot_e_v = [81.143, 0.025]
    triplet = [82.643, 0.034]
    all = [80.140, 0.023]

    barWidth = 0.03
    br1 = [0, barWidth * 8]
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]
    br5 = [x + barWidth for x in br4]
    br6 = [x + barWidth for x in br5]

    ax1.bar(br1[0], erm[0], color='orange', width=barWidth, edgecolor='black', linewidth=1.5)
    ax1.bar(br2[0], ot[0], color='limegreen', width=barWidth, edgecolor='black', linewidth=1.5)
    ax1.bar(br3[0], ot_e[0], color='deepskyblue', width=barWidth, edgecolor='black', linewidth=1.5)
    ax1.bar(br4[0], ot_e_v[0], color='royalblue', width=barWidth, edgecolor='black', linewidth=1.5)
    ax1.bar(br5[0], triplet[0], color='purple', width=barWidth, edgecolor='black', linewidth=1.5)
    ax1.bar(br6[0], all[0], color='peru', width=barWidth, edgecolor='black', linewidth=1.5)
    ax1.set_ylim([78, 85])

    ax2 = ax1.twinx()
    ax2.bar(br1[1], erm[1], color='orange', width=barWidth, edgecolor='black', linewidth=1.5, label='ERM')
    ax2.bar(br2[1], ot[1], color='limegreen', width=barWidth, edgecolor='black', linewidth=1.5, label='OT Only')
    ax2.bar(br3[1], ot_e[1], color='deepskyblue', width=barWidth, edgecolor='black', linewidth=1.5, label='OT with Entropy')
    ax2.bar(br4[1], ot_e_v[1], color='royalblue', width=barWidth, edgecolor='black', linewidth=1.5, label='OT with All Regularizers')
    ax2.bar(br5[1], triplet[1], color='purple', width=barWidth, edgecolor='black', linewidth=1.5, label='Triplet Only')
    ax2.bar(br6[1], all[1], color='peru', width=barWidth, edgecolor='black', linewidth=1.5, label='MROT')
    ax2.set_ylim([0.02, 0.05])

    plt.ylabel('MAE', fontsize=fontsize)
    plt.xticks([r + barWidth * 2 for r in br1], ['QM7', 'QM8'])

    plt.legend(loc="upper right")
    if show:
        plt.show()
    else:
        plt.savefig('../../motif_ablation.pdf', bbox_inches='tight')


def t_SNE():
    """ only support ERM and MROT """
    args = parse_args()
    if not args.model_path: args.model_path = f'../save/model_{args.data}_{args.method}.pt'
    set_seed(args.seed)
    data_x, data_pos, data_y = torch.load(f'../data/{args.data}.pt')
    if args.data == 'qm9': data_y = data_y[:, args.qm9_index]

    # you cannot scale label randomly
    dataloader = DataLoader(TensorDataset(data_x, data_pos, data_y), batch_size=args.bs, shuffle=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    print(f'{"=" * 40} Visualization {"=" * 40}\nDataset: {args.data}; Method: {args.method}')
    if args.data == 'qm9':
        targets = ['mu', 'alpha', 'homo', 'lumo', 'gap']
        print(f'Target: {targets[args.qm9_index]}')

    model = build_model(args.atom_class, 1, dist_bar=[0.8, 1.6]).cuda()
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint)
    print('Loading model successfully...')
    if len(args.gpu_id) > 1:  model = torch.nn.DataParallel(model)

    mse = 0
    model.eval()
    feats, ys = [], []
    for src_x, src_pos, y in dataloader:
        src_x, src_pos, y = src_x.long().cuda(), src_pos.cuda(), y.cuda()
        src_mask = (src_x != 0).unsqueeze(1)
        src_dist = torch.cdist(src_pos, src_pos).float()
        with torch.no_grad():
            feat, pred = model(src_x, src_mask, src_dist)
            feats.append(feat)
            ys.append(y)
            mse += torch.nn.functional.mse_loss(pred[:, 0], y)

    feats = torch.cat(feats, dim=0).cpu().numpy()
    ys = torch.cat(ys, dim=0).cpu().numpy()

    # PCA is too slow, UMAP performs badly, TSNE can tune iterations
    tsne = TSNE(n_components=2, verbose=1, n_iter=2000)
    feats = tsne.fit_transform(feats)

    plt.figure()
    plt.scatter(feats[:, 0], feats[:, 1], s=1.5, c=ys, alpha=0.8)
    plt.colorbar(label='property')
    plt.savefig(f'../pca_{args.data}_{args.method}.png', bbox_inches='tight')
    print('mse:', mse)


def cost_function():
    """ visualization of cost functions under different metrics """
    import matplotlib.ticker as mtick
    p = 2
    fontsize = 18

    def f_add(x, y):
        return x ** p + y ** p

    def f_KL(x, y, k=0):
        return np.abs(y ** p * (np.log(x ** p / (y ** p + 1e-2)) + k)) + np.abs(x ** p * (np.log(y ** p / (x ** p + 1e-2)) + k))

    x = np.linspace(0.1, 1, 100)
    y = np.linspace(0.1, 1, 100)

    X, Y = np.meshgrid(x, y)
    Z_add = f_add(X, Y)
    Z_KL = 0.2 * f_KL(X, Y) + f_add(X, Y)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)
    ax.zaxis.set_tick_params(labelsize=fontsize)
    ax.zaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    # ax.contour3D(X, Y, Z, 50, cmap='Greens')
    ax.plot_surface(X, Y, Z_add, cmap='viridis')  # , edgecolor='none')
    # ax.set_xlabel('$d_Y(y^s,y^t)$', fontsize=20)
    # ax.set_ylabel('$d_H(f(x^s),f(x^t))$', fontsize=20)
    # ax.set_zlabel('$D_z$', fontsize=20, rotation=90)
    ax.view_init(30, 140)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)
    ax.zaxis.set_tick_params(labelsize=fontsize)
    ax.zaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    ax.plot_surface(X, Y, Z_KL, cmap='viridis')  # , edgecolor='none')
    # ax.set_xlabel('$d_Y(y^s,y^t)$', fontsize=20)
    # ax.set_ylabel('$d_H(f(x^s),f(x^t))$', fontsize=20)
    # ax.set_zlabel('$D_z$', fontsize=20, rotation=0)
    ax.view_init(30, 140)
    plt.tight_layout()
    plt.show()


plot()
