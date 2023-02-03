import os
from time import time

import torch
import torch.optim as opt
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import mse_loss
from torch.nn.utils.rnn import pad_sequence
from scipy import stats
from pytorch_metric_learning import losses
from sklearn.model_selection import train_test_split
import sys

sys.path.append("..")
from egnn import EGNN_Network
from utils.model import build_model
from utils.dann import Classifier, get_lambda
from utils.mrot import pot_mrot
from utils.kmeans import kmeans, kmeans_predict
from utils.utils import parse_args, Logger, set_seed


def main():
    args = parse_args()
    args.n_clusters = [int(i) for i in args.n_clusters.split(',')]
    set_seed(args.seed)
    log = Logger(args.save_path, f'mof_{args.method}_{args.model}.log')
    args.ep = 100
    args.bs = 16 * len(args.gpu.split(','))
    args.lr = 5e-5 * len(args.gpu.split(','))

    data_x, data_y = torch.load('../data/coremof.pt')
    data_x = pad_sequence(data_x, batch_first=True, padding_value=0)[:, :args.max_len]
    data_x, data_pos = data_x[..., 3], data_x[..., :3]
    exp_x, exp_pos, exp_y = torch.load('../data/co2.pt')
    exp_x, exp_pos = exp_x[:, :args.max_len], exp_pos[:, :args.max_len]
    exp_y = exp_y[:, -1]
    if args.model == 'egnn':
        data_x, data_pos, data_y = data_x.double(), data_pos.double(), data_y.double()         # double precision
        exp_x, exp_pos, exp_y = exp_x.double(), exp_pos.double(), exp_y.double()

    # split the dataset into seen and unseen parts
    unlabeled_x, labeled_x, unlabeled_pos, labeled_pos, unlabeled_y, labeled_y = train_test_split(exp_x, exp_pos, exp_y, test_size=args.ratio, random_state=args.seed)
    train_size, test_size = len(data_x), len(unlabeled_y)
    train_dataset = TensorDataset(data_x, data_pos, data_y)
    train_dataset_labeled = TensorDataset(labeled_x, labeled_pos, labeled_y)
    test_dataset = TensorDataset(unlabeled_x, unlabeled_pos, unlabeled_y)
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    train_loader_labeled = DataLoader(train_dataset_labeled, batch_size=args.bs, shuffle=True)

    if args.method in ['dann', 'cadn', 'mrot']:
        test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)   # here DANN、CADN only consider using unseen data for adversarial training for simplicity
    else:
        test_loader = DataLoader(test_dataset, batch_size=args.bs * 2, shuffle=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    log.logger.info(f'{"=" * 40} Domain Adaption {"=" * 40}\nDataset: MOF; Method: {args.method}; Train: {train_size}; Test: {test_size}; Ratio: {args.ratio}\n'
                    f'GPU: {args.gpu}; Epoch: {args.ep}; Batch_size: {args.bs}; Save: {args.save}\n{"=" * 40} Start Training {"=" * 40}')
    if args.method == 'mrot': log.logger.info(f'Clusters: {args.n_clusters}')
    if args.model == 'egnn':
        model = EGNN_Network(num_tokens=100, dim=32, depth=6, num_nearest_neighbors=5, norm_coors=True, coor_weights_clamp_value=2., aggregate=True).cuda()
    else:
        model = build_model(args.atom_class, 1, dist_bar=[3, 7]).cuda()
    if args.model == 'egnn': model = model.double()  # double precision

    if len(args.gpu) > 1:  model = torch.nn.DataParallel(model)
    if args.method in ['dann', 'cadn']:
        test_set = iter(test_loader)
        bce = torch.nn.BCELoss()
        discriminator = Classifier().cuda()
        d_optimizer = opt.Adam(discriminator.parameters(), lr=args.lr)
        if len(args.gpu) > 1:  discriminator = torch.nn.DataParallel(discriminator)
    if args.method == 'mrot': loss_func = losses.TripletMarginLoss()
    criterion = torch.nn.MSELoss()
    optimizer = opt.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.6, patience=5, min_lr=5e-7)
    best_metric, best_loss, t0, early_stop = 1e8, 1e8, time(), 0

    for epoch in range(0, args.ep):
        model.train()
        loss, t1, step = 0.0, time(), 0
        cluster_centers = [None] * len(args.n_clusters)
        for src_x, src_pos, src_y in train_loader:
            src_x, src_pos, src_y = src_x.long().cuda(), src_pos.cuda(), src_y.cuda()
            if args.method == 'mldg':
                src_y, index = src_y.sort()
                tmp_size = int(len(src_y) * 0.8)
                meta_src_x, meta_src_pos = src_x[index[:tmp_size]], src_pos[index[:tmp_size]]
                meta_tgt_x, meta_tgt_pos = src_x[index[tmp_size:]], src_pos[index[tmp_size:]]
                meta_src_mask = (meta_src_x != 0)
                meta_tgt_mask = (meta_tgt_x != 0)
                feat, pred = model(meta_src_x, meta_src_pos, mask=meta_src_mask)
            else:
                src_mask = (src_x != 0)

            if args.method in ['dann', 'cadn']:
                discriminator.train()
                if step % len(test_loader) == 0: test_set = iter(test_loader)
                tgt_x, tgt_pos, _ = test_set.next()
                tgt_x, tgt_pos = tgt_x.long().cuda(), tgt_pos.cuda()
                tgt_mask = (tgt_x != 0)
                d_src = torch.ones(len(src_x), 1).cuda()
                d_tgt = torch.zeros(len(tgt_x), 1).cuda()
                d_y = torch.cat([d_src, d_tgt], dim=0)

                feat, pred = model(torch.cat([src_x, tgt_x], dim=0), torch.cat([src_pos, tgt_pos], dim=0), mask=torch.cat([src_mask, tgt_mask], dim=0))
                if args.method == 'cadn': feat = feat * pred     # torch.bmm cannot be used here because the prediction is only one-dimension
                d_pred = discriminator(feat.detach())
                d_loss = bce(d_pred, d_y)
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()

                if args.method == 'cadn': feat = feat * pred
                d_loss = bce(discriminator(feat), d_y)
            elif args.method == 'mrot':
                test_set = iter(test_loader)
                tgt_x, tgt_pos, tgt_y = test_set.next()
                tgt_x, tgt_pos, tgt_y = tgt_x.long().cuda(), tgt_pos.cuda(), tgt_y.cuda()
                tgt_mask = (tgt_x != 0)

                mass_src = torch.ones(len(src_x)).cuda() / len(src_x)
                mass_tgt = torch.ones(len(tgt_x)).cuda() / len(tgt_x)

                feat, pred = model(torch.cat([src_x, tgt_x], dim=0), torch.cat([src_pos, tgt_pos], dim=0), mask=torch.cat([src_mask, tgt_mask], dim=0))

                # adopt semi-supervised MROT distance formula, cosine_similarity in pytorch is not pairwise
                # https://stackoverflow.com/questions/50411191/how-to-compute-the-cosine-similarity-in-pytorch-for-all-rows-in-a-matrix-with-re
                a_norm = feat[:len(src_x)] / feat[:len(src_x)].norm(dim=1)[:, None]
                b_norm = feat[len(src_x):] / feat[len(src_x):].norm(dim=1)[:, None]
                M_x = 1 - torch.mm(a_norm, b_norm.transpose(0, 1))
                M_y = torch.cdist(src_y.unsqueeze(-1), tgt_y.unsqueeze(-1))

                M_y = M_y / torch.max(M_y) + 1e-3  # add residue to prevent the case that M_x / M_y is too large
                M = (M_x * (M_x / M_y).log())
                M[M < 0] = 1e-4   # ensure distance is not negative
                ot_loss = pot_mrot(mass_src, mass_tgt, M, args.reg1, args.reg2, src_y)   # OT loss

                triplet_loss = 0
                for i in range(len(args.n_clusters)):
                    if cluster_centers[i] is not None:
                        cluster_ids = kmeans_predict(feat, cluster_centers[i], device=torch.device(f'cuda:{args.gpu[0]}'))
                        if len(torch.unique(cluster_ids)) != 1:
                            cluster_centers[i] = torch.stack([torch.mean(feat[cluster_ids == j], dim=0) for j in range(args.n_clusters[i])], dim=0)
                        else:
                            cluster_ids, cluster_centers_tmp = kmeans(X=feat, num_clusters=args.n_clusters[i], device=torch.device('cuda', int(args.gpu[0])))
                            cluster_centers[i] = cluster_centers_tmp
                    else:
                        cluster_ids, cluster_centers_tmp = kmeans(X=feat, num_clusters=args.n_clusters[i], device=torch.device('cuda', int(args.gpu[0])))
                        cluster_centers[i] = cluster_centers_tmp
                    triplet_loss += loss_func(feat, cluster_ids)

            elif args.method != 'mldg':
                feat, pred = model(src_x, src_pos, mask=src_mask)

            if args.method == 'mldg':
                loss_batch = criterion(pred[..., 0], src_y[index[:tmp_size]])
            else:
                loss_batch = criterion(pred[:len(src_x), 0], src_y)
            loss += loss_batch.item() / train_size * len(src_x)

            for tgt_x_labeled, tgt_pos_labeled, tgt_y_labeled in train_loader_labeled:    # regression loss of seen data
                tgt_x_labeled, tgt_pos_labeled, tgt_y_labeled = tgt_x_labeled.long().cuda(), tgt_pos_labeled.cuda(), tgt_y_labeled.cuda()
                tgt_mask_labeled = (tgt_x_labeled != 0)
                feat, pred = model(tgt_x_labeled, tgt_pos_labeled, mask=tgt_mask_labeled)
                loss_batch += criterion(pred[..., 0], tgt_y_labeled)

            if args.method in ['dann', 'cadn']:
                loss_batch = loss_batch - 1e2 * get_lambda(epoch, args.ep) * d_loss            # add adversarial loss
                discriminator.zero_grad()
                step += 1
            elif args.method == 'mrot':
                reg_loss = loss_batch.item()
                ot_loss = args.ot_weight * ot_loss   # 加上OT loss
                triplet_loss = args.triplet_weight * triplet_loss
                loss_batch += ot_loss + triplet_loss

            if args.method == 'mldg':
                feat, pred = model(meta_tgt_x, meta_tgt_pos, mask=meta_tgt_mask)
                meta_loss_batch = loss_batch + args.meta_val_beta * criterion(pred[..., 0], src_y[index[tmp_size:]])
                optimizer.zero_grad()
                meta_loss_batch.backward()
                optimizer.step()
            else:
                optimizer.zero_grad()
                loss_batch.backward()
                optimizer.step()

        model.eval()
        val_metric = 0
        y_pred = []
        for tgt_x, tgt_pos, src_y in test_loader:
            tgt_x, tgt_pos, src_y = tgt_x.long().cuda(), tgt_pos.cuda(), src_y.cuda()
            tgt_mask = (tgt_x != 0)
            with torch.no_grad():
                feat, pred = model(tgt_x, tgt_pos, mask=tgt_mask)
                y_pred.append(pred)
            val_metric += mse_loss(pred[..., 0], src_y).item() / test_size * len(tgt_x)
        y_pred = torch.cat(y_pred)[:, 0]
        spearman = stats.spearmanr(y_pred.cpu().numpy(), unlabeled_y.numpy())[0]
        pearson = stats.pearsonr(y_pred.cpu().numpy(), unlabeled_y.numpy())[0]

        val_metric = val_metric ** 0.5    # calculate RMSE
        if args.method == 'mrot':
            log.logger.info('Epoch: {} | Time: {:.1f}s | reg_loss: {:.3f} | ot_loss: {:.3f} | triplet_loss: {:.3f} |'
                            ' Val_Metric: {:.3f} | Spearman: {:.3f} | Pearson: {:.3f} | Lr: {:.3f}'
                            .format(epoch + 1, time() - t1, reg_loss, ot_loss.item(), triplet_loss.item(), val_metric, spearman, pearson, optimizer.param_groups[0]['lr'] * 1e5))
        else:
            log.logger.info('Epoch: {} | Time: {:.1f}s | Loss: {:.3f} | Val_Metric: {:.3f} | Spearman: {:.3f} | Pearson: {:.3f} | Lr: {:.3f}'
                            .format(epoch + 1, time() - t1, loss, val_metric, spearman, pearson, optimizer.param_groups[0]['lr'] * 1e5))
        lr_scheduler.step(val_metric)
        if val_metric < best_metric:
            best_ep = epoch
            best_metric = val_metric
            if args.save: best_model = model
        if loss < best_loss:
            best_loss = loss
            early_stop = 0
        else:
            early_stop += 1
        if early_stop >= 30:
            log.logger.info('Early Stopping!!! No Improvement on Loss for 30 Epochs.')
            break
    log.logger.info(f'{"=" * 20} End Training (Time: {(time() - t0) / 3600:.2f}h) {"=" * 20}\nBest Metric for {args.method} in mof: {best_metric} (epoch: {best_ep})')
    if args.save: torch.save(best_model.state_dict(), f'../save/model_mof_{args.method}.pt')


if __name__ == '__main__':
    main()








