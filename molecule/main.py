import os
from time import time, strftime, localtime

import torch
import torch.optim as opt
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import l1_loss, mse_loss
from pytorch_metric_learning import losses
import ot
import sys

sys.path.append("..")
from egnn import EGNN_Network
from utils.model import build_model
from utils.dann import Classifier, get_lambda
from utils.mrot import pot_mrot   # TODO: rename to MROT
from utils.kmeans import kmeans, kmeans_predict
from utils.utils import parse_args, Logger, set_seed

args = parse_args()
set_seed(args.seed)
log = Logger(args.save_path, f'{args.data}_{args.method}_{args.model}.log')
args.n_clusters = '2'
args.n_clusters = [int(i) for i in args.n_clusters.split(',')]
args.ep = 1000
if args.method != 'mrot':
    args.bs = 1024 * len(args.gpu.split(','))
else:
    args.bs = 256 * len(args.gpu.split(','))  # bs cannot be too large to compute triplet loss https://discuss.pytorch.org/t/torch-geometric/106975/2
args.lr = 1e-4 * len(args.gpu.split(','))


def test(model, loader, test_size):
    model.eval()
    test_metric = 0
    for tgt_x, tgt_pos, src_y in loader:
        tgt_x, tgt_pos, src_y = tgt_x.long().cuda(), tgt_pos.cuda(), src_y.cuda()
        tgt_mask = (tgt_x != 0)
        with torch.no_grad():
            feat, pred = model(tgt_x, tgt_pos, mask=tgt_mask)
        if args.data in ['qm7', 'qm8', 'qm9']:
            test_metric += l1_loss(pred[..., 0], src_y).item() / test_size * len(tgt_x)
        else:
            test_metric += mse_loss(pred[..., 0], src_y).item() / test_size * len(tgt_x)

    if args.data not in ['qm7', 'qm8', 'qm9']: test_metric = test_metric ** 0.5  # use RMSE
    return test_metric


def main():
    data_x, data_pos, data_y = torch.load(f'../data/{args.data}.pt')
    if args.data == 'qm9': data_y = data_y[:, args.qm9_index]
    data_y, index = data_y.sort()
    train_size = int(len(data_x) * 0.8)
    val_size = (len(data_x) - train_size) // 2
    if args.data == 'qm9' and args.model == 'egnn': data_x, data_pos = data_x.double(), data_pos.double()  # double precision
    train_dataset = TensorDataset(data_x[index[:train_size]], data_pos[index[:train_size]], data_y[:train_size])
    val_dataset = TensorDataset(data_x[index[train_size: train_size + val_size]], data_pos[index[train_size: train_size + val_size]], data_y[train_size: train_size + val_size])
    test_dataset = TensorDataset(data_x[index[train_size + val_size:]], data_pos[index[train_size + val_size:]], data_y[train_size + val_size:])
    train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)

    if args.method in ['dann', 'cdan', 'mrot']:
        val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)
    else:
        val_loader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.bs * 2, shuffle=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    log.logger.info(f'{strftime("%Y-%m-%d_%H-%M-%S", localtime())}\n'
                    f'{"=" * 40} Domain Adaption {"=" * 40}\nBackbone: {args.model}; Dataset: {args.data}; Method: {args.method}; Train: {train_size}; Val: {val_size} '
                    f'Test: {len(test_dataset)}\nGPU: {args.gpu}; Epoch: {args.ep}; Batch_size: {args.bs}; Save: {args.save}\n{"=" * 40} Start Training {"=" * 40}')
    if args.data == 'qm9':
        targets = ['mu', 'alpha', 'homo', 'lumo', 'gap']
        log.logger.info(f'Target: {targets[args.qm9_index]}')

    if args.model == 'egnn':
        model = EGNN_Network(num_tokens=100, dim=32, depth=6, num_nearest_neighbors=5, norm_coors=True,
                             coor_weights_clamp_value=2., aggregate=True).cuda()
    else:
        model = build_model(args.atom_class, 1, dist_bar=[0.8, 1.6, 3.0]).cuda()
    if args.data == 'qm9' and args.model == 'egnn': model = model.double()  # double precision

    if len(args.gpu) > 1:  model = torch.nn.DataParallel(model)
    if args.method in ['dann', 'cdan']:
        test_set = iter(test_loader)
        bce = torch.nn.BCELoss()
        discriminator = Classifier().cuda()
        d_optimizer = opt.Adam(discriminator.parameters(), lr=args.lr)
        if len(args.gpu) > 1:  discriminator = torch.nn.DataParallel(discriminator)
    if args.method == 'mrot': loss_func = losses.TripletMarginLoss()
    criterion = torch.nn.L1Loss() if args.data in ['qm7', 'qm8', 'qm9'] else torch.nn.MSELoss()

    optimizer = opt.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = opt.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.6, patience=10, min_lr=5e-7)
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

            if args.method in ['dann', 'cdan']:
                discriminator.train()
                if step % len(test_loader) == 0: test_set = iter(test_loader)
                tgt_x, tgt_pos, _ = test_set.next()
                tgt_x, tgt_pos = tgt_x.long().cuda(), tgt_pos.cuda()
                tgt_mask = (tgt_x != 0)
                d_src = torch.ones(len(src_x), 1).cuda()
                d_tgt = torch.zeros(len(tgt_x), 1).cuda()
                d_y = torch.cat([d_src, d_tgt], dim=0)

                feat, pred = model(torch.cat([src_x, tgt_x], dim=0), torch.cat([src_pos, tgt_pos], dim=0), mask=torch.cat([src_mask, tgt_mask], dim=0))
                if args.method == 'cdan':
                    feat = feat * pred  # because the prediction is 1-dimension，we cannot use torch.bmm there
                d_pred = discriminator(feat.detach())
                d_loss = bce(d_pred, d_y)
                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()

                if args.method == 'cdan': feat = feat * pred
                d_loss = bce(discriminator(feat), d_y)    # keep the gradient of feat
            elif args.method == 'jdot':
                src_feat, pred = model(src_x, src_pos, mask=src_mask)
                test_set = iter(test_loader)
                tgt_x, tgt_pos, tgt_y = test_set.next()
                tgt_x, tgt_pos, tgt_y = tgt_x.long().cuda(), tgt_pos.cuda(), tgt_y.cuda()
                tgt_mask = (tgt_x != 0)

                mass_src = torch.ones(len(src_x)).cuda() / len(src_x)
                mass_tgt = torch.ones(len(tgt_x)).cuda() / len(tgt_x)

                tgt_feat, tgt_pred = model(tgt_x, tgt_pos, mask=tgt_mask)
                src = torch.cat([src_feat, src_y.unsqueeze(-1)], dim=-1)
                tgt = torch.cat([tgt_feat, tgt_pred], dim=-1)

                M = torch.cdist(src.float(), tgt.float())
                ot_loss = ot.sinkhorn2(mass_src, mass_tgt, M, 1e2)   # regularization term is 100

            elif args.method == 'mrot':
                test_set = iter(test_loader)
                tgt_x, tgt_pos, _ = test_set.next()
                tgt_x, tgt_pos = tgt_x.long().cuda(), tgt_pos.cuda()
                tgt_mask = (tgt_x != 0)

                mass_src = torch.ones(len(src_x)).cuda() / len(src_x)
                mass_tgt = torch.ones(len(tgt_x)).cuda() / len(tgt_x)

                feat, pred = model(torch.cat([src_x, tgt_x], dim=0), torch.cat([src_pos, tgt_pos], dim=0), mask=torch.cat([src_mask, tgt_mask], dim=0))
                M = torch.cdist(feat[:len(src_x)], feat[len(src_x):])      # calculate the distance matrix of features
                if args.ot == 'new':
                    ot_loss = pot_mrot(mass_src, mass_tgt, M, reg1=1e3, reg2=1e3, y=src_y)      # calculate the OT loss
                else:
                    ot_loss = ot.sinkhorn2(mass_src, mass_tgt, M, reg=1e2, numItermax=1000)

                triplet_loss = 0     # construct dynamic clusters
                for i in range(len(args.n_clusters)):
                    if cluster_centers[i] is not None:
                        cluster_ids = kmeans_predict(feat, cluster_centers[i], device=torch.device(f'cuda:{args.gpu[0]}'))

                        if len(torch.unique(cluster_ids)) != 1:
                            cluster_centers[i] = torch.stack([torch.mean(feat[cluster_ids == j], dim=0) for j in range(args.n_clusters[i])], dim=0)
                        else:  # if there is only one cluster, then re-cluster from the beginning
                            cluster_ids, cluster_centers_tmp = kmeans(X=feat, num_clusters=args.n_clusters[i], device=torch.device('cuda', int(args.gpu[0])))
                            cluster_centers[i] = cluster_centers_tmp
                    else:
                        cluster_ids, cluster_centers_tmp = kmeans(X=feat, num_clusters=args.n_clusters[i], device=torch.device('cuda', int(args.gpu[0])))
                        cluster_centers[i] = cluster_centers_tmp
                    triplet_loss += loss_func(feat, cluster_ids)  # calculate the triplet loss

            elif args.method != 'mldg':
                feat, pred = model(src_x, src_pos, mask=src_mask)

            if args.method == 'mldg':
                loss_batch = criterion(pred[..., 0], src_y[index[:tmp_size]])
            else:
                loss_batch = criterion(pred[:len(src_x), 0], src_y)
            loss += loss_batch.item() / train_size * len(src_x)

            if args.method in ['dann', 'cdan']:
                loss_batch = loss_batch - 1e2 * get_lambda(epoch, args.ep) * d_loss  # add the adversarial loss
                discriminator.zero_grad()
                step += 1
            elif args.method == 'jdot':
                args.ot_weight = 1e2
                loss_batch += args.ot_weight * ot_loss      # add the OT loss
            elif args.method == 'mrot':
                reg_loss = loss_batch.item()
                ot_loss = 1e-1 * ot_loss
                triplet_loss = 1e2 * triplet_loss
                loss_batch += ot_loss + triplet_loss      # add the OT loss

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

        val_metric = test(model, val_loader, val_size)
        lr_scheduler.step(val_metric)
        if val_metric < best_metric:
            best_ep = epoch + 1
            best_metric = val_metric
            if args.save: best_model = model
        if loss < best_loss:
            best_loss = loss
            early_stop = 0
        else:
            early_stop += 1
        if args.method == 'mrot':
            log.logger.info(
                'Epoch: {} | Time: {:.1f}s | Best Epoch: {} | reg_loss: {:.3f} | ot_loss: {:.3f} | triplet_loss: {:.3f} | Val_Metric: {:.3f} | Lr: {:.3f}'.format(
                    epoch + 1, time() - t1, best_ep, reg_loss, ot_loss.item(), triplet_loss.item(), val_metric, optimizer.param_groups[0]['lr'] * 1e5))
        else:
            log.logger.info('Epoch: {} | Time: {:.1f}s | Best Epoch: {} | Loss: {:.3f} | Test_Metric: {:.3f} | Lr: {:.3f}'
                            .format(epoch + 1, time() - t1, best_ep, loss, val_metric, optimizer.param_groups[0]['lr'] * 1e5))
        if early_stop >= 30:
            log.logger.info('Early Stopping!!! No Improvement on Loss for 30 Epochs.')
            break
    final_metric = test(model, test_loader, len(test_dataset))
    log.logger.info(f'{"=" * 20} End Training (Time: {(time() - t0) / 3600:.2f}h) {"=" * 20}\nTest Metric for {args.method} in {args.data}: {final_metric}.')
    if args.save: torch.save(best_model.state_dict(), f'../save/model_{args.data}_{args.method}.pt')


if __name__ == '__main__':
    main()  # print()
