"""
Video Face Manipulation Detection Through Ensemble of CNNs

Image and Sound Processing Lab - Politecnico di Milano

Nicolò Bonettini
Edoardo Daniele Cannas
Sara Mandelli
Luca Bondi
Paolo Bestagini
"""
import argparse
import os
import shutil
import warnings

import albumentations as A
import numpy as np
import torch
import torch.multiprocessing

from isplutils import utils, split

torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import ImageChops, Image

from core.config import create_log
from architectures import fornet
from isplutils.data import FrameFaceIterableDataset, load_face
from isplutils.utils import save_model
from core.train_val import validation, batch_forward, tb_attention
from test_model import test


def main():
    # Args
    parser = argparse.ArgumentParser()
    # 命名规定：网络名称（用net拼接）-其他实验因素，下划线区分
    parser.add_argument('--exp_id', type=str, default='SgeMspNet-sge_msp')
    parser.add_argument('--env', type=int, default=0)
    parser.add_argument('--device', type=int, help='GPU device id', default=0)
    parser.add_argument('--net', type=str, help='Net model class', default='SgeMspNet')
    parser.add_argument('--traindb', type=list, help='Training datasets', nargs='+', choices=split.available_datasets,
                        default=['ff-c40-720-140-140'])
    parser.add_argument('--valdb', type=list, help='Validation datasets', nargs='+', choices=split.available_datasets,
                        default=['ff-c40-720-140-140'])
    parser.add_argument('--ffpp_faces_df_path', type=str, action='store',
                        help='Path to the Pandas Dataframe obtained from extract_faces.py on the FF++ dataset. '
                             'Required for training/validating on the FF++ dataset.',
                        default='FF/preprocess/facesDataFrames')
    parser.add_argument('--ffpp_faces_dir', type=str, action='store',
                        help='Path to the directory containing the faces extracted from the FF++ dataset. '
                             'Required for training/validating on the FF++ dataset.',
                        default='FF/preprocess/faces')
    parser.add_argument('--face', type=str, help='Face crop or scale',
                        choices=['scale', 'tight'],
                        default='scale')
    parser.add_argument('--size', type=int, help='Train patch size', default=224)

    parser.add_argument('--batch', type=int, help='Batch size to fit in GPU memory', default=18)
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--valint', type=int, help='Validation interval (iterations)', default=500)
    parser.add_argument('--patience', type=int, help='Patience before dropping the LR [validation intervals]',
                        default=10)
    parser.add_argument('--maxiter', type=int, help='Maximum number of iterations', default=40000)
    parser.add_argument('--init', type=str, help='Weight initialization file')
    parser.add_argument('--scratch', action='store_true', help='Train from scratch')

    parser.add_argument('--trainsamples', type=int, help='Limit the number of train samples per epoch', default=-1)
    parser.add_argument('--valsamples', type=int, help='Limit the number of validation samples per epoch',
                        default=6000)

    parser.add_argument('--logint', type=int, help='Training log interval (iterations)', default=100)
    parser.add_argument('--workers', type=int, help='Num workers for data loaders', default=8)
    parser.add_argument('--seed', type=int, help='Random seed', default=41)

    parser.add_argument('--debug', action='store_true', help='Activate debug')
    parser.add_argument('--suffix', type=str, help='Suffix to default tag')

    parser.add_argument('--attention', action='store_true',
                        help='Enable Tensorboard log of attention masks')
    parser.add_argument('--output_dir', type=str, help='Directory for saving the training logs',
                        default='output')
    # --------------------------------------------------------------------
    parser.add_argument('--dfdc_faces_df_path', type=str, action='store',
                        help='Path to the Pandas Dataframe obtained from extract_faces.py on the DFDC dataset. '
                             'Required for training/validating on the DFDC dataset.')
    parser.add_argument('--dfdc_faces_dir', type=str, action='store',
                        help='Path to the directory containing the faces extracted from the DFDC dataset. '
                             'Required for training/validating on the DFDC dataset.')

    args = parser.parse_args()

    # Parse arguments
    ds_root = ['F:/ggy/dataset', '/home/disk/weixing/datasets']
    ffpp_df_path = os.path.join(ds_root[args.env], args.ffpp_faces_df_path)
    ffpp_faces_dir = os.path.join(ds_root[args.env], args.ffpp_faces_dir)
    dfdc_df_path = args.dfdc_faces_df_path
    dfdc_faces_dir = args.dfdc_faces_dir

    net_class = getattr(fornet, args.net)
    train_datasets = args.traindb
    val_datasets = args.valdb
    face_policy = args.face
    face_size = args.size

    batch_size = args.batch
    initial_lr = args.lr
    validation_interval = args.valint
    patience = args.patience
    max_num_iterations = args.maxiter
    initial_model = args.init
    train_from_scratch = args.scratch

    max_train_samples = args.trainsamples
    max_val_samples = args.valsamples

    log_interval = args.logint
    num_workers = args.workers
    args.device = 0 if args.env == 0 else args.device
    device = torch.device('cuda:{:d}'.format(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    seed = args.seed

    # log config
    logger, weights_dir, log_dir = create_log(args.output_dir, args.exp_id)
    for arg in vars(args):
        logger.info(format(arg, '<20')  + ' ' + format(str(getattr(args, arg)), '<'))   # str, arg_type

    debug = args.debug
    suffix = args.suffix

    enable_attention = args.attention
    logger.info("=====> enable_attention: {}".format(enable_attention))


    # Random initialization
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    # Load net
    net: nn.Module = net_class().to(device)

    dummy = torch.randn((18, 3, face_size, face_size), device=device)
    dummy = dummy.to(device)
    net(dummy)
    # Loss and optimizers
    criterion = nn.BCEWithLogitsLoss()

    min_lr = initial_lr * 1e-5
    optimizer = optim.Adam(net.get_trainable_parameters(), lr=initial_lr)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        factor=0.1,
        patience=patience,
        cooldown=2 * patience,
        min_lr=min_lr,
    )

    # tag = utils.make_train_tag(net_class=net_class,
    #                            traindb=train_datasets,
    #                            face_policy=face_policy,
    #                            patch_size=face_size,
    #                            seed=seed,
    #                            suffix=suffix,
    #                            debug=debug,
    #                            )

    # Model checkpoint paths
    bestval_path = os.path.join(weights_dir, 'bestval.pth')
    last_path = os.path.join(weights_dir, 'last.pth')
    periodic_path = os.path.join(weights_dir, 'it{:06d}.pth')

    # Load model
    val_loss = min_val_loss = 10
    epoch = iteration = 0
    net_state = None
    opt_state = None
    if initial_model is not None:
        # If given load initial model
        logger.info('Loading model form: {}'.format(initial_model))
        state = torch.load(initial_model, map_location='cpu')
        net_state = state['net']
    # 自动恢复训练，从：last
    elif not train_from_scratch and os.path.exists(last_path):
        logger.info('Loading model form: {}'.format(last_path))
        state = torch.load(last_path, map_location='cpu')
        net_state = state['net']
        opt_state = state['opt']
        iteration = state['iteration'] + 1
        epoch = state['epoch']
    # 取到最小 val_loss?
    if not train_from_scratch and os.path.exists(bestval_path):
        state = torch.load(bestval_path, map_location='cpu')
        min_val_loss = state['val_loss']

    if net_state is not None:
        incomp_keys = net.load_state_dict(net_state, strict=False)
        logger.info(incomp_keys)
    if opt_state is not None:
        for param_group in opt_state['param_groups']:
            param_group['lr'] = initial_lr
        optimizer.load_state_dict(opt_state)

    # Initialize Tensorboard
    if iteration == 0:
        # If training from scratch or initialization remove history if exists
        shutil.rmtree(log_dir, ignore_errors=True)

    # TensorboardX instance
    tb = SummaryWriter(logdir=log_dir)
    if iteration == 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # tb.add_graph(net, [dummy, ], verbose=False)

    transformer = utils.get_transformer(face_policy=face_policy, patch_size=face_size,
                                        net_normalizer=net.get_normalizer(), train=True)

    # Datasets and data loaders
    logger.info('Loading data')
    # Check if paths for DFDC and FF++ extracted faces and DataFrames are provided
    for dataset in train_datasets:
        if dataset.split('-')[0] == 'dfdc' and (dfdc_df_path is None or dfdc_faces_dir is None):
            raise RuntimeError('Specify DataFrame and directory for DFDC faces for training!')
        elif dataset.split('-')[0] == 'ff' and (ffpp_df_path is None or ffpp_faces_dir is None):
            raise RuntimeError('Specify DataFrame and directory for FF++ faces for training!')
    for dataset in val_datasets:
        if dataset.split('-')[0] == 'dfdc' and (dfdc_df_path is None or dfdc_faces_dir is None):
            raise RuntimeError('Specify DataFrame and directory for DFDC faces for validation!')
        elif dataset.split('-')[0] == 'ff' and (ffpp_df_path is None or ffpp_faces_dir is None):
            raise RuntimeError('Specify DataFrame and directory for FF++ faces for validation!')
    # Load splits with the make_splits function
    splits = split.make_splits(dfdc_df=dfdc_df_path, ffpp_df=ffpp_df_path, dfdc_dir=dfdc_faces_dir, ffpp_dir=ffpp_faces_dir,
                               dbs={'train': train_datasets, 'val': val_datasets})
    train_dfs = [splits['train'][db][0] for db in splits['train']]
    train_roots = [splits['train'][db][1] for db in splits['train']]
    val_roots = [splits['val'][db][1] for db in splits['val']]
    val_dfs = [splits['val'][db][0] for db in splits['val']]

    train_dataset = FrameFaceIterableDataset(roots=train_roots,
                                             dfs=train_dfs,
                                             scale=face_policy,
                                             num_samples=max_train_samples,
                                             transformer=transformer,
                                             size=face_size,
                                             )

    val_dataset = FrameFaceIterableDataset(roots=val_roots,
                                           dfs=val_dfs,
                                           scale=face_policy,
                                           num_samples=max_val_samples,
                                           transformer=transformer,
                                           size=face_size,
                                           )

    train_loader = DataLoader(train_dataset, num_workers=num_workers, batch_size=batch_size, )

    val_loader = DataLoader(val_dataset, num_workers=num_workers, batch_size=batch_size, )

    logger.info('Training samples: {}'.format(len(train_dataset)))
    logger.info('Validation samples: {}'.format(len(val_dataset)))

    if len(train_dataset) == 0:
        logger.info('No training samples. Halt.')
        return

    if len(val_dataset) == 0:
        logger.info('No validation samples. Halt.')
        return

    stop = False
    while not stop:

        # Training
        optimizer.zero_grad()

        train_loss = train_num = 0
        train_pred_list = []
        train_labels_list = []
        logger.info("======== training: {} th epoch".format(epoch))
        train_loader = tqdm(train_loader, leave=False)
        for train_batch in train_loader:
            net.train()
            batch_data, batch_labels = train_batch

            train_batch_num = len(batch_labels)
            train_num += train_batch_num
            train_labels_list.append(batch_labels.numpy().flatten())

            train_batch_loss, train_batch_pred = batch_forward(net, device, criterion, batch_data, batch_labels)
            train_pred_list.append(train_batch_pred.flatten())

            if torch.isnan(train_batch_loss):
                raise ValueError('NaN loss')

            train_loss += train_batch_loss.item() * train_batch_num

            # Optimization
            train_batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loader.set_description("Train Loss: {:.5f}".format(train_batch_loss.item()))
            # Logging
            if iteration > 0 and (iteration % log_interval == 0):
                train_loss /= train_num
                tb.add_scalar('train/loss', train_loss, iteration)
                tb.add_scalar('lr', optimizer.param_groups[0]['lr'], iteration)
                tb.add_scalar('epoch', epoch, iteration)

                # Checkpoint
                save_model(net, optimizer, train_loss, val_loss, iteration, batch_size, epoch, last_path)
                train_loss = train_num = 0

            # ----- Validation -----
            if iteration > 0 and (iteration % validation_interval == 0):

                # Model checkpoint
                save_model(net, optimizer, train_loss, val_loss, iteration, batch_size, epoch,
                           periodic_path.format(iteration))

                # Train cumulative stats
                train_labels = np.concatenate(train_labels_list)
                train_pred = np.concatenate(train_pred_list)
                train_labels_list = []
                train_pred_list = []

                train_roc_auc = roc_auc_score(train_labels, train_pred)
                tb.add_scalar('train/roc_auc', train_roc_auc, iteration)
                tb.add_pr_curve('train/pr', train_labels, train_pred, iteration)

                # Validation
                val_loss, val_roc_auc = validation(net, device, val_loader, criterion, tb, iteration, 'val')
                logger.info("val_loss: {}, auc: {}".format(val_loss, val_roc_auc))
                tb.flush()

                # LR Scheduler
                lr_scheduler.step(val_loss)

                # Model checkpoint
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    logger.info('=====> val_loss[{}] is mimimum, saving bestval. epoch: {}, iter: {}'
                                .format(val_loss, epoch, iteration))
                    save_model(net, optimizer, train_loss, val_loss, iteration, batch_size, epoch, bestval_path)

                # Attention
                if enable_attention and hasattr(net, 'get_attention'):
                    net.eval()
                    # For each dataframe show the attention for a real,fake couple of frames
                    for df, root, sample_idx, tag in [
                        (train_dfs[0], train_roots[0], train_dfs[0][train_dfs[0]['label'] == False].index[0],
                         'train/att/real'),
                        (train_dfs[0], train_roots[0], train_dfs[0][train_dfs[0]['label'] == True].index[0],
                         'train/att/fake'),
                    ]:
                        record = df.loc[sample_idx]
                        tb_attention(tb, tag, iteration, net, device, face_size, face_policy,
                                     transformer, root, record)

                if optimizer.param_groups[0]['lr'] == min_lr:
                    logger.info('Reached minimum learning rate. Stopping.')
                    stop = True
                    break

            iteration += 1

            if iteration > max_num_iterations:
                logger.info('Maximum number of iterations reached')
                stop = True
                break

            # End of iteration

        epoch += 1

    # Needed to flush out last events
    tb.close()

    logger.info('Train Completed')

    test(args.exp_id)


if __name__ == '__main__':
    main()
