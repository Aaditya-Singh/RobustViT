import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from ViT.helpers import init_distributed, LinearClassifier
from torch.nn.parallel import DistributedDataParallel

# Uncomment the expected model below

# ViT
# from ViT.ViT import vit_base_patch16_224 as vit
# from ViT.ViT import vit_large_patch16_224 as vit

# ViT-AugReg
# from ViT.ViT_new import vit_small_patch16_224 as vit
# from ViT.ViT_new import vit_base_patch16_224 as vit
# from ViT.ViT_new import vit_large_patch16_224 as vit

# DeiT
from ViT.helpers import load_ssl_pretrained
from ViT.ViT import deit_base_patch16_224 as deit
# from ViT.ViT import deit_small_patch16_224 as vit

from robustness_dataset import RobustnessDataset
from objectnet_dataset import ObjectNetDataset

# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))
# model_names.append("vit")

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to resume checkpoint (default: none)')
parser.add_argument('--port', default=40111, type=int,
                    help='port for launching distributed training')                    
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_on_itp', action='store_true')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
# parser.add_argument('--dist-backend', default='nccl', type=str,
#                     help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--mp_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument("--isV2", default=False, action='store_true',
                    help='is dataset imagenet V2.')
parser.add_argument("--isSI", default=False, action='store_true',
                    help='is dataset SI-score.')
parser.add_argument("--isObjectNet", default=False, action='store_true',
                    help='is dataset SI-score.')
parser.add_argument('--num_classes', default=1000, type=int,
                    help='number of classes for classification.')                    
parser.add_argument('--model_name', type=str, default='deit_small',
    help='model architecture')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    num_gpus = torch.cuda.device_count()
    args.mp_distributed = True if num_gpus > 1 else False

    if args.mp_distributed:
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=num_gpus, args=(args.port, num_gpus, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(rank, port, world_size, args):
    global best_acc1

    if args.gpu is not None:
        device = torch.device('cuda:0')
        print("Use GPU: {} for training".format(args.gpu))

    if args.mp_distributed:
        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
        os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
        world_size, rank = init_distributed(port=port, \
            rank_and_world_size=(rank, world_size))

    # create model
    print("=> creating model")
    if args.checkpoint:
        # NOTE: Load pretrained weights and match MSN's norm and classifier
        model = deit(pretrained=False)
        model.norm = None
        emb_dim = 384 if 'small' in args.model_name else 768 if 'base' in \
            args.model_name else 1024
        model.head = LinearClassifier(dim=emb_dim, num_labels=args.num_classes)
        load_ssl_pretrained(model, args.checkpoint)
    else:
        model = deit(pretrained=True)
    print("=> model created")

    device = None
    if args.gpu is not None:
        device = args.gpu
        model.to(device)
    elif args.mp_distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        device = torch.device('cuda:0')
        model.to(device)
        model = DistributedDataParallel(model)
    else:
        device = 'cpu'
        model.to(device) 
        print('using CPU, this will be slow')

    cudnn.benchmark = True

    if args.isObjectNet:
        val_dataset = ObjectNetDataset(args.data)
    else:
        val_dataset = RobustnessDataset(args.data, isV2=args.isV2, isSI=args.isSI)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    validate(val_loader, model, args, device)
    return


def validate(val_loader, model, args, device='cpu'):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)

            # measure accuracy only
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
