import argparse
import os
import random
import shutil
import time
import warnings
import pprint
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from ViT.helpers import init_distributed, LinearClassifier
from torch.nn.parallel import DistributedDataParallel
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from tokencut_dataset import SegmentationDataset
from robustness_dataset import RobustnessDataset

# Uncomment the expected model below

# ViT
# from ViT.ViT import vit_base_patch16_224 as vit
# from ViT.ViT import vit_large_patch16_224 as vit

# ViT-AugReg
# from ViT.ViT_new import vit_small_patch16_224 as vit
# from ViT.ViT_new import vit_base_patch16_224 as vit
# from ViT.ViT_new import vit_large_patch16_224 as vit

from ViT.helpers import load_ssl_pretrained
from ViT.ViT import deit_base_patch16_224 as deit
# from ViT.ViT import deit_small_patch16_224 as vit

from ViT.explainer import generate_relevance, get_image_with_relevance
import torchvision
import cv2
from torch.utils.tensorboard import SummaryWriter
import json

# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))
# model_names.append("vit")

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DATA',
                    help='path to dataset')
parser.add_argument('--subset', metavar='SUBSET',
                    help='path to subset file')
parser.add_argument('--seg_data', metavar='SEG_DATA',
                    help='path to segmentation dataset')
parser.add_argument('--val_data', metavar='VAL_DATA',
                    help='path to validation dataset')                
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=10, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=3e-6, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', type=str, default=None,
                    help='path to pretrained model')
parser.add_argument('--linear_eval', dest='linear_eval', action='store_true',
                    help='whether to finetune head only')
parser.add_argument('--port', default=40111, type=int,
                    help='port for launching distributed training')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_on_itp', action='store_true')
parser.add_argument('--dist-url', default='env://', type=str,
                    help='url used to set up distributed training')
# parser.add_argument('--dist-backend', default='nccl', type=str,
#                     help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--save_interval', default=20, type=int,
                    help='interval to save segmentation results.')
parser.add_argument('--mp_distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--lambda_seg', default=0.1, type=float,
                    help='influence of segmentation loss.')
parser.add_argument('--lambda_acc', default=1, type=float,
                    help='influence of accuracy loss.')
parser.add_argument('--experiment_folder', default=None, type=str,
                    help='path to folder to use for experiment.')
parser.add_argument('--dilation', default=0, type=float,
                    help='Use dilation on the segmentation maps.')
parser.add_argument('--lambda_background', default=1, type=float,
                    help='coefficient of loss for segmentation background.')
parser.add_argument('--lambda_foreground', default=0.3, type=float,
                    help='coefficient of loss for segmentation foreground.')
parser.add_argument('--num_classes', default=1000, type=int,
                    help='number of classes for classification.')
parser.add_argument('--temperature', default=1, type=float,
                    help='temperature for softmax (mostly for DeiT).')
parser.add_argument('--model_name', type=str, default='deit_small',
    help='model architecture')
best_top1 = 0.0


def main():
    args = parser.parse_args()
    pprint.pprint(args)
    if args.experiment_folder is None:
        args.experiment_folder = f'experiment/' \
                                 f'lr_{args.lr}_seg_{args.lambda_seg}_acc_{args.lambda_acc}' \
                                 f'_bckg_{args.lambda_background}_fgd_{args.lambda_foreground}'
        if args.temperature != 1:
            args.experiment_folder = args.experiment_folder + f'_tempera_{args.temperature}'
        if args.batch_size != 8:
            args.experiment_folder = args.experiment_folder + f'_bs_{args.batch_size}'
        if args.num_classes != 500:
            args.experiment_folder = args.experiment_folder + f'_num_classes_{args.num_classes}'
        if args.num_samples != 3:
            args.experiment_folder = args.experiment_folder + f'_num_samples_{args.num_samples}'
        if args.epochs != 150:
            args.experiment_folder = args.experiment_folder + f'_num_epochs_{args.epochs}'

    os.makedirs(args.experiment_folder, exist_ok=True)
    os.makedirs(f'{args.experiment_folder}/train_samples', exist_ok=True)

    with open(f'{args.experiment_folder}/commandline_args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

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
        main_worker(args.gpu, num_gpus, args)


def main_worker(rank, port, world_size, args):
    global best_loss

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.mp_distributed:
        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes  
        world_size, rank = init_distributed(port=port, \
            rank_and_world_size=(rank, world_size))
    
    # create model
    print("=> creating model")
    # NOTE: Load pretrained weights and match MSN's norm and classifier
    model = deit(pretrained=False).to(rank)
    model.norm = None
    emb_dim = 384 if 'small' in args.model_name else 768 if 'base' in \
        args.model_name else 1024
    model.head = LinearClassifier(dim=emb_dim, num_labels=args.num_classes)
    load_ssl_pretrained(model, args.pretrained)
    if args.linear_eval:
        print("=> freezing model parameters except head")
        args.lambda_seg = 0.
        for n, p in model.named_parameters():
            if 'head' not in n and p.requires_grad: p.requires_grad_(False)        
    n_params = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)    
    print(f"=> model created with {n_params} trainable parameters")

    print("=> creating original model")
    orig_model = deit(pretrained=False).to(rank)
    load_ssl_pretrained(orig_model, args.pretrained)
    orig_model.eval()
    print("=> original model created")

    device = None
    if args.gpu is not None:
        device = args.gpu
        model.to(device)
        orig_model.to(device)
    elif args.mp_distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        device = rank
        model.to(device)
        orig_model.to(device)
        model = DistributedDataParallel(model)
        orig_model = DistributedDataParallel(orig_model)
    else:
        device = 'cpu'
        model.to(device)
        orig_model.to(device)        
        print('using CPU, this will be slow')

    # NOTE: define loss function and match MSN's optimizer configs
    criterion = nn.CrossEntropyLoss()
    if args.linear_eval:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, \
            nesterov=True, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), args.lr, \
            weight_decay=args.weight_decay)        

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            if args.gpu is not None:
                # best_loss may be from a checkpoint from a different GPU
                best_loss = best_loss.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    # NOTE: Load full ImageNet subsets
    train_dataset = SegmentationDataset(args.seg_data, args.data)

    if args.mp_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    # NOTE: use IN1k val set for validation instead of partial subset
    val_dataset = RobustnessDataset(args.val_data)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=10, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, 0, args, device)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.mp_distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        log_dir = os.path.join(args.experiment_folder, 'logs')
        logger = SummaryWriter(log_dir=log_dir)
        args.logger = logger

        # train for one epoch
        train(train_loader, model, orig_model, criterion, optimizer, epoch, args, device)

        # NOTE: evaluate on validation set after every #epochs//10 epochs
        if (epoch + 1) % (args.epochs // 10) == 0:
            top1 = validate(val_loader, model, orig_model, criterion, epoch, args, device)

            # NOTE: remember best acc@1 and save checkpoint
            is_best = top1 > best_top1
            best_top1 = max(top1, best_top1)

            if not args.mp_distributed or (args.mp_distributed and rank == 0):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, folder=args.experiment_folder)


def train(train_loader, model, orig_model, criterion, optimizer, epoch, args, device='cpu'):
    mse_criterion = torch.nn.MSELoss(reduction='mean')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    orig_top1 = AverageMeter('Acc@1_orig', ':6.2f')
    orig_top5 = AverageMeter('Acc@5_orig', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        # [batch_time, data_time, losses, top1, top5, orig_top1, orig_top5],
        [losses, top1, top5, orig_top1, orig_top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (seg_map, image_ten, class_name) in enumerate(train_loader):

        image_ten = image_ten.to(device)
        seg_map = seg_map.to(device)
        class_name = class_name.to(device)

        # segmentation loss
        if args.lambda_seg > 0.:
            model_wo_ddp = model.module if isinstance(model, DistributedDataParallel) else model  
            relevance = generate_relevance(model_wo_ddp, image_ten, index=class_name, device=device)

            reverse_seg_map = seg_map.clone()
            reverse_seg_map[reverse_seg_map == 1] = -1
            reverse_seg_map[reverse_seg_map == 0] = 1
            reverse_seg_map[reverse_seg_map == -1] = 0
            background_loss = mse_criterion(relevance * reverse_seg_map, torch.zeros_like(relevance))
            foreground_loss = mse_criterion(relevance * seg_map, seg_map)
            segmentation_loss = args.lambda_background * background_loss
            segmentation_loss += args.lambda_foreground * foreground_loss
        else:
            segmentation_loss = 0.

        # NOTE: classification loss w.r.t ground truth labels
        output = model(image_ten)
        if args.temperature != 1:
            output = output / args.temperature
        # _, pred = output.topk(1, 1, True, True)
        # pred = pred.flatten()
        # classification_loss = criterion(output, pred)
        classification_loss = criterion(output, class_name)

        loss = args.lambda_seg * segmentation_loss + args.lambda_acc * classification_loss

        # debugging output
        if i % args.save_interval == 0 and args.lambda_seg > 0.:
            orig_model_wo_ddp = orig_model.module if \
                isinstance(orig_model, DistributedDataParallel) else orig_model
            orig_relevance = generate_relevance(orig_model_wo_ddp, image_ten, \
                index=class_name, device=device)
            for j in range(image_ten.shape[0]):
                image = get_image_with_relevance(image_ten[j], torch.ones_like(image_ten[j]))
                new_vis = get_image_with_relevance(image_ten[j], relevance[j])
                old_vis = get_image_with_relevance(image_ten[j], orig_relevance[j])
                gt = get_image_with_relevance(image_ten[j], seg_map[j])
                h_img = cv2.hconcat([image, gt, old_vis, new_vis])
                cv2.imwrite(f'{args.experiment_folder}/train_samples/res_{i}_{j}.jpg', h_img)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, class_name, topk=(1, 5))
        losses.update(loss.item(), image_ten.size(0))
        top1.update(acc1[0], image_ten.size(0))
        top5.update(acc5[0], image_ten.size(0))

        # metrics for original vit
        with torch.no_grad():
            output_orig = orig_model(image_ten)        
        acc1_orig, acc5_orig = accuracy(output_orig, class_name, topk=(1, 5))
        orig_top1.update(acc1_orig[0], image_ten.size(0))
        orig_top5.update(acc5_orig[0], image_ten.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0:
            progress.display(i)
            args.logger.add_scalar('{}/{}'.format('train', 'segmentation_loss'), segmentation_loss,
                                   epoch*len(train_loader)+i)
            args.logger.add_scalar('{}/{}'.format('train', 'classification_loss'), classification_loss,
                                   epoch * len(train_loader) + i)
            args.logger.add_scalar('{}/{}'.format('train', 'orig_top1'), acc1_orig,
                                   epoch * len(train_loader) + i)
            args.logger.add_scalar('{}/{}'.format('train', 'top1'), acc1,
                                   epoch * len(train_loader) + i)
            args.logger.add_scalar('{}/{}'.format('train', 'orig_top5'), acc5_orig,
                                   epoch * len(train_loader) + i)
            args.logger.add_scalar('{}/{}'.format('train', 'top5'), acc5,
                                   epoch * len(train_loader) + i)
            args.logger.add_scalar('{}/{}'.format('train', 'tot_loss'), loss,
                                   epoch * len(train_loader) + i)


def validate(val_loader, model, orig_model, criterion, epoch, args, device='cpu'):
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    orig_top1 = AverageMeter('Acc@1_orig', ':6.2f')
    orig_top5 = AverageMeter('Acc@5_orig', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [losses, top1, top5, orig_top1, orig_top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            images = images.to(device)
            target = target.to(device)

            # NOTE: compute output and classification loss w.r.t class labels
            output = model(images)
            if args.temperature != 1:
                output = output / args.temperature
            classification_loss = criterion(output, target)
            loss = args.lambda_acc * classification_loss

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            losses.update(loss.item(), images.size(0))

            # metrics for original vit
            output_orig = orig_model(images)
            acc1_orig, acc5_orig = accuracy(output_orig, target, topk=(1, 5))
            orig_top1.update(acc1_orig[0], images.size(0))
            orig_top5.update(acc5_orig[0], images.size(0))

            if i % args.print_freq == 0:
                progress.display(i)
                args.logger.add_scalar('{}/{}'.format('val', 'classification_loss'), classification_loss,
                                       epoch * len(val_loader) + i)
                args.logger.add_scalar('{}/{}'.format('val', 'orig_top1'), acc1_orig,
                                       epoch * len(val_loader) + i)
                args.logger.add_scalar('{}/{}'.format('val', 'top1'), acc1,
                                       epoch * len(val_loader) + i)
                args.logger.add_scalar('{}/{}'.format('val', 'orig_top5'), acc5_orig,
                                       epoch * len(val_loader) + i)
                args.logger.add_scalar('{}/{}'.format('val', 'top5'), acc5,
                                       epoch * len(val_loader) + i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, folder, filename='checkpoint.pth.tar'):
    torch.save(state, f'{folder}/{filename}')
    if is_best:
        shutil.copyfile(f'{folder}/{filename}', f'{folder}/model_best.pth.tar')


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


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.85 ** (epoch // 2))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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