#!/usr/bin/env python3

import os
import os.path as osp
import math
import random
import warnings
import shutil
import time
import datetime
from functools import partial
import numpy as np
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
from collections import OrderedDict


def func_init(args):
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
        print("Use GPU: {} for training".format(args.gpu))


def device_func(model, args):
    ngpus_per_node = torch.cuda.device_count()
    if not torch.cuda.is_available():
        print('using CPU, this will be slow.')
    elif args.distributed:
        # apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        # raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather/rank implementation in this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    print(model)  # print model after SyncBatchNorm.
    return model


def adjust_learning_rate(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    lr = args.lr
    if epoch < args.warmup_epochs:
        lr = lr * epoch / args.warmup_epochs
    else:
        lr = lr * 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_moco_momentum(epoch, args, q_on=True):
    """Adjust moco momentum based on current epoch"""
    if q_on:
        moco_m = args.moco_m_q
    else:
        moco_m = args.moco_m_k

    m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / args.epochs)) * (1. - moco_m)
    return m


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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
    
    def newupdate(self, acc, valid):
        self.val = acc
        self.sum += acc
        self.count += valid
        self.avg = self.sum / self.count *100.

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class EtaMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = ''
        self.ep = ''

    def update(self, val):
        self.val = val
        self.avg = str(datetime.timedelta(seconds=int(val)))

    def update_ep(self, countdown):
        self.countdown = str(datetime.timedelta(seconds=int(countdown)))

    def __str__(self):
        fmtstr = '{name} {avg} {countdown}'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def append(self, meter):
        self.meters.append(meter)

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        return '\t'.join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def collate_feats_with_none(b):
    b = filter(lambda x: x is not None, b)
    return list(zip(*b))


def save_model(epoch, scaler, output_dir, model, optimizer, part_acc, best_acc, args):
    if (epoch + 1) % args.save_freq == 0 or epoch + 1 == args.epochs:
        if args.rank == 0:  # only the first GPU saves checkpoint.
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'part_acc': part_acc
            }, is_best=False, filename=osp.join(output_dir, 'checkpoint_%04d.pth.tar' % (epoch + 1)))
        if part_acc > best_acc:
            if args.rank == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'part_acc': part_acc
                }, is_best=True, filename=osp.join(output_dir, 'checkpoint_best.pth.tar'))

def load_old_model(model, args, resume_on=False):
    if resume_on:
        args.checkpoint = args.resume

    if os.path.isfile(args.checkpoint):
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        if args.gpu is None:
            checkpoint = torch.load(args.checkpoint)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.checkpoint, map_location=loc)
            print(checkpoint['opt_network']['param_groups'])
            exit()
        try:
            model.load_state_dict(checkpoint['opt_network'])
        except:
            old_state_dict = checkpoint['state_dict']
            new_state_dict = OrderedDict()
            for key, value in old_state_dict.items():
                if "module" in key:
                    key = ".".join(key.split(".")[1:])  # remove "module."
                new_state_dict[key] = value
            if resume_on:
                model.load_state_dict(new_state_dict, strict=False)
            else:
                model.load_state_dict(new_state_dict)
        print("=> loaded checkpoint '{}' (epoch {})".format(args.checkpoint, checkpoint['epoch']))
        return model
    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))


def load_model(model, args, resume_on=False):
    if resume_on:
        args.checkpoint = args.resume

    if os.path.isfile(args.checkpoint):
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        if args.gpu is None:
            checkpoint = torch.load(args.checkpoint)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.checkpoint, map_location=loc)
        try:
            model.load_state_dict(checkpoint['state_dict'])
            # model.load_state_dict(checkpoint)
        except:
            old_state_dict = checkpoint['state_dict']
            new_state_dict = OrderedDict()
            for key, value in old_state_dict.items():
                if "module" in key:
                    key = ".".join(key.split(".")[1:])  # remove "module."
                new_state_dict[key] = value
            if resume_on:
                model.load_state_dict(new_state_dict, strict=False)
            else:
                model.load_state_dict(new_state_dict)
        print("=> loaded checkpoint '{}' (epoch {})".format(args.checkpoint, checkpoint['epoch']))
        return model
    else:
        print("=> no checkpoint found at '{}'".format(args.checkpoint))

