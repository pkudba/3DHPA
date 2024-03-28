#!/usr/bin/env python3
import time
import copy
import torch
from scripts.z_utils import EtaMeter, AverageMeter, ProgressMeter, adjust_learning_rate, adjust_moco_momentum
from datasets.partnet import DATA_FEATURES as data_features
from models.losses import comp_losses


def train_one_epoch(train_loader, model, optimizer, scaler, summary_writer, log_writer, epoch, countdown, args):

    eta_time = EtaMeter('Eta')
    batch_time = AverageMeter('Time', ':6.6f')
    data_time = AverageMeter('Data', ':6.6f')

    learning_rates = AverageMeter('lr', ':6.6f')

    losses = AverageMeter('TotalLoss', ':6.6f')
    trans_l2_losses = AverageMeter('TransL2Loss', ':6.6f')
    rot_l2_losses = AverageMeter('RotL2Loss', ':6.6f')
    trans_cd_losses = AverageMeter('TransCDLoss', ':6.6f')
    rot_cd_losses = AverageMeter('RotCDLoss', ':6.6f')
    shape_cd_losses = AverageMeter('ShapeCDLoss', ':6.6f')
    part_cd_losses = AverageMeter('PartCDLoss', ':6.6f')
    pointnet_losses = AverageMeter('PointNetLoss', ':6.6f')

    progress = ProgressMeter(
        len(train_loader),
        [eta_time, batch_time, data_time,learning_rates, losses,
         trans_l2_losses, rot_l2_losses, trans_cd_losses,
         rot_cd_losses, shape_cd_losses, part_cd_losses,
         pointnet_losses],
        prefix="Epoch: [{}]".format(epoch + 1))  

    # switch to train mode.
    model.train()

    end = time.time()
    iters_per_epoch = len(train_loader)
    for i, batch_data in enumerate(train_loader):
        # measure data loading time.
        data_time.update(time.time() - end)

        # adjust learning rate and momentum coefficient per iteration.
        if args.type_sched == "cosine":
            lr = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, args)
        learning_rates.update(optimizer.param_groups[0]["lr"])

        # data pre-process.
        part_pcs = torch.cat(batch_data[data_features.index('part_pcs')], dim=0)  
        part_valids = torch.cat(batch_data[data_features.index(r'part_valids')], dim=0)  
        gt_part_poses = torch.cat(batch_data[data_features.index('part_poses')], dim=0)  
        match_ids = batch_data[data_features.index('match_ids')]  
        part_ids = torch.cat(batch_data[data_features.index('part_ids')], dim=0) 
        contact_points = torch.cat(batch_data[data_features.index('contact_points')], dim=0) 

        if args.gpu is not None:
            part_pcs = part_pcs.cuda(args.gpu, non_blocking=False)
            part_valids = part_valids.cuda(args.gpu, non_blocking=False)
            gt_part_poses = gt_part_poses.cuda(args.gpu, non_blocking=False)
            part_ids = part_ids.cuda(args.gpu, non_blocking=False)

        # compute output.
        with torch.cuda.amp.autocast(True):
            pred_part_poses, loss, trans_l2_loss, rot_l2_loss, trans_cd_loss, rot_cd_loss, shape_cd_loss, part_cd_loss, pointnet_loss, output \
                = model(part_pcs, part_valids, gt_part_poses, match_ids, part_ids, contact_points)

        # compute gradient and do SGD step.
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=20.0)
        scaler.step(optimizer)
        scaler.update()

        batch_size = part_pcs.size(0)
        losses.update(loss.item(), batch_size)
        trans_l2_losses.update(trans_l2_loss.item(), batch_size)
        rot_l2_losses.update(rot_l2_loss.item(), batch_size)
        trans_cd_losses.update(trans_cd_loss.item(), batch_size)
        rot_cd_losses.update(rot_cd_loss.item(), batch_size)
        shape_cd_losses.update(shape_cd_loss, batch_size)
        part_cd_losses.update(part_cd_loss, batch_size)
        pointnet_losses.update(pointnet_loss, batch_size)

        if args.rank == 0:
            summary_writer.add_scalar("total_loss", loss.item(), epoch * iters_per_epoch + i)
            summary_writer.add_scalar("total_loss_per_epoch", losses.avg, epoch + 1)  # per epoch.
            summary_writer.add_scalar("trans_l2_loss", trans_l2_loss.item(), epoch * iters_per_epoch + i)
            summary_writer.add_scalar("trans_l2_loss_per_epoch", trans_l2_losses.avg, epoch + 1)
            summary_writer.add_scalar("rot_l2_loss", rot_l2_loss.item(), epoch * iters_per_epoch + i)
            summary_writer.add_scalar("rot_l2_loss_per_epoch", rot_l2_losses.avg, epoch + 1)
            summary_writer.add_scalar("trans_cd_loss", rot_cd_loss.item(), epoch * iters_per_epoch + i)
            summary_writer.add_scalar("trans_cd_loss_per_epoch", rot_cd_losses.avg, epoch + 1)
            summary_writer.add_scalar("rot_cd_loss", rot_cd_loss.item(), epoch * iters_per_epoch + i)
            summary_writer.add_scalar("rot_cd_loss_per_epoch", rot_cd_losses.avg, epoch + 1)
            summary_writer.add_scalar("shape_cd_loss", shape_cd_loss.item(), epoch * iters_per_epoch + i)
            summary_writer.add_scalar("shape_cd_loss_per_epoch", shape_cd_losses.avg, epoch + 1)
            summary_writer.add_scalar("part_cd_loss", part_cd_loss.item(), epoch * iters_per_epoch + i)
            summary_writer.add_scalar("part_cd_loss_per_epoch", part_cd_losses.avg, epoch + 1)
            summary_writer.add_scalar("pointnet_loss", pointnet_loss.item(), epoch * iters_per_epoch + i)
            summary_writer.add_scalar("pointnet_loss_per_epoch", pointnet_losses.avg, epoch + 1)

        # measure elapsed time.
        batch_time.update(time.time() - end)
        end = time.time()

        # measure eta time.
        eta_time.update(batch_time.avg * (len(train_loader) - i))
        ep_countdown = batch_time.avg if countdown == 0 else countdown
        eta_time.update_ep(ep_countdown * (len(train_loader) * (args.epochs - epoch - 1) + len(train_loader) - i))

        if i % args.print_freq == 0:
            infos = progress.display(i)
            log_writer.write(infos + "\n")
            log_writer.flush()

    return batch_time.avg
