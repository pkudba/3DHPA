#!/usr/bin/env python3
import os.path
import time
import copy
import torch
from scripts.z_utils import AverageMeter, ProgressMeter
from datasets.partnet import DATA_FEATURES as data_features
from scripts.vis import gt_vis, pred_pose_vis


def eval_func(val_loader, model, log_writer, args):
    print("Begin_eval...")
    batch_time = AverageMeter('Time', ':6.4f')
    data_time = AverageMeter('Data', ':6.4f')

    trans_l2_loss_dist = AverageMeter('Translation L2 Loss', ':6.4f')
    cdsV1_loss_dist = AverageMeter('QDS Loss', ':6.7f')
    cdsV2_loss_dist = AverageMeter('WQDS Loss', ':6.7f')
    rot_cd_loss_dist = AverageMeter('Rotation CD Loss', ':6.4f')
    part_cd_loss_dist = AverageMeter('Part CD Loss', ':6.4f')
    shape_chamfer_dist = AverageMeter('Shape Chamfer Distance', ':6.4f')
    part_acc = AverageMeter('Part Accuracy', ':6.4f')
    connectivity_acc = AverageMeter('Connectivity Accuracy', ':6.4f')

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, trans_l2_loss_dist, rot_cd_loss_dist, part_cd_loss_dist,
         shape_chamfer_dist, part_acc, connectivity_acc, cdsV1_loss_dist, cdsV2_loss_dist
         ],
        prefix="TransAssembly Inference: ")

    # switch to eval mode.
    model.eval()

    end = time.time()
    val_num_batch = len(val_loader)

    sum_cdsV1_sum = 0.
    sum_cdsV2_sum = 0.

    sum_part_cd_loss = 0.
    sum_shape_cd_loss = 0.
    sum_rot_cd_loss = 0.
    sum_part_cd_loss = 0
    sum_trans_l2_loss = 0.
    sum_contact_point_loss = 0.
    total_acc_part = 0.
    total_valid_part = 0.
    total_contact_correct = 0.
    total_contact_point = 0.
    num_ins = 0
    real_val_data_set = 0
    for i, batch_data in enumerate(val_loader):
        # measure data loading time.
        data_time.update(time.time() - end)

        # data pre-process.
        if not batch_data:
            continue

        # ground truth visualization.
        if args.gt_vis:
            root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
            if root_dir not in args.gt_vis_dir:
                args.gt_vis_dir = os.path.join(root_dir, args.gt_vis_dir)

            part_pcs = batch_data[data_features.index('part_pcs')]
            part_valids = batch_data[data_features.index('part_valids')]
            part_poses = batch_data[data_features.index('part_poses')]
            part_ids = batch_data[data_features.index('part_ids')]
            cur_batch_size = gt_vis(part_pcs, part_poses, part_valids, num_ins, args=args)
            num_ins += cur_batch_size
            # continue

        # process.
        part_pcs = torch.cat(batch_data[data_features.index('part_pcs')], dim=0) 
        part_valids = torch.cat(batch_data[data_features.index('part_valids')], dim=0) 
        gt_part_poses = torch.cat(batch_data[data_features.index('part_poses')], dim=0)  
        match_ids = batch_data[data_features.index('match_ids')]  
        part_ids = torch.cat(batch_data[data_features.index('part_ids')], dim=0)  
        contact_points = torch.cat(batch_data[data_features.index('contact_points')], dim=0)  
        sym_info = torch.cat(batch_data[data_features.index('sym')], dim=0)  

        # print("Processing...")

        if args.gpu is not None:
            part_pcs = part_pcs.cuda(args.gpu, non_blocking=True)
            part_valids = part_valids.cuda(args.gpu, non_blocking=True)
            gt_part_poses = gt_part_poses.cuda(args.gpu, non_blocking=True)
            part_ids = part_ids.cuda(args.gpu, non_blocking=True)
            contact_points = contact_points.cuda(args.gpu, non_blocking=True)
            sym_info = sym_info.cuda(args.gpu, non_blocking=True)

        # compute output.
        with torch.no_grad():
            trans_l2_loss, rot_cd_loss, part_cd_loss, shape_cd_loss, contact_point_loss, acc_per_batch, valid_per_batch, \
                contact_correct_per_batch, contact_point_per_batch, batch_size, output, cdsV1_sum, cdsV2_sum \
                = model(part_pcs, part_valids, gt_part_poses, match_ids, part_ids,
                        contact_points=contact_points, sym_info=sym_info)
       
            if args.pred_encoder_vis:
                root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
                if root_dir not in args.pred_encoder_vis_dir:
                    args.pred_encoder_vis_dir = os.path.join(root_dir, args.pred_encoder_vis_dir)

                pred_poses = output["pred_poses"]
                _ = pred_pose_vis(part_pcs, pred_poses, part_valids, num_ins, args=args)

            # cal.
            sum_cdsV1_sum += cdsV1_sum
            sum_cdsV2_sum += cdsV2_sum

            sum_part_cd_loss += part_cd_loss * batch_size
            sum_shape_cd_loss += shape_cd_loss * batch_size
            sum_rot_cd_loss += rot_cd_loss * batch_size
            sum_trans_l2_loss += trans_l2_loss * batch_size
            sum_contact_point_loss += contact_point_loss * batch_size

            total_acc_part += acc_per_batch
            total_valid_part += valid_per_batch
            total_contact_correct += contact_correct_per_batch
            total_contact_point += contact_point_per_batch
            num_ins += batch_size
            real_val_data_set += part_pcs.shape[0]

        # measure elapsed time.
        batch_time.update(time.time() - end)
        end = time.time()
    
        shape_chamfer_dist.update(shape_cd_loss.item(), batch_size)
        trans_l2_loss_dist.update(trans_l2_loss.item(), batch_size)
        rot_cd_loss_dist.update(rot_cd_loss.item(), batch_size)
        part_cd_loss_dist.update(part_cd_loss.item(), batch_size)
        part_acc.update((acc_per_batch / valid_per_batch).item() * 100., batch_size)

        cdsV1_loss_dist.update(cdsV1_sum.item(), part_pcs.shape[0])
        cdsV2_loss_dist.update(cdsV2_sum.item(), part_pcs.shape[0])

        connectivity_acc.update((contact_correct_per_batch / contact_point_per_batch).item() * 100., batch_size)
        if i % args.print_freq == 0:
            infos = progress.display(i)

    # compute results.
    res_shape_cd = sum_shape_cd_loss / num_ins
    res_part_acc = total_acc_part / total_valid_part * 100.
    res_contact_acc = total_contact_correct / total_contact_point * 100.
    real_total_cdsV1 = sum_cdsV1_sum / real_val_data_set
    real_total_cdsV2 = sum_cdsV2_sum / real_val_data_set
    print("==========================================================")
    print("Shape Chamfer Distance: {}".format(res_shape_cd.item()))
    print("Part Accuracy: {}".format(res_part_acc.item()))
    print("Connectivity Accuracy: {}".format(res_contact_acc.item()))
    print(f'QDS: {real_total_cdsV1.item():.9f}')
    print(f'WQDS: {real_total_cdsV2.item():.9f}')
    print("==========================================================")

    # logger.
    res_info = "\n==========================================================" + "\n" + \
               "Shape Chamfer Distance: {}".format(res_shape_cd.item()) + "\t" + \
               "Part Accuracy: {}".format(res_part_acc.item()) + "\t" + \
               "Connectivity Accuracy: {}".format(res_contact_acc.item()) + "\n" + \
               "==========================================================\n"
    log_writer.write(res_info + "\n")
    log_writer.flush()
    return res_part_acc.item()











