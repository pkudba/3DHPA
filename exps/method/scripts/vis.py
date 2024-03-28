#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../utils'))
from quaternion import qrot
import render_using_blender as render_utils


def gt_vis(part_pcs, part_poses, part_valids, num_ins, args):
    vis_root = os.path.join(args.gt_vis_dir, args.category)
    gt_dir = os.path.join(vis_root, "gt_assembly")
    part_dir = os.path.join(vis_root, "input_part_pcs")

    if not os.path.exists(gt_dir):
        os.makedirs(gt_dir)
    if not os.path.exists(part_dir):
        os.makedirs(part_dir)

    batch_size = len(part_pcs)
    for cur_idx in range(batch_size):
        data_name = args.category + "-%04d.png" % (num_ins + cur_idx)
        print("Processing: {}".format(data_name))

        cur_part_valid = int(part_valids[cur_idx].sum().item())
        input_part_pcs = part_pcs[cur_idx].squeeze(0)[:cur_part_valid]

        # process.
        cur_part_poses = part_poses[cur_idx].squeeze(0)[:cur_part_valid]
        cur_part_trans = cur_part_poses[:, :3].unsqueeze(1).repeat(1, input_part_pcs.size(1), 1)
        cur_part_rot = cur_part_poses[:, 3:].unsqueeze(1).repeat(1, input_part_pcs.size(1), 1)
        gt_part_pcs = qrot(cur_part_rot, input_part_pcs) + cur_part_trans

        # save.
        input_part_pcs = input_part_pcs.numpy()
        render_utils.render_part_pts(os.path.join(part_dir, data_name), input_part_pcs, blender_fn='object_centered.blend')

        gt_part_pcs = gt_part_pcs.numpy()
        render_utils.render_part_pts(os.path.join(gt_dir, data_name), gt_part_pcs, blender_fn='object_centered.blend')

    return batch_size


def pred_pose_vis(part_pcs, part_poses, part_valids, num_ins, args):
    vis_root = os.path.join(args.pred_encoder_vis_dir, args.category)
    pred_dir = os.path.join(vis_root, "pred_assembly")
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)

    batch_size, eval_mon, num_part, _ = part_poses.size()
    for bs_id in range(batch_size):
        data_dir = os.path.join(pred_dir, args.category + "-%04d" % (num_ins + bs_id))
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)

        cur_part_valid = int(part_valids[bs_id].sum().item())
        input_part_pcs = part_pcs[bs_id][:cur_part_valid]
        cur_part_poses = part_poses[bs_id][:, :cur_part_valid]
        for mon_id in range(args.num_pred_vis):
            data_name = args.category + "-%04d-%02d.png" % (num_ins + bs_id, mon_id)
            print("Processing: {}".format(data_name))

            # process.
            cur_part_trans = cur_part_poses[mon_id, :, :3].unsqueeze(1).repeat(1, input_part_pcs.size(1), 1)
            cur_part_rot = cur_part_poses[mon_id, :, 3:].unsqueeze(1).repeat(1, input_part_pcs.size(1), 1)
            pred_part_pcs = qrot(cur_part_rot, input_part_pcs) + cur_part_trans

            # save.
            pred_part_pcs = pred_part_pcs.cpu().detach().numpy()
            render_utils.render_part_pts(os.path.join(data_dir, data_name), pred_part_pcs, blender_fn='object_centered.blend')

    return batch_size


def gt_vis_individual(part_pcs, part_poses, part_valids, num_ins, args):
    import numpy as np
    vis_root = os.path.join(args.gt_vis_dir, args.category)
    gt_dir = os.path.join(vis_root, "gt_assembly")
    part_dir = os.path.join(vis_root, "individual_parts")

    if not os.path.exists(gt_dir):
        os.makedirs(gt_dir)
    if not os.path.exists(part_dir):
        os.makedirs(part_dir)

    batch_size = len(part_pcs)
    for cur_idx in range(batch_size):
        data_name = args.category + "-%04d.png" % (num_ins + cur_idx)
        print("Processing: {}".format(data_name))
        data_file = args.category + "-%04d" % (num_ins + cur_idx)
        data_dir = os.path.join(part_dir, data_file)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        cur_part_valid = int(part_valids[cur_idx].sum().item())
        input_part_pcs = part_pcs[cur_idx].squeeze(0)[:cur_part_valid]

        # process.
        cur_part_poses = part_poses[cur_idx].squeeze(0)[:cur_part_valid]
        cur_part_trans = cur_part_poses[:, :3].unsqueeze(1).repeat(1, input_part_pcs.size(1), 1)
        cur_part_rot = cur_part_poses[:, 3:].unsqueeze(1).repeat(1, input_part_pcs.size(1), 1)
        gt_part_pcs = qrot(cur_part_rot, input_part_pcs) + cur_part_trans

        # save.
        input_part_pcs = input_part_pcs.numpy()
        for part_id in range(cur_part_valid):
            part_id_file = os.path.join(data_dir, "part-%02d.png" % part_id)
            print("process {}".format(part_id_file))
            cur_part_pcs = input_part_pcs[part_id][np.newaxis, :, :]
            render_utils.render_part_pts(part_id_file, cur_part_pcs, blender_fn='object_centered.blend')

        gt_part_pcs = gt_part_pcs.numpy()
        render_utils.render_part_pts(os.path.join(gt_dir, data_name), gt_part_pcs, blender_fn='object_centered.blend')

    return batch_size


def gt_vis_individual_2(part_pcs, part_poses, part_valids, num_ins, args):
    import numpy as np
    import torch
    vis_root = os.path.join(args.gt_vis_dir, args.category)
    gt_dir = os.path.join(vis_root, "gt_assembly")
    part_dir = os.path.join(vis_root, "individual_parts_aaa")

    if not os.path.exists(gt_dir):
        os.makedirs(gt_dir)
    if not os.path.exists(part_dir):
        os.makedirs(part_dir)

    batch_size = len(part_pcs)
    for cur_idx in range(batch_size):
        data_name = args.category + "-%04d.png" % (num_ins + cur_idx)
        print("Processing: {}".format(data_name))
        data_file = args.category + "-%04d" % (num_ins + cur_idx)
        data_dir = os.path.join(part_dir, data_file)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        cur_part_valid = int(part_valids[cur_idx].sum().item())
        input_part_pcs = part_pcs[cur_idx].squeeze(0)[:cur_part_valid]

        # process.
        cur_part_poses = part_poses[cur_idx].squeeze(0)[:cur_part_valid]
        cur_part_trans = cur_part_poses[:, :3].unsqueeze(1).repeat(1, input_part_pcs.size(1), 1)
        cur_part_rot = cur_part_poses[:, 3:].unsqueeze(1).repeat(1, input_part_pcs.size(1), 1)
        gt_part_pcs = qrot(cur_part_rot, input_part_pcs) + cur_part_trans

        # save.
        input_part_pcs = input_part_pcs.numpy()
        for part_id in range(cur_part_valid):
            part_id_file = os.path.join(data_dir, "part-%02d.png" % part_id)
            print("process {}".format(part_id_file))
            aa = torch.zeros((cur_part_valid, 1000, 3))
            bb = torch.tensor(input_part_pcs[part_id])
            aa[part_id] = bb

            # cur_part_pcs = input_part_pcs[part_id][np.newaxis, :, :]
            render_utils.render_part_pts(part_id_file, aa.numpy(), blender_fn='object_centered.blend')

        gt_part_pcs = gt_part_pcs.numpy()
        render_utils.render_part_pts(os.path.join(gt_dir, data_name), gt_part_pcs, blender_fn='object_centered.blend')

    return batch_size


def gt_vis_wip(part_pcs, part_poses, part_valids, part_cates, num_ins, args):
    import numpy as np
    import random
    import torch
    vis_root = os.path.join(args.gt_vis_dir, args.category)
    gt_dir = os.path.join(vis_root, "gt_assembly")
    part_dir = os.path.join(vis_root, "wip_parts")

    if not os.path.exists(gt_dir):
        os.makedirs(gt_dir)
    if not os.path.exists(part_dir):
        os.makedirs(part_dir)

    batch_size = len(part_pcs)
    for cur_idx in range(batch_size):
        data_name = args.category + "-%04d.png" % (num_ins + cur_idx)
        print("Processing: {}".format(data_name))
        data_file = args.category + "-%04d" % (num_ins + cur_idx)
        data_dir = os.path.join(part_dir, data_file)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        cur_part_valid = int(part_valids[cur_idx].sum().item())
        input_part_pcs = part_pcs[cur_idx].squeeze(0)[:cur_part_valid]

        # process.
        cur_part_poses = part_poses[cur_idx].squeeze(0)[:cur_part_valid]
        cur_part_trans = cur_part_poses[:, :3].unsqueeze(1).repeat(1, input_part_pcs.size(1), 1)
        cur_part_rot = cur_part_poses[:, 3:].unsqueeze(1).repeat(1, input_part_pcs.size(1), 1)
        gt_part_pcs = qrot(cur_part_rot, input_part_pcs) + cur_part_trans

        cur_part_cates = part_cates[cur_idx].squeeze(0)[:cur_part_valid]
        max_id = int(cur_part_cates.max().item())
        candidates = []
        for id in range(1, max_id + 1):
            aa = (cur_part_cates == id).nonzero()
            if len(aa) == 1:
                candidates.append(aa[0][0].item())
            else:
                aa = aa.squeeze().numpy().tolist()
                random.shuffle(aa)
                candidates.append(aa[0])

        pairs = []
        for p_i in range(len(candidates)):
            i_candidates = candidates[p_i+1:]
            for p_j in i_candidates:
                pairs.append([candidates[p_i], p_j])
        pairs = torch.Tensor(pairs).long()

        # save.
        input_part_pcs = input_part_pcs.numpy()
        for pair_id, pair in enumerate(pairs):
            pair_id_dir = os.path.join(data_dir, "pair-%02d" % pair_id)
            print("process {}".format(pair_id_dir))
            if not os.path.exists(pair_id_dir):
                os.makedirs(pair_id_dir)
            assembly_file = os.path.join(pair_id_dir, "pair-%02d.png" % pair_id)

            mask = torch.ones(cur_part_valid).bool()
            mask[pair] = False
            cur_gt_part_pcs = gt_part_pcs[mask].numpy()
            render_utils.render_part_pts(assembly_file, cur_gt_part_pcs,
                                         blender_fn='object_centered.blend')

            for part_id in pair:
                cur_part_file = os.path.join(pair_id_dir, "part-%02d.png" % part_id)
                cur_part_pcs = input_part_pcs[part_id][np.newaxis, :, :]
                render_utils.render_part_pts(cur_part_file, cur_part_pcs, blender_fn='object_centered.blend')

    return batch_size
