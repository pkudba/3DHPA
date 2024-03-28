# coding: utf-8
import torch
import torch.nn.functional as F
from .func import linear_assignment, get_trans_l2_loss, get_rot_l2_loss, get_rot_cd_loss, get_shape_cd_loss, get_total_cd_loss, get_trans_cd_loss


def comp_losses(pred_poses, gt_poses, part_pcs, part_valids, match_ids, mse_weight, args=None):
    batch_size, num_trans, num_part, dim_pred = pred_poses.size()
    mse_weight = mse_weight.to(pred_poses.device)
    for trans_ind in range(num_trans):
        pred_poses_per_trans = pred_poses[:, trans_ind]
        for bs_ind in range(batch_size):
            cur_match_ids = match_ids[bs_ind]
            for ins_id in range(1, num_part + 1):
                need_to_match_part = list()
                for part_ind in range(num_part):
                    if cur_match_ids[part_ind] == ins_id:
                        need_to_match_part.append(part_ind)
                if not need_to_match_part:
                    break
                cur_pts = part_pcs[bs_ind, need_to_match_part]
                cur_pred_poses = pred_poses_per_trans[bs_ind, need_to_match_part]
                cur_pred_centers = cur_pred_poses[:, :3]
                cur_pred_quats = cur_pred_poses[:, 3:]
                cut_gt_poses = gt_poses[bs_ind, need_to_match_part]
                cur_gt_centers = cut_gt_poses[:, :3]
                cur_gt_quats = cut_gt_poses[:, 3:]

                # linear assignment.
                matched_pred_ids, matched_gt_ids = linear_assignment(cur_pts, cur_pred_centers, cur_pred_quats,
                                                                     cur_gt_centers, cur_gt_quats)
                pred_poses_per_trans[bs_ind, need_to_match_part] = cur_pred_poses[matched_pred_ids]
                gt_poses[bs_ind, need_to_match_part] = cut_gt_poses[matched_gt_ids]

        # compute losses.
        pred_trans = pred_poses_per_trans[:, :, :3]
        pred_rot = pred_poses_per_trans[:, :, 3:]
        gt_trans = gt_poses[:, :, :3]
        gt_rot = gt_poses[:, :, 3:]
        trans_l2_loss_per_trans = get_trans_l2_loss(pred_trans, gt_trans, part_valids, mse_weight)  # B
        trans_cd_loss_per_trans = get_trans_cd_loss(part_pcs, pred_trans, gt_trans, part_valids)  # B
        rot_l2_loss_per_trans = get_rot_l2_loss(part_pcs, pred_rot, gt_rot, part_valids)
        rot_cd_loss_per_trans = get_rot_cd_loss(part_pcs, pred_rot, gt_rot, part_valids)
        shape_cd_loss_per_trans = get_shape_cd_loss(part_pcs, pred_rot, gt_rot, pred_trans, gt_trans, part_valids)
        part_cd_loss_per_trans, _ = get_total_cd_loss(part_pcs, pred_rot, gt_rot, pred_trans, gt_trans, part_valids)

        # for each type of loss, compute avg loss per batch.
        trans_l2_loss = trans_l2_loss_per_trans.mean()
        rot_l2_loss = rot_l2_loss_per_trans.mean()
        trans_cd_loss = trans_cd_loss_per_trans.mean()
        rot_cd_loss = rot_cd_loss_per_trans.mean()
        shape_cd_loss = shape_cd_loss_per_trans.mean()
        part_cd_loss = part_cd_loss_per_trans.mean()


        # compute total loss.
        if trans_ind == 0:
            total_loss = trans_l2_loss * args.loss_weight_trans_l2 + \
                         rot_l2_loss * args.loss_weight_rot_l2 + \
                         rot_cd_loss * args.loss_weight_rot_cd + \
                         trans_cd_loss * args.loss_weight_trans_cd + \
                         shape_cd_loss * args.loss_weight_shape_cd + \
                         part_cd_loss * args.loss_weight_part_cd
            total_trans_l2_loss = trans_l2_loss
            total_rot_l2_loss = rot_l2_loss
            total_rot_cd_loss = rot_cd_loss
            total_trans_cd_loss = trans_cd_loss
            total_shape_cd_loss = shape_cd_loss
            total_part_cd_loss = part_cd_loss
        else:
            total_loss += trans_l2_loss * args.loss_weight_trans_l2 + \
                          rot_l2_loss * args.loss_weight_rot_l2 + \
                          rot_cd_loss * args.loss_weight_rot_cd + \
                          trans_cd_loss * args.loss_weight_trans_cd + \
                          shape_cd_loss * args.loss_weight_shape_cd + \
                          part_cd_loss * args.loss_weight_part_cd
            total_trans_l2_loss += trans_l2_loss
            total_rot_l2_loss += rot_l2_loss
            total_rot_cd_loss += rot_cd_loss
            total_trans_cd_loss += trans_cd_loss
            total_shape_cd_loss += shape_cd_loss
            total_part_cd_loss += part_cd_loss

    total_loss /= num_trans
    total_trans_l2_loss /= num_trans
    total_rot_l2_loss /= num_trans
    total_rot_cd_loss /= num_trans
    total_trans_cd_loss /= num_trans
    total_shape_cd_loss /= num_trans
    total_part_cd_loss /= num_trans
    return total_loss, total_trans_l2_loss, total_rot_l2_loss, total_trans_cd_loss, total_rot_cd_loss, total_shape_cd_loss, total_part_cd_loss


def comp_decoder_losses(pred_poses, gt_poses, pred_cates, gt_cates, part_pcs, pos_ids, args=None):
    batch_size, num_trans, num_part, dim_pred = pred_poses.size()
    _, num_pos = pos_ids.size()
    pred_poses = pred_poses[:, :, :num_pos, :]
    part_pcs = part_pcs[:, :num_pos].contiguous()
    part_valids = pred_poses.new_ones((batch_size, num_pos))

    for trans_ind in range(num_trans):
        pred_poses_per_trans = pred_poses[:, trans_ind]
        pred_cates_per_trans = pred_cates[:, trans_ind]

        # compute pose loss.
        pred_trans = pred_poses_per_trans[:, :, :3]
        pred_rot = pred_poses_per_trans[:, :, 3:]
        gt_trans = gt_poses[:, :, :3]
        gt_rot = gt_poses[:, :, 3:]
        trans_l2_loss_per_trans = get_trans_l2_loss(pred_trans, gt_trans, part_valids)  # B
        rot_l2_loss_per_trans = get_rot_l2_loss(part_pcs, pred_rot, gt_rot, part_valids)
        rot_cd_loss_per_trans = get_rot_cd_loss(part_pcs, pred_rot, gt_rot, part_valids)
        shape_cd_loss_per_trans = get_shape_cd_loss(part_pcs, pred_rot, gt_rot, pred_trans, gt_trans, part_valids)

        # for each type of loss, compute avg loss per batch.
        trans_l2_loss = trans_l2_loss_per_trans.mean()
        rot_l2_loss = rot_l2_loss_per_trans.mean()
        rot_cd_loss = rot_cd_loss_per_trans.mean()
        shape_cd_loss = shape_cd_loss_per_trans.mean()

        # compute cate loss.
        cate_loss = F.binary_cross_entropy_with_logits(pred_cates_per_trans, gt_cates, reduction="mean")

        # compute total loss.
        if trans_ind == 0:
            total_loss = trans_l2_loss * args.loss_weight_trans_l2 + \
                         rot_l2_loss * args.loss_weight_rot_l2 + \
                         rot_cd_loss * args.loss_weight_rot_cd + \
                         shape_cd_loss * args.loss_weight_shape_cd + \
                         cate_loss * args.loss_weight_cate
            total_trans_l2_loss = trans_l2_loss
            total_rot_l2_loss = rot_l2_loss
            total_rot_cd_loss = rot_cd_loss
            total_shape_cd_loss = shape_cd_loss
            total_cate_loss = cate_loss
        else:
            total_loss += trans_l2_loss * args.loss_weight_trans_l2 + \
                          rot_l2_loss * args.loss_weight_rot_l2 + \
                          rot_cd_loss * args.loss_weight_rot_cd + \
                          shape_cd_loss * args.loss_weight_shape_cd + \
                          cate_loss * args.loss_weight_cate
            total_trans_l2_loss += trans_l2_loss
            total_rot_l2_loss += rot_l2_loss
            total_rot_cd_loss += rot_cd_loss
            total_shape_cd_loss += shape_cd_loss
            total_cate_loss += cate_loss

    total_loss /= num_trans
    total_trans_l2_loss /= num_trans
    total_rot_l2_loss /= num_trans
    total_rot_cd_loss /= num_trans
    total_shape_cd_loss /= num_trans
    total_cate_loss /= num_trans

    return total_loss, total_trans_l2_loss, total_rot_l2_loss, total_rot_cd_loss, total_shape_cd_loss, total_cate_loss


def comp_decoder_losses_v2(pred_poses, gt_poses, part_pcs, pos_ids, args=None):
    batch_size, num_trans, num_part, dim_pred = pred_poses.size()
    _, num_pos = pos_ids.size()
    pred_poses = pred_poses[:, :, :num_pos, :]
    part_pcs = part_pcs[:, :num_pos].contiguous()
    part_valids = pred_poses.new_ones((batch_size, num_pos))

    for trans_ind in range(num_trans):
        pred_poses_per_trans = pred_poses[:, trans_ind]

        # compute pose loss.
        pred_trans = pred_poses_per_trans[:, :, :3]
        pred_rot = pred_poses_per_trans[:, :, 3:]
        gt_trans = gt_poses[:, :, :3]
        gt_rot = gt_poses[:, :, 3:]
        trans_l2_loss_per_trans = get_trans_l2_loss(pred_trans, gt_trans, part_valids)  # B
        rot_l2_loss_per_trans = get_rot_l2_loss(part_pcs, pred_rot, gt_rot, part_valids)
        rot_cd_loss_per_trans = get_rot_cd_loss(part_pcs, pred_rot, gt_rot, part_valids)
        shape_cd_loss_per_trans = get_shape_cd_loss(part_pcs, pred_rot, gt_rot, pred_trans, gt_trans, part_valids)

        # for each type of loss, compute avg loss per batch.
        trans_l2_loss = trans_l2_loss_per_trans.mean()
        rot_l2_loss = rot_l2_loss_per_trans.mean()
        rot_cd_loss = rot_cd_loss_per_trans.mean()
        shape_cd_loss = shape_cd_loss_per_trans.mean()

        # compute total loss.
        if trans_ind == 0:
            total_loss = trans_l2_loss * args.loss_weight_trans_l2 + \
                         rot_l2_loss * args.loss_weight_rot_l2 + \
                         rot_cd_loss * args.loss_weight_rot_cd + \
                         shape_cd_loss * args.loss_weight_shape_cd
            total_trans_l2_loss = trans_l2_loss
            total_rot_l2_loss = rot_l2_loss
            total_rot_cd_loss = rot_cd_loss
            total_shape_cd_loss = shape_cd_loss
        else:
            total_loss += trans_l2_loss * args.loss_weight_trans_l2 + \
                          rot_l2_loss * args.loss_weight_rot_l2 + \
                          rot_cd_loss * args.loss_weight_rot_cd + \
                          shape_cd_loss * args.loss_weight_shape_cd
            total_trans_l2_loss += trans_l2_loss
            total_rot_l2_loss += rot_l2_loss
            total_rot_cd_loss += rot_cd_loss
            total_shape_cd_loss += shape_cd_loss

    total_loss /= num_trans
    total_trans_l2_loss /= num_trans
    total_rot_l2_loss /= num_trans
    total_rot_cd_loss /= num_trans
    total_shape_cd_loss /= num_trans

    return total_loss, total_trans_l2_loss, total_rot_l2_loss, total_rot_cd_loss, total_shape_cd_loss


def comp_decoder_losses_freeze(pred_poses, gt_poses, part_pcs, part_valids, match_ids, part_mask, args=None):
    batch_size, num_trans, num_part, dim_pred = pred_poses.size()

    for trans_ind in range(num_trans):
        pred_poses_per_trans = pred_poses[:, trans_ind]

        # bilinear match for geometrically-equivalent parts.
        for bs_ind in range(batch_size):
            cur_match_ids = match_ids[bs_ind]
            for ins_id in range(1, num_part + 1):
                need_to_match_part = list()
                for part_ind in range(num_part):
                    if cur_match_ids[part_ind] == ins_id:
                        need_to_match_part.append(part_ind)
                if not need_to_match_part:
                    break
                cur_pts = part_pcs[bs_ind, need_to_match_part]

                # pred & gt poses.
                cur_pred_poses = pred_poses_per_trans[bs_ind, need_to_match_part]
                cur_pred_centers = cur_pred_poses[:, :3]
                cur_pred_quats = cur_pred_poses[:, 3:]
                cut_gt_poses = gt_poses[bs_ind, need_to_match_part]
                cur_gt_centers = cut_gt_poses[:, :3]
                cur_gt_quats = cut_gt_poses[:, 3:]

                # linear assignment.
                matched_pred_ids, matched_gt_ids = linear_assignment(cur_pts, cur_pred_centers, cur_pred_quats,
                                                                     cur_gt_centers, cur_gt_quats)
                pred_poses_per_trans[bs_ind, need_to_match_part] = cur_pred_poses[matched_pred_ids]
                gt_poses[bs_ind, need_to_match_part] = cut_gt_poses[matched_gt_ids]

        # compute losses.
        # note here we assume others are correct.
        part_mask_3 = part_mask.transpose(0, 1).repeat(1, 1, dim_pred)
        pred_poses_per_trans[part_mask_3] = gt_poses[part_mask_3]
        pred_trans = pred_poses_per_trans[:, :, :3]
        pred_rot = pred_poses_per_trans[:, :, 3:]
        gt_trans = gt_poses[:, :, :3]
        gt_rot = gt_poses[:, :, 3:]

        part_mask_2 = (~part_mask).transpose(0, 1).squeeze(-1)
        # translation l2 loss.
        trans_l2_loss_per_trans = get_trans_l2_loss(pred_trans, gt_trans, part_valids, return_raw=True)[part_mask_2]

        # rotation l2 loss.
        rot_l2_loss_per_trans = get_rot_l2_loss(part_pcs, pred_rot, gt_rot, part_valids, return_raw=True)[part_mask_2]

        mask_valid = part_mask_2.unsqueeze(-1).repeat(1, 1, part_pcs.size(2))
        # rotation cd loss.
        rot_dist_1, rot_dist_2 = get_rot_cd_loss(part_pcs, pred_rot, gt_rot, part_valids, return_raw=True)
        rot_valid = mask_valid.view(-1, part_pcs.size(2)).contiguous()
        rot_dist_1 = rot_dist_1[rot_valid].view(batch_size, part_pcs.size(2)).contiguous()
        rot_dist_2 = rot_dist_2[rot_valid].view(batch_size, part_pcs.size(2)).contiguous()
        rot_cd_loss_per_trans = rot_dist_1.mean(1) + rot_dist_2.mean(1)

        # overall shape cd loss.
        shape_dist_1, shape_dist_2 = get_shape_cd_loss(part_pcs, pred_rot, gt_rot,
                                                       pred_trans, gt_trans, part_valids, return_raw=True)
        shape_valid = mask_valid.view(batch_size, -1).contiguous()
        shape_dist_1 = shape_dist_1[shape_valid].view(batch_size, part_pcs.size(2)).contiguous()
        shape_dist_2 = shape_dist_2[shape_valid].view(batch_size, part_pcs.size(2)).contiguous()
        shape_cd_loss_per_trans = shape_dist_1.mean(1) + shape_dist_2.mean(1)

        # for each type of loss, compute avg loss per batch.
        trans_l2_loss = trans_l2_loss_per_trans.mean()
        rot_l2_loss = rot_l2_loss_per_trans.mean()
        rot_cd_loss = rot_cd_loss_per_trans.mean()
        shape_cd_loss = shape_cd_loss_per_trans.mean()

        # compute total loss.
        if trans_ind == 0:
            total_loss = trans_l2_loss * args.loss_weight_trans_l2 + \
                         rot_l2_loss * args.loss_weight_rot_l2 + \
                         rot_cd_loss * args.loss_weight_rot_cd + \
                         shape_cd_loss * args.loss_weight_shape_cd
            total_trans_l2_loss = trans_l2_loss
            total_rot_l2_loss = rot_l2_loss
            total_rot_cd_loss = rot_cd_loss
            total_shape_cd_loss = shape_cd_loss
        else:
            total_loss += trans_l2_loss * args.loss_weight_trans_l2 + \
                          rot_l2_loss * args.loss_weight_rot_l2 + \
                          rot_cd_loss * args.loss_weight_rot_cd + \
                          shape_cd_loss * args.loss_weight_shape_cd
            total_trans_l2_loss += trans_l2_loss
            total_rot_l2_loss += rot_l2_loss
            total_rot_cd_loss += rot_cd_loss
            total_shape_cd_loss += shape_cd_loss

    total_loss /= num_trans
    total_trans_l2_loss /= num_trans
    total_rot_l2_loss /= num_trans
    total_rot_cd_loss /= num_trans
    total_shape_cd_loss /= num_trans
    return total_loss, total_trans_l2_loss, total_rot_l2_loss, total_rot_cd_loss, total_shape_cd_loss

