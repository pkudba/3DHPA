# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from .pointnet.pointnet_cls import get_model as pointnet_cls
from .pointnet.pointnet_cls import trans_loss
from .build_model import build_ende, build_hpa
from .models import MLP, Predictor
from .losses import comp_losses
from .inference import inference, shape_diversity_score

from .func import (get_shape_transformed, linear_assignment,
                   get_trans_l2_loss, get_rot_l2_loss,
                   get_rot_cd_loss, get_shape_cd_loss_default,
                   get_total_cd_loss, get_contact_point_loss, batch_get_contact_point_loss,
                   get_contact_point_loss_for_single_part)

class hierarchical_trans(nn.Module):
    def __init__(self, args):
        super(hierarchical_trans, self).__init__()
        self.args = args

        if args.backbone == "pointnet_cls":
            self.feat_extract = pointnet_cls(k=args.feat_dim, normal_channel=False)
            self.feat_extract2 = pointnet_cls(k=args.feat_dim, normal_channel=False)
        else:
            raise NotImplementedError
        self.trans_criterion = trans_loss()
        self.encoder = build_hpa(args, mode='base')
        self.encoder2 = build_hpa(args, mode='relative')

        if args.decode_on:
            self.encoder2 = build_ende(args, mode='relative')
        self.mse_weight = torch.tensor([1, 1, 1])

    def branch1(self, part_pcs, part_valid, gt_part_poses, match_ids, part_ids,
                contact_points, sym_info=None):
        batch_size, num_part, _, _ = part_pcs.size()
        base_feat, trans_feat = self.feat_extract(part_pcs.view(batch_size * num_part, -1, 3).permute(0, 2, 1))
        base_feat = base_feat.view(batch_size, num_part, -1)
        trans_loss = self.trans_criterion(trans_feat)

        if self.args.bi_pn==1:
            base_feat2, trans_feat2 = self.feat_extract2(part_pcs.view(batch_size * num_part, -1, 3).permute(0, 2, 1))
            base_feat2 = base_feat2.view(batch_size, num_part, -1)
            trans_loss += self.trans_criterion(trans_feat2)

        output = dict()
        kwargs = {"part_ids": part_ids}

        base_contact_points = contact_points

        for mon_idx in range(self.args.train_mon):
            if self.args.bi_pn==1:
                preds, memory = self.encoder(base_feat, base_feat2, part_valid, None, None, **kwargs)
            else:
                preds, memory = self.encoder(base_contact_points, base_feat, part_valid, gt_part_poses, None, None, **kwargs)
        return preds, memory
            

    def forward(self, raw_part_pcs, part_valid, gt_part_poses, match_ids, part_ids,
                contact_points, sym_info=None):
        preds1, memory1 = self.branch1(raw_part_pcs, part_valid, gt_part_poses, match_ids, part_ids, contact_points, sym_info=None)
        preds1_rot = preds1[:, -1, :, 3:]
        preds1_trans = preds1[:, -1, :, :3]
        base_preds1 = preds1[:, -1, :, :]

        part_pcs = get_shape_transformed(raw_part_pcs, preds1_rot, preds1_trans)

        batch_size, num_part, _, _ = part_pcs.size()
        base_feat, trans_feat = self.feat_extract2(part_pcs.view(batch_size * num_part, -1, 3).permute(0, 2, 1))
        base_feat = base_feat.view(batch_size, num_part, -1)
        trans_loss = self.trans_criterion(trans_feat)

        if self.args.bi_pn==1:
            base_feat2, trans_feat2 = self.feat_extract2(part_pcs.view(batch_size * num_part, -1, 3).permute(0, 2, 1))
            base_feat2 = base_feat2.view(batch_size, num_part, -1)
            trans_loss += self.trans_criterion(trans_feat2)

        if self.training:
            loss_per_mon1, trans_l2_loss1, rot_l2_loss1, trans_cd_loss1, rot_cd_loss1, shape_cd_loss1, part_cd_loss1 = \
                    comp_losses(preds1, gt_part_poses, raw_part_pcs, part_valid, match_ids, self.mse_weight, args=self.args)

            # mask filter in encoder during training.
    
            if self.args.filter_on and part_valid.dim()==2:
                flag_valid = (part_valid.sum(1) == 1).sum().bool()
                if flag_valid:
                    pass
                else:
                    prob = torch.rand(1).item()
                    if prob < self.args.filter_thresh:
                        memory1, base_feat, part_pcs, raw_part_pcs, part_valid, gt_part_poses, base_preds1, part_ids, match_ids, contact_points, _, part_mask = \
                            self.prepare_filters_v1(memory1, base_feat, part_pcs, raw_part_pcs, part_valid, gt_part_poses, base_preds1, part_ids, match_ids, contact_points=contact_points)

            output = dict()
            kwargs = {"part_ids": part_ids}
            base = base_preds1.unsqueeze(1).repeat(1, 6, 1, 1)
            for mon_idx in range(self.args.train_mon):
                # encoder func: [bs, num_trans, num_part, 3 + 4], [num_part, bs, F]
                if self.args.bi_pn==1:
                    preds, _ = self.encoder2(base_feat, base_feat2, part_valid, None, None, **kwargs)
                else:
                    preds, memory2 = self.encoder2(memory1, base_feat, part_valid, gt_part_poses, base_preds1.permute(1,0,2), None, **kwargs)
                    final_preds = preds
                loss_per_mon, trans_l2_loss, rot_l2_loss, trans_cd_loss, rot_cd_loss, shape_cd_loss, part_cd_loss = \
                    comp_losses(final_preds, gt_part_poses, raw_part_pcs, part_valid, match_ids, self.mse_weight, args=self.args)
                if mon_idx == 0:
                    loss = loss_per_mon.clone()
                else:
                    loss = torch.min(loss, loss_per_mon)
            loss += trans_loss
            return final_preds, loss, trans_l2_loss, rot_l2_loss, trans_cd_loss, rot_cd_loss, shape_cd_loss, part_cd_loss, trans_loss, output

        else:
            output = dict()
            base = base_preds1.unsqueeze(1).repeat(1, 6, 1, 1)
            if self.args.type_eval == "encoder":
                if self.args.bi_pn==1:
                    trans_l2_loss_per_trans, rot_cd_loss_per_trans, part_cd_loss, shape_cd_loss, contact_point_loss, acc, valid, \
                        num_contact_correct, num_contact_point, batch_size, pred_poses, cdsV1_sum, cdsV2_sum = \
                        self.inference_encoder(base_feat, base_feat2, raw_part_pcs, part_valid, gt_part_poses, base_preds1.permute(1,0,2),
                                           part_ids, match_ids, memory1, sym_info)
                else:
                    trans_l2_loss_per_trans, rot_cd_loss_per_trans, part_cd_loss, shape_cd_loss, contact_point_loss, acc, valid, \
                        num_contact_correct, num_contact_point, batch_size, pred_poses, cdsV1_sum, cdsV2_sum = \
                        self.inference_encoder(memory1, base_feat, base_feat, raw_part_pcs, part_valid, gt_part_poses, base_preds1.permute(1,0,2), base,
                                           part_ids, match_ids, contact_points, sym_info)
                output["pred_poses"] = pred_poses
            elif self.args.type_eval == "wip":
                part_cd_loss, shape_cd_loss, contact_point_loss, acc, valid, \
                    num_contact_correct, num_contact_point, batch_size = \
                    self.inference_wip(base_feat, raw_part_pcs, part_valid, gt_part_poses,
                                       part_ids, match_ids, contact_points, sym_info)
            else:
                raise NotImplementedError
            


            return trans_l2_loss_per_trans, rot_cd_loss_per_trans, part_cd_loss, shape_cd_loss, contact_point_loss, acc, valid, num_contact_correct, num_contact_point, batch_size, output, cdsV1_sum, cdsV2_sum

    def prepare_decoder_v1(self, part_feat, part_pcs, part_valid):
        batch_size, num_part, num_channel = part_feat.size()
        _, _, num_point, _ = part_pcs.size()

        if self.args.rand_pos:
            num_pos = torch.randperm(self.args.num_pos)[0].item() + 1
        else:
            num_pos = self.args.num_pos

        # generate ids.
        rand_ids = [torch.randperm(int(bs)) for bs in part_valid.sum(1)]
        pos_ids = torch.cat([ids[:num_pos] for ids in rand_ids]).to(part_valid.device)
        pos_bs = torch.tensor([_ for _ in range(batch_size)]).unsqueeze(1).repeat(1, num_pos).view(-1).to(part_valid.device)
        neg_ids = torch.cat([ids[num_pos:] for ids in rand_ids]).to(part_valid.device)
        neg_bs_ = [[bs_id] * int(bs_num - num_pos) for bs_id, bs_num in enumerate(part_valid.sum(1))]
        neg_bs = []
        for bs_id in range(batch_size):
            neg_bs += neg_bs_[bs_id]
        neg_bs = torch.tensor(neg_bs).to(part_valid.device)

        # prepare decode feats & pcs.
        decode_feat = part_feat.new_zeros((batch_size, self.args.num_queries, num_channel))
        decode_pcs = part_pcs.new_zeros((batch_size, self.args.num_queries, num_point, 3))
        # add pos feats & pcs.
        pos_feat = part_feat[pos_bs, pos_ids].view(batch_size, num_pos, -1)
        decode_feat[:, :num_pos, :] = pos_feat
        pos_pcs = part_pcs[pos_bs, pos_ids].view(batch_size, num_pos, num_point, -1)
        decode_pcs[:, :num_pos, :, :] = pos_pcs
        # add neg feats & pcs.
        neg_feat_gallery = part_feat[neg_bs, neg_ids]
        neg_pcs_gallery = part_pcs[neg_bs, neg_ids]
        num_neg = self.args.num_queries - num_pos
        for bs_id in range(batch_size):
            neg_mask = neg_bs != bs_id
            neg_feat = neg_feat_gallery[neg_mask]
            neg_pcs = neg_pcs_gallery[neg_mask]
            total_neg = neg_feat.size(0)
            if total_neg >= num_neg:
                select_ids = torch.randperm(total_neg)[:num_neg].to(part_valid.device)
                decode_feat[bs_id, num_pos:, :] = neg_feat[select_ids, :]
                decode_pcs[bs_id, num_pos:, :, :] = neg_pcs[select_ids, :, :]
            else:
                decode_feat[bs_id, -total_neg:, :] = neg_feat.clone()
                decode_pcs[bs_id, -total_neg:, :, :] = neg_pcs.clone()

        # prepare decode masks.
        decode_mask = part_feat.new_ones((batch_size, num_part, 1))
        decode_mask[pos_bs, pos_ids] = 0.
        decode_mask = decode_mask.transpose(0, 1).bool()

        return decode_feat, decode_pcs, decode_mask, pos_ids.view(batch_size, num_pos)

    def prepare_labels_v1(self, memory, gt_poses, decode_mask, pos_ids):
        """
        """
        num_part, batch_size, len_feat = memory.size()
        _, num_pos = pos_ids.size()

        # prepare decode memory --> [num_part_, bs, F].
        decode_memory = memory[decode_mask.repeat(1, 1, len_feat)].view(num_part - num_pos, batch_size, len_feat)

        # prepare decode gt pose --> [bs, num_pos, 7].
        pos_ids = pos_ids.view(-1)
        pos_bs = torch.arange(batch_size).unsqueeze(1).repeat(1, num_pos).view(-1).to(gt_poses.device)
        decode_poses = gt_poses[pos_bs, pos_ids, :].view(batch_size, num_pos, 7)

        # prepare decode cate label --> [bs, num_queries, 1]
        cate_labels = gt_poses.new_zeros((batch_size, self.args.num_queries, 1))
        cate_labels[:, :num_pos, :] = 1.

        return decode_memory, decode_poses, cate_labels

    def prepare_filters_v1(self, memory, part_feat, part_pcs, raw_part_pcs, part_valid, part_pose, base_preds1, part_ids, match_ids,
                           contact_points=None, sym_info=None, filter_id=0):
        """
        Now only support num_filter == n.
        """
        batch_size, num_part, len_feat = part_feat.size()
        _, _, num_point, _ = part_pcs.size()
        num_filter = self.args.num_filter
        num_res = num_part - num_filter

        part_mask = part_feat.new_ones((batch_size, num_part))
        if self.training:
            rand_ids = [torch.randperm(int(bs)) for bs in part_valid.sum(1)]
            filter_ids = torch.cat([ids[:num_filter] for ids in rand_ids]).to(part_valid.device)
        else:
            filter_ids = torch.tensor([filter_id for _ in range(batch_size)]).to(part_valid.device)
        filter_bs = torch.tensor([_ for _ in range(batch_size)]).unsqueeze(1).repeat(1, num_filter).view(-1).to(part_valid.device)
        part_mask[filter_bs, filter_ids] = 0.
        part_mask = part_mask.bool()
        part_mask_2 = part_mask.clone()
        part_mask_3 = part_mask.unsqueeze(-1)
        part_mask_4 = part_mask.unsqueeze(-1).unsqueeze(-1)

        memory_filter = memory.clone().permute(1,0,2)[part_mask_3.repeat(1, 1, len_feat)].view(num_res, batch_size, len_feat).contiguous()
        new_part_feat = part_feat.clone()[part_mask_3.repeat(1, 1, len_feat)].view(batch_size, num_res, len_feat).contiguous()
        new_part_pcs = part_pcs.clone()[part_mask_4.repeat(1, 1, num_point, 3)].view(batch_size, num_res, num_point, 3).contiguous()
        new_raw_part_pcs = raw_part_pcs.clone()[part_mask_4.repeat(1, 1, num_point, 3)].view(batch_size, num_res, num_point, 3).contiguous()
        new_part_valid = part_valid.clone()[part_mask_2].view(batch_size, num_res).contiguous()
        new_part_pose = part_pose.clone()[part_mask_3.repeat(1, 1, 7)].view(batch_size, num_res, 7).contiguous()
        new_base_preds1 = base_preds1.clone()[part_mask_3.repeat(1, 1, 7)].view(batch_size, num_res, 7).contiguous()
        new_part_ids = part_ids.clone()[part_mask_2].view(batch_size, num_res).contiguous()
        if sym_info is not None:
            new_sym_info = sym_info.clone()[part_mask_3.repeat(1, 1, 3)].view(batch_size, num_res, 3).contiguous()
        else:
            new_sym_info = sym_info

        if contact_points is not None:
            contact_points2 = contact_points.clone()[part_mask_4.repeat(1, 1, num_part, 4)].view(batch_size, num_res, num_part, 4).contiguous()
            part_mask_4_ = part_mask.unsqueeze(1).unsqueeze(-1).repeat(1, num_res, 1, 4)
            new_contact_points = contact_points2.clone()[part_mask_4_].view(batch_size, num_res, num_res, 4).contiguous()

        # filter match ids.
        filter_match_ids = list()
        part_mask = part_mask.cpu().numpy()
        for bs, match_id in enumerate(match_ids):
            filter_match_ids.append(match_id[part_mask[bs]])

        return memory_filter, new_part_feat, new_part_pcs, new_raw_part_pcs, new_part_valid, new_part_pose, new_base_preds1, new_part_ids, filter_match_ids, new_contact_points, new_sym_info, part_mask_2

    def inference_encoder(self, cross, base_feat, base_feat2, part_pcs, part_valid, part_poses, base_pred, base, part_ids, match_ids, contact_points, sym_info):
        batch_size, num_part, len_feat = base_feat.size()

        pred_poses = list()
        measures = list()
        kwargs = {"part_ids": part_ids}
        array_sds_cd_per_data = []

        for mon_id in range(self.args.eval_mon):
            # forward func.
            if self.args.bi_pn==1:
                preds, _ = self.encoder2(base_feat, base_feat2, part_valid, base_pred, None, **kwargs)
            else:
                preds, _ = self.encoder2(cross, base_feat, part_valid, part_poses, base_pred, None, **kwargs)
            final_preds = preds

            # evaluation.
            acc_per_mon, trans_l2_loss_per_trans, rot_cd_loss_per_trans, pred_poses_per_mon, part_cd_loss_per_mon, shape_cd_loss_per_mon, contact_point_loss_per_mon, \
                contact_correct_per_mon, num_contact_point = inference(final_preds, part_poses, part_pcs, part_valid, match_ids, contact_points, sym_info, args=self.args)

            contact_point_loss_per_data, count, total_num, batch_count, batch_total_num = batch_get_contact_point_loss(final_preds[:, -1, :, :3], final_preds[:, -1, :, 3:], contact_points, sym_info)
            batch_single_ca = batch_count.float() / batch_total_num.float()
            mask_nan = torch.isnan(batch_single_ca)
            batch_single_ca[mask_nan] = 0.0
            array_sds_cd_per_data.append([
                part_pcs.clone(),
                final_preds[:, -1, :, :].clone(),
                part_valid.clone(),
                shape_cd_loss_per_mon.clone(),
                batch_single_ca.to(final_preds.device)
            ])

            cdsV1, cdsV2 = shape_diversity_score(array_sds_cd_per_data, batch_size)

            cdsV1_sum = cdsV1.sum()
            cdsV2_sum = cdsV2.sum()


            # MoN loss func.
            if mon_id == 0:
                part_cd_loss = part_cd_loss_per_mon.clone()
                shape_cd_loss = shape_cd_loss_per_mon.clone()

                trans_l2_loss = trans_l2_loss_per_trans.clone()
                rot_cd_loss = rot_cd_loss_per_trans.clone()
                
                contact_point_loss = contact_point_loss_per_mon.clone()
                acc = acc_per_mon.clone()
                num_contact_correct = contact_correct_per_mon.clone()
            elif self.args.worst_mon:
                part_cd_loss = part_cd_loss.max(part_cd_loss_per_mon)
                shape_cd_loss = shape_cd_loss.max(shape_cd_loss_per_mon)

                trans_l2_loss = trans_l2_loss.max(trans_l2_loss_per_trans)
                rot_cd_loss = rot_cd_loss.max(rot_cd_loss_per_trans)

                contact_point_loss = contact_point_loss.max(contact_point_loss_per_mon)
                acc = acc.min(acc_per_mon)
                num_contact_correct = num_contact_correct.min(contact_correct_per_mon)
            else:
                # different ins (best) could from different model.
                part_cd_loss = part_cd_loss.min(part_cd_loss_per_mon)
                shape_cd_loss = shape_cd_loss.min(shape_cd_loss_per_mon)
                
                trans_l2_loss = trans_l2_loss.min(trans_l2_loss_per_trans)
                rot_cd_loss = rot_cd_loss.min(rot_cd_loss_per_trans)

                contact_point_loss = contact_point_loss.min(contact_point_loss_per_mon)
                acc = acc.max(acc_per_mon)
                num_contact_correct = num_contact_correct.max(contact_correct_per_mon)

            pred_poses.append(pred_poses_per_mon.unsqueeze(1))
            measures.append(acc_per_mon.unsqueeze(1))

        part_cd_loss = part_cd_loss.mean()
        shape_cd_loss = shape_cd_loss.mean()
        trans_l2_loss = trans_l2_loss.mean()
        rot_cd_loss = rot_cd_loss.mean()
        contact_point_loss = contact_point_loss.mean()
        acc = acc.sum()  # how many parts are right in total in current batch.
        valid = part_valid.sum()  # how many parts are valid in current batch.
        num_contact_correct = num_contact_correct.sum()
        num_contact_point = num_contact_point.sum()

        # pred sort.
        pred_poses = torch.cat(pred_poses, dim=1)
        measures = torch.cat(measures, dim=1)
        if self.args.pred_encoder_vis:
            _, sort_indices = measures.sort(dim=1, descending=True)
            for bs_id in range(batch_size):
                pred_poses[bs_id] = pred_poses[bs_id, sort_indices[bs_id]]

        return trans_l2_loss, rot_cd_loss, part_cd_loss, shape_cd_loss, contact_point_loss, acc, valid, \
               num_contact_correct, num_contact_point, batch_size, pred_poses, cdsV1_sum, cdsV2_sum

    def inference_wip(self, base_feat, part_pcs, part_valid, part_poses, part_ids, match_ids, contact_points, sym_info):
        batch_size, num_part, len_feat = base_feat.size()

        # statistics.
        part_cd_loss_wip = base_feat.new_zeros(batch_size)
        shape_cd_loss_wip = base_feat.new_zeros(batch_size)
        contact_point_loss_wip = base_feat.new_zeros(batch_size)
        acc_wip = base_feat.new_zeros(batch_size)
        valid_wip = base_feat.new_zeros(batch_size)
        num_contact_correct_wip = base_feat.new_zeros(batch_size)
        num_contact_point_wip = base_feat.new_zeros(batch_size)
        ins_valid_wip = base_feat.new_zeros(batch_size)

        # process.
        num_part_per_ins = part_valid.sum(1)
        num_part_max = num_part_per_ins.max().long().item()
        for filter_id in range(num_part_max):
            cur_base_feat, cur_part_pcs, cur_part_valid, cur_part_poses, \
            cur_part_ids, cur_match_ids, cur_contact_points, cur_sym_info, part_mask = \
                self.prepare_filters_v1(base_feat, part_pcs, part_valid, part_poses, part_ids, match_ids,
                                        contact_points, sym_info, filter_id)
            kwargs = {"part_ids": cur_part_ids}
            for mon_id in range(self.args.eval_mon):
                # encoder func: [bs, num_trans, num_part, 3 + 4], [num_part, bs, F]
                preds, _ = self.encoder(cur_base_feat, cur_part_valid, None, None, **kwargs)

                # evaluation.
                _, part_cd_loss_per_mon, shape_cd_loss_per_mon, contact_point_loss_per_mon, \
                acc_per_mon, contact_correct_per_mon, num_contact_point = \
                    inference(preds, cur_part_poses, cur_part_pcs, cur_part_valid, cur_match_ids,
                              cur_contact_points, cur_sym_info, args=self.args)

                # MoN loss func.
                if mon_id == 0:
                    part_cd_loss = part_cd_loss_per_mon.clone()
                    shape_cd_loss = shape_cd_loss_per_mon.clone()
                    contact_point_loss = contact_point_loss_per_mon.clone()
                    acc = acc_per_mon.clone()
                    num_contact_correct = contact_correct_per_mon.clone()
                else:
                    # different ins (best) could from different model.
                    part_cd_loss = part_cd_loss.min(part_cd_loss_per_mon)
                    shape_cd_loss = shape_cd_loss.min(shape_cd_loss_per_mon)
                    contact_point_loss = contact_point_loss.min(contact_point_loss_per_mon)
                    acc = acc.max(acc_per_mon)
                    num_contact_correct = num_contact_correct.max(contact_correct_per_mon)

            ins_valid = (num_part_per_ins > filter_id).float()
            part_cd_loss_wip += part_cd_loss * ins_valid
            shape_cd_loss_wip += shape_cd_loss * ins_valid
            contact_point_loss_wip += contact_point_loss * ins_valid
            acc_wip += acc * ins_valid
            valid_wip += cur_part_valid.sum(-1) * ins_valid
            num_contact_correct_wip += num_contact_correct * ins_valid
            num_contact_point_wip += num_contact_point * ins_valid
            ins_valid_wip += ins_valid

        part_cd_loss = (part_cd_loss_wip / ins_valid_wip).mean()
        shape_cd_loss = (shape_cd_loss_wip / ins_valid_wip).mean()
        contact_point_loss = (contact_point_loss_wip / ins_valid_wip).mean()
        acc = (acc_wip / ins_valid_wip).sum()  # how many parts are right in total in current batch.
        valid = (valid_wip / ins_valid_wip).sum()  # how many parts are valid in current batch.
        num_contact_correct = (num_contact_correct_wip / ins_valid_wip).sum()
        num_contact_point = (num_contact_point_wip / ins_valid_wip).sum()
        return part_cd_loss, shape_cd_loss, contact_point_loss, acc, valid, \
               num_contact_correct, num_contact_point, batch_size
