# coding: utf-8
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../utils'))
import torch
from cd.chamfer import chamfer_distance
from quaternion import qrot
from scipy.optimize import linear_sum_assignment


def linear_assignment(pts, centers1, quats1, centers2, quats2):
    import random
    pts_to_select = torch.tensor(random.sample([i for i in range(pts.size(1))], 100))
    pts = pts[:, pts_to_select]
    cur_part_cnt, num_point, _ = pts.size()

    with torch.no_grad():
        cur_quats1 = quats1.unsqueeze(1).repeat(1, num_point, 1)
        cur_centers1 = centers1.unsqueeze(1).repeat(1, num_point, 1)
        cur_pts1 = qrot(cur_quats1, pts) + cur_centers1

        cur_quats2 = quats2.unsqueeze(1).repeat(1, num_point, 1)
        cur_centers2 = centers2.unsqueeze(1).repeat(1, num_point, 1)
        cur_pts2 = qrot(cur_quats2, pts) + cur_centers2

        cur_pts1 = cur_pts1.unsqueeze(1).repeat(1, cur_part_cnt, 1, 1).view(-1, num_point, 3)
        cur_pts2 = cur_pts2.unsqueeze(0).repeat(cur_part_cnt, 1, 1, 1).view(-1, num_point, 3)
        dist1, dist2 = chamfer_distance(cur_pts1, cur_pts2, transpose=False)
        dist_mat = (dist1.mean(1) + dist2.mean(1)).view(cur_part_cnt, cur_part_cnt)
        rind, cind = linear_sum_assignment(dist_mat.cpu().numpy())

    return rind, cind


def smooth_l1_loss(input, target, beta=1. / 12, reduction = 'none'):
    n = torch.abs(input - target)
    cond = n < beta
    ret = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret


def get_trans_l2_loss(trans1, trans2, valids, mse_weight, return_raw=False):
    loss_per_data = smooth_l1_loss(trans1, trans2)
    loss_per_data = (loss_per_data*mse_weight).sum(dim=-1)

    if return_raw:
        pass
    else:
        loss_per_data = (loss_per_data * valids).sum(1) / valids.sum(1)

    return loss_per_data


def get_rot_l2_loss(pts, quat1, quat2, valids, return_raw=False):
    num_point = pts.shape[2]

    pts1 = qrot(quat1.unsqueeze(2).repeat(1, 1, num_point, 1), pts)
    pts2 = qrot(quat2.unsqueeze(2).repeat(1, 1, num_point, 1), pts)

    loss_per_data = (pts1 - pts2).pow(2).sum(-1).mean(-1)

    if return_raw:
        pass
    else:
        loss_per_data = (loss_per_data * valids).sum(1) / valids.sum(1)

    return loss_per_data

def get_trans_cd_loss(pts, center1, center2, valids, return_raw=False):
    batch_size, _, num_point, _ = pts.size()

    center1 = center1.unsqueeze(2).repeat(1, 1, num_point, 1)
    center2 = center2.unsqueeze(2).repeat(1, 1, num_point, 1)

    pts1 = pts + center1
    pts2 = pts + center2

    dist1, dist2 = chamfer_distance(pts1.view(-1, num_point, 3), pts2.view(-1, num_point, 3), transpose=False)
    loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
    loss_per_data = loss_per_data.view(batch_size, -1)
    loss_per_data = (loss_per_data * valids).sum(1) / valids.sum(1)

    if return_raw:
        return dist1, dist2
    else:
        return loss_per_data


def get_rot_cd_loss(pts, quat1, quat2, valids, return_raw=False):
    batch_size, _, num_point, _ = pts.size()

    pts1 = qrot(quat1.unsqueeze(2).repeat(1, 1, num_point, 1), pts)
    pts2 = qrot(quat2.unsqueeze(2).repeat(1, 1, num_point, 1), pts)

    dist1, dist2 = chamfer_distance(pts1.view(-1, num_point, 3), pts2.view(-1, num_point, 3), transpose=False)
    loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
    loss_per_data = loss_per_data.view(batch_size, -1)
    loss_per_data = (loss_per_data * valids).sum(1) / valids.sum(1)

    if return_raw:
        return dist1, dist2
    else:
        return loss_per_data
    
def batch_get_contact_point_loss(center, quat, contact_points, sym_info):
    batch_size = center.shape[0]
    num_part = center.shape[1]
    contact_point_loss = torch.zeros(batch_size)
    total_num = 0
    batch_total_num = torch.zeros(batch_size, dtype=torch.long)
    count = 0
    batch_count = torch.zeros(batch_size, dtype=torch.long)
    for b in range(batch_size):
        sum_loss = 0
        for i in range(num_part):
            for j in range(num_part):
                if contact_points[b, i, j, 0]:
                    contact_point_1 = contact_points[b, i, j, 1:]
                    contact_point_2 = contact_points[b, j, i, 1:]
                    sym1 = sym_info[b, i]
                    sym2 = sym_info[b, j]
                    point_list_1 = get_possible_point_list2(contact_point_1, sym1)
                    point_list_2 = get_possible_point_list2(contact_point_2, sym2)
                    dist = get_min_l2_dist2(point_list_1, point_list_2, center[b, i, :], center[b, j, :], quat[b, i, :], quat[b, j, :])  # 1
                    if dist < 0.01:
                        count += 1
                        batch_count[b] += 1
                    total_num += 1
                    batch_total_num[b] += 1
                    sum_loss += dist
        contact_point_loss[b] = sum_loss
    return contact_point_loss, count, total_num, batch_count, batch_total_num

def get_sym_point2(point, x, y, z):
    if x:
        point[0] = - point[0]
    if y:
        point[1] = - point[1]
    if z:
        point[2] = - point[2]

    return point.tolist()

def get_possible_point_list2(point, sym):
    sym = torch.tensor([1.0,1.0,1.0]) 
    point_list = []
    if sym.equal(torch.tensor([0.0, 0.0, 0.0])):
        point_list.append(get_sym_point2(point, 0, 0, 0))
    elif sym.equal(torch.tensor([1.0, 0.0, 0.0])):
        point_list.append(get_sym_point2(point, 0, 0, 0))
        point_list.append(get_sym_point2(point, 1, 0, 0))
    elif sym.equal(torch.tensor([0.0, 1.0, 0.0])):
        point_list.append(get_sym_point2(point, 0, 0, 0))
        point_list.append(get_sym_point2(point, 0, 1, 0))
    elif sym.equal(torch.tensor([0.0, 0.0, 1.0])):
        point_list.append(get_sym_point2(point, 0, 0, 0))
        point_list.append(get_sym_point2(point, 0, 0, 1))
    elif sym.equal(torch.tensor([1.0, 1.0, 0.0])):
        point_list.append(get_sym_point2(point, 0, 0, 0))
        point_list.append(get_sym_point2(point, 1, 0, 0))
        point_list.append(get_sym_point2(point, 0, 1, 0))
        point_list.append(get_sym_point2(point, 1, 1, 0))
    elif sym.equal(torch.tensor([1.0, 0.0, 1.0])):
        point_list.append(get_sym_point2(point, 0, 0, 0))
        point_list.append(get_sym_point2(point, 1, 0, 0))
        point_list.append(get_sym_point2(point, 0, 0, 1))
        point_list.append(get_sym_point2(point, 1, 0, 1))
    elif sym.equal(torch.tensor([0.0, 1.0, 1.0])):
        point_list.append(get_sym_point2(point, 0, 0, 0))
        point_list.append(get_sym_point2(point, 0, 1, 0))
        point_list.append(get_sym_point2(point, 0, 0, 1))
        point_list.append(get_sym_point2(point, 0, 1, 1))
    else:
        point_list.append(get_sym_point2(point, 0, 0, 0))
        point_list.append(get_sym_point2(point, 1, 0, 0))
        point_list.append(get_sym_point2(point, 0, 1, 0))
        point_list.append(get_sym_point2(point, 0, 0, 1))
        point_list.append(get_sym_point2(point, 1, 1, 0))
        point_list.append(get_sym_point2(point, 1, 0, 1))
        point_list.append(get_sym_point2(point, 0, 1, 1))
        point_list.append(get_sym_point2(point, 1, 1, 1))

    return point_list

def get_min_l2_dist2(list1, list2, center1, center2, quat1, quat2):

    list1 = torch.tensor(list1) # m x 3
    list2 = torch.tensor(list2) # n x 3
    len1 = list1.shape[0]
    len2 = list2.shape[0]
    center1 = center1.unsqueeze(0).repeat(len1, 1)
    center2 = center2.unsqueeze(0).repeat(len2, 1)
    quat1 = quat1.unsqueeze(0).repeat(len1, 1)
    quat2 = quat2.unsqueeze(0).repeat(len2, 1)
    list1 = list1.to(center1.device)
    list2 = list2.to(center1.device)
    list1 = center1 + qrot(quat1, list1)
    list2 = center2 + qrot(quat2, list2)
    mat1 = list1.unsqueeze(1).repeat(1, len2, 1)
    mat2 = list2.unsqueeze(0).repeat(len1, 1, 1)
    mat = (mat1 - mat2) * (mat1 - mat2)
    #ipdb.set_trace()
    mat = mat.sum(dim=-1)
    return mat.min()

def get_shape_cd_loss2(pts, quat1, quat2, valids, center1, center2):
        batch_size = pts.shape[0]
        num_part = pts.shape[1]
        num_point = pts.shape[2]
        center1 = center1.unsqueeze(2).repeat(1,1,num_point,1)
        center2 = center2.unsqueeze(2).repeat(1,1,num_point,1)
        pts1 = qrot(quat1.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center1
        pts2 = qrot(quat2.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center2

        pts1 = pts1.view(batch_size,num_part*num_point,3)
        pts2 = pts2.view(batch_size,num_part*num_point,3)
        dist1, dist2 = chamfer_distance(pts1, pts2, transpose=False)
        valids = valids.unsqueeze(2).repeat(1,1,1000).view(batch_size,-1)
        dist1 = dist1 * valids
        dist2 = dist2 * valids
        loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
        
        loss_per_data = loss_per_data.to(center1.device)
        return loss_per_data

def get_shape_cd_loss(pts, quat1, quat2, center1, center2, valids, return_raw=False):
    batch_size, num_part, num_point, _ = pts.size()

    center1 = center1.unsqueeze(2).repeat(1, 1, num_point, 1)
    center2 = center2.unsqueeze(2).repeat(1, 1, num_point, 1)
    pts1 = qrot(quat1.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center1
    pts2 = qrot(quat2.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center2

    pts1 = pts1.view(batch_size, num_part * num_point, 3)
    pts2 = pts2.view(batch_size, num_part * num_point, 3)
    dist1, dist2 = chamfer_distance(pts1, pts2, transpose=False)
    valids = valids.unsqueeze(2).repeat(1, 1, num_point).view(batch_size, -1)
    dist1 = dist1 * valids
    dist2 = dist2 * valids
    loss_per_data = (torch.sum(dist1, dim=1) + torch.sum(dist2, dim=1)) / torch.sum(valids, dim=1)
    if return_raw:
        return dist1, dist2
    else:
        return loss_per_data

def get_shape_transformed(pts, quat, center):
    """
        Input: B x P x N x 3, B x P x 3, B x P x 3, B x P x 4, B x P x 4, B x P
        Output: B
    """
    batch_size, num_part, num_point, _ = pts.size()

    center = center.unsqueeze(2).repeat(1, 1, num_point, 1)
    pts = qrot(quat.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center

    pts = pts.view(batch_size, num_part, num_point, 3)
    return pts


# Following the implementation in Generative 3D Part Assembly via Dynamic Graph Learning
def get_shape_cd_loss_default(pts, quat1, quat2, center1, center2, valids, return_raw=False):
    """
        Input: B x P x N x 3, B x P x 3, B x P x 3, B x P x 4, B x P x 4, B x P
        Output: B
    """
    batch_size, num_part, num_point, _ = pts.size()

    center1 = center1.unsqueeze(2).repeat(1, 1, num_point, 1)
    center2 = center2.unsqueeze(2).repeat(1, 1, num_point, 1)
    pts1 = qrot(quat1.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center1
    pts2 = qrot(quat2.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center2

    pts1 = pts1.view(batch_size, num_part * num_point, 3)
    pts2 = pts2.view(batch_size, num_part * num_point, 3)
    dist1, dist2 = chamfer_distance(pts1, pts2, transpose=False)
    valids = valids.unsqueeze(2).repeat(1, 1, num_point).view(batch_size, -1)
    dist1 = dist1 * valids
    dist2 = dist2 * valids
    loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)

    if return_raw:
        return dist1, dist2
    else:
        return loss_per_data


def get_total_cd_loss(pts, quat1, quat2, center1, center2, valids):
    """
        Input: B x P x N x 3, B x P x 3, B x P x 3, B x P x 4, B x P x 4, B x P
        Output: B, B x P
    """
    batch_size, num_part, num_point, _ = pts.size()

    center1 = center1.unsqueeze(2).repeat(1, 1, num_point, 1)
    center2 = center2.unsqueeze(2).repeat(1, 1, num_point, 1)
    pts1 = qrot(quat1.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center1
    pts2 = qrot(quat2.unsqueeze(2).repeat(1, 1, num_point, 1), pts) + center2

    dist1, dist2 = chamfer_distance(pts1.view(-1, num_point, 3), pts2.view(-1, num_point, 3), transpose=False)
    loss_per_data = torch.mean(dist1, dim=1) + torch.mean(dist2, dim=1)
    loss_per_data = loss_per_data.view(batch_size, -1)

    thresh = 0.01
    acc = (loss_per_data < thresh).float() * valids
    loss_per_data = (loss_per_data * valids).sum(1) / valids.sum(1)

    return loss_per_data, acc


def get_sym_point(point, x, y, z):
    if point.dim() == 1:
        if x:
            point[0] = - point[0]
        if y:
            point[1] = - point[1]
        if z:
            point[2] = - point[2]

    elif point.dim() == 2:
        if x:
            point[:, 0] = - point[:, 0]
        if y:
            point[:, 1] = - point[:, 1]
        if z:
            point[:, 2] = - point[:, 2]

    else:
        raise NotImplementedError

    return point.tolist()


def get_possible_point_list(point, sym=None):
    point_list = []
    sym = torch.tensor([1.0, 1.0, 1.0])
    if sym.equal(torch.tensor([0.0, 0.0, 0.0])):
        point_list.append(get_sym_point(point, 0, 0, 0))
    elif sym.equal(torch.tensor([1.0, 0.0, 0.0])):
        point_list.append(get_sym_point(point, 0, 0, 0))
        point_list.append(get_sym_point(point, 1, 0, 0))
    elif sym.equal(torch.tensor([0.0, 1.0, 0.0])):
        point_list.append(get_sym_point(point, 0, 0, 0))
        point_list.append(get_sym_point(point, 0, 1, 0))
    elif sym.equal(torch.tensor([0.0, 0.0, 1.0])):
        point_list.append(get_sym_point(point, 0, 0, 0))
        point_list.append(get_sym_point(point, 0, 0, 1))
    elif sym.equal(torch.tensor([1.0, 1.0, 0.0])):
        point_list.append(get_sym_point(point, 0, 0, 0))
        point_list.append(get_sym_point(point, 1, 0, 0))
        point_list.append(get_sym_point(point, 0, 1, 0))
        point_list.append(get_sym_point(point, 1, 1, 0))
    elif sym.equal(torch.tensor([1.0, 0.0, 1.0])):
        point_list.append(get_sym_point(point, 0, 0, 0))
        point_list.append(get_sym_point(point, 1, 0, 0))
        point_list.append(get_sym_point(point, 0, 0, 1))
        point_list.append(get_sym_point(point, 1, 0, 1))
    elif sym.equal(torch.tensor([0.0, 1.0, 1.0])):
        point_list.append(get_sym_point(point, 0, 0, 0))
        point_list.append(get_sym_point(point, 0, 1, 0))
        point_list.append(get_sym_point(point, 0, 0, 1))
        point_list.append(get_sym_point(point, 0, 1, 1))
    else:
        point_list.append(get_sym_point(point, 0, 0, 0))
        point_list.append(get_sym_point(point, 1, 0, 0))
        point_list.append(get_sym_point(point, 0, 1, 0))
        point_list.append(get_sym_point(point, 0, 0, 1))
        point_list.append(get_sym_point(point, 1, 1, 0))
        point_list.append(get_sym_point(point, 1, 0, 1))
        point_list.append(get_sym_point(point, 0, 1, 1))
        point_list.append(get_sym_point(point, 1, 1, 1))
    return point_list


def get_min_l2_dist(list1, list2, center1, center2, quat1, quat2):
    num_part = list1.size(0)
    len1 = list1.size(1)
    len2 = list2.size(1)

    center1 = center1.unsqueeze(1).repeat(1, len1, 1)
    center2 = center2.unsqueeze(1).repeat(1, len2, 1)
    quat1 = quat1.unsqueeze(1).repeat(1, len1, 1)
    quat2 = quat2.unsqueeze(1).repeat(1, len2, 1)

    list1 = center1 + qrot(quat1, list1)
    list2 = center2 + qrot(quat2, list2)

    mat1 = list1.unsqueeze(2).repeat(1, 1, len2, 1)
    mat2 = list2.unsqueeze(1).repeat(1, len1, 1, 1)
    mat = (mat1 - mat2) * (mat1 - mat2)
    mat = mat.sum(dim=-1).view(num_part, -1)
    dist, _ = mat.min(-1)
    return dist


def get_contact_point_loss(center, quat, contact_points, sym_info):
    """
        Contact point loss metric
        Input: B x P x 3, B x P x 4, B x P x P x 4, B x P x 3
        Ouput: B
    """
    batch_size, num_part, _ = center.size()
    contact_point_loss = center.new_zeros(batch_size)
    num_contact_pairs = center.new_zeros(batch_size)
    num_correct_pairs = center.new_zeros(batch_size)
    thresh = 0.01
    for bs_ind in range(batch_size):
        cur_contact_point = contact_points[bs_ind]  # P x P x 4
        contact_1 = (cur_contact_point[..., 0] == 1).view(-1)  # P*P
        contact_2 = ((cur_contact_point[..., 0].transpose(0, 1).contiguous()) == 1).view(-1)  # P*P
        if contact_1.sum() == 0:
            continue

        contact_point_1 = cur_contact_point.view(-1, 4)[contact_1][:, 1:]
        contact_point_2 = cur_contact_point.transpose(0, 1).contiguous().view(-1, 4)[contact_2][:, 1:]

        cur_sym = sym_info[bs_ind]  # ignore actually.
        point_list_1 = center.new_tensor(get_possible_point_list(contact_point_1)).transpose(0, 1).contiguous()
        point_list_2 = center.new_tensor(get_possible_point_list(contact_point_2)).transpose(0, 1).contiguous()

        cur_center = center[bs_ind]
        center_1 = cur_center.unsqueeze(1).repeat(1, num_part, 1).view(-1, 3)[contact_1]
        center_2 = cur_center.unsqueeze(0).repeat(num_part, 1, 1).view(-1, 3)[contact_2]

        cur_quat = quat[bs_ind]
        quat_1 = cur_quat.unsqueeze(1).repeat(1, num_part, 1).view(-1, 4)[contact_1]
        quat_2 = cur_quat.unsqueeze(0).repeat(num_part, 1, 1).view(-1, 4)[contact_2]

        dists = get_min_l2_dist(point_list_1, point_list_2, center_1, center_2, quat_1, quat_2)
        num_correct_pairs[bs_ind] = (dists < thresh).sum()
        num_contact_pairs[bs_ind] = contact_1.sum()
        contact_point_loss[bs_ind] = dists.sum()
    return contact_point_loss, num_correct_pairs, num_contact_pairs


def get_contact_point_loss_for_single_part(center, quat, contact_points, sym_info, part_mask):
    """
        Contact point loss metric
        Input: B x P x 3, B x P x 4, B x P x P x 4, B x P x 3
        Ouput: B
    """
    batch_size, num_part, _ = center.size()
    contact_point_loss = center.new_zeros(batch_size)
    num_contact_pairs = center.new_zeros(batch_size)
    num_correct_pairs = center.new_zeros(batch_size)
    thresh = 0.01
    _, pos_ids = (~part_mask).nonzero(as_tuple=True)
    for bs_ind in range(batch_size):
        cur_contact_point = contact_points[bs_ind]  # P x P x 4
        pos_id = pos_ids[bs_ind]

        contact = (cur_contact_point[pos_id][..., 0] == 1).view(-1)  # P
        if contact.sum() == 0:
            continue
        contact_3 = contact.unsqueeze(-1).repeat(1, 3)
        contact_4 = contact.unsqueeze(-1).repeat(1, 4)

        contact_point_1 = cur_contact_point[pos_id][contact_4].view(-1, 4).contiguous()[:, 1:]
        contact_point_2 = cur_contact_point.transpose(0, 1).contiguous()[pos_id][contact_4].view(-1, 4).contiguous()[:, 1:]

        point_list_1 = center.new_tensor(get_possible_point_list(contact_point_1)).transpose(0, 1).contiguous()
        point_list_2 = center.new_tensor(get_possible_point_list(contact_point_2)).transpose(0, 1).contiguous()

        cur_center = center[bs_ind]
        center_1 = cur_center.unsqueeze(1).repeat(1, num_part, 1)[pos_id][contact_3].view(-1, 3).contiguous()
        center_2 = cur_center.unsqueeze(0).repeat(num_part, 1, 1)[pos_id][contact_3].view(-1, 3).contiguous()

        cur_quat = quat[bs_ind]
        quat_1 = cur_quat.unsqueeze(1).repeat(1, num_part, 1)[pos_id][contact_4].view(-1, 4).contiguous()
        quat_2 = cur_quat.unsqueeze(0).repeat(num_part, 1, 1)[pos_id][contact_4].view(-1, 4).contiguous()

        dists = get_min_l2_dist(point_list_1, point_list_2, center_1, center_2, quat_1, quat_2)
        num_correct_pairs[bs_ind] = (dists < thresh).sum()
        num_contact_pairs[bs_ind] = contact.sum()
        contact_point_loss[bs_ind] = dists.sum()

    return contact_point_loss, num_correct_pairs, num_contact_pairs
