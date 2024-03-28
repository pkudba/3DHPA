import argparse
import h5py
import torch
import scipy.spatial
import collections
from matplotlib import pyplot as plt
import pytorch3d.ops
import numpy as np
import random

def softmax(x):
    exp_x = torch.exp(x)
    result = torch.zeros([1, 8]).cuda(device=6)
    for y in exp_x:
        result = torch.cat((result, (y/torch.sum(y)).unsqueeze(0)), 0)
    return result[1:321, :]


def create_output(vertices, colors, filename):
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])
    np.savetxt(filename, vertices, fmt='%f %f %f %d %d %d')     
    ply_header = '''ply
    		format ascii 1.0
    		element vertex %(vert_num)d
    		property float x
    		property float y
    		property float z
    		property uchar red
    		property uchar green
    		property uchar blue
    		end_header
    		\n
    		'''
    with open(filename, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header % dict(vert_num=len(vertices)))
        f.write(old)     


def Addlabel(x, lst):
    for index, line in enumerate(lst):
        if x == line.split()[1]:
            return index
    return 0


def load_h5_data_label_seg(file_name):
    f = h5py.File(file_name)
    data = f['data'][:]  
    label = f['label'][:]  
    seg = f['pid'][:] 
    return data, label, seg


def generate_pairs(x, num):
    normals_ds = x[0:num, :]
    cnt = 0
    flag = np.zeros((320,), dtype=int)
    chooses = np.empty([320, 2])
    cangle = np.empty(320)
    while cnt < 320:
        if cnt < 100:
            sample_patch = cnt
            t1 = sample_patch
            t2 = sample_patch+1
        elif cnt < 200:
            sample_patch = cnt-100
            t1 = sample_patch
            t2 = sample_patch+9
        elif cnt < 290:
            sample_patch = cnt-200
            t1 = sample_patch
            t2 = sample_patch+25
        else:
            sample_patch = random.randint(1, 70)
            if flag[sample_patch] == 1:
                continue
            t1 = sample_patch
            t2 = t1 + 49
            flag[sample_patch] = 1
        chooses[cnt][0] = t1
        chooses[cnt][1] = t2

        cangle[cnt] = CosVector(normals_ds[t1], normals_ds[t2])
        cnt += 1
    
    chooses = torch.tensor(chooses, dtype=torch.int64)
    chooses = chooses.permute(1, 0)
    cangle = torch.tensor(cangle, dtype=torch.float64)
    return chooses, cangle


def import_class(name):
    try:
        components = name.split('.')
        module = __import__(components[0])
        for c in components[1:]:
            module = getattr(module, c)
    except AttributeError:
        module = None
    return module


def save_weights(epoch, model, optimizer, save_path):
    model_weights = collections.OrderedDict([
        (k.split('module.')[-1], v.cpu())
        for k, v in model.state_dict().items()
    ])
    optim_weights = optimizer.state_dict()
    save_dict = {
        'epoch': epoch,
        'model': model_weights,
        'optimizer': optim_weights,
    }
    torch.save(save_dict, save_path)


def draw_hist(myList, Title, Xlabel, Ylabel, Xmin, Xmax, Ymin, Ymax):
    plt.hist(myList, 100)
    plt.Xlabel(Xlabel)
    plt.Ylabel(Ylabel)
    plt.ylim(Ymin, Ymax)
    plt.xlim(Xmin, Xmax)
    plt.title(Title)
    plt.show()


def knn2(x, y, k, batch_x=None, batch_y=None):
    if batch_x is None:
        batch_x = x.new_zeros(x.size(0), dtype=torch.long)

    if batch_y is None:
        batch_y = y.new_zeros(y.size(0), dtype=torch.long)

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y

    assert x.dim() == 2 and batch_x.dim() == 1
    assert y.dim() == 2 and batch_y.dim() == 1
    assert x.size(1) == y.size(1)
    assert x.size(0) == batch_x.size(0)
    assert y.size(0) == batch_y.size(0)

    # Rescale x and y.
    min_xy = min(x.min().item(), y.min().item())
    x, y = x - min_xy, y - min_xy

    max_xy = max(x.max().item(), y.max().item())
    x, y, = x / max_xy, y / max_xy

    # Concat batch/features to ensure no cross-links between examples exist.
    x = torch.cat([x, 2 * x.size(1) * batch_x.view(-1, 1).to(x.dtype)], dim=-1)
    y = torch.cat([y, 2 * y.size(1) * batch_y.view(-1, 1).to(y.dtype)], dim=-1)

    tree = scipy.spatial.cKDTree(x.detach().numpy())
    dist, col = tree.query(
        y.detach().cpu(), k=k, distance_upper_bound=x.size(1))
    dist = torch.from_numpy(dist).to(x.dtype)
    col = torch.from_numpy(col).to(torch.long)
    row = torch.arange(col.size(0), dtype=torch.long).view(-1, 1).repeat(1, k)
    mask = ~torch.isinf(dist).view(-1)
    row, col = row.view(-1)[mask], col.view(-1)[mask]

    return torch.stack([row, col], dim=0)


def radius(x, y, r, batch_x=None, batch_y=None, max_num_neighbors=64):
    if batch_x is None:
        batch_x = x.new_zeros(x.size(0), dtype=torch.long)

    if batch_y is None:
        batch_y = y.new_zeros(y.size(0), dtype=torch.long)

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y

    assert x.dim() == 2 and batch_x.dim() == 1
    assert y.dim() == 2 and batch_y.dim() == 1
    assert x.size(1) == y.size(1)
    assert x.size(0) == batch_x.size(0)
    assert y.size(0) == batch_y.size(0)

    x = torch.cat([x, 2 * r * batch_x.view(-1, 1).to(x.dtype)], dim=-1)
    y = torch.cat([y, 2 * r * batch_y.view(-1, 1).to(y.dtype)], dim=-1)

    tree = scipy.spatial.cKDTree(x.detach().numpy())
    _, col = tree.query(
        y.detach().numpy(), k=max_num_neighbors, distance_upper_bound=r + 1e-8)
    col = [torch.from_numpy(c).to(torch.long) for c in col]
    row = [torch.full_like(c, i) for i, c in enumerate(col)]
    row, col = torch.cat(row, dim=0), torch.cat(col, dim=0)
    mask = col < int(tree.n)
    return torch.stack([row[mask], col[mask]], dim=0)


def load_h5(file_name):
    f = h5py.File(file_name)
    data = f['data'][:]
    label = f['label'][:]
    normal = f['normal'][:]
    return data, label, normal


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_total_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total, 'Trainable': trainable}


def knn_o(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    x2 = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -x2 - inner - x2.transpose(2, 1)
    indices = pairwise_distance.topk(k=k, dim=-1)[1]
    return indices  

def get_edge_feature(x, k=20):
    batch_siz = x.size(0)
    num_points = x.size(2)
    num_feats = x.size(1)
    indices = knn_o(x.view(batch_siz, -1, num_points), k=k) 

    indices_base = torch.arange(
        0, batch_siz, device=x.device).view(-1, 1, 1) * num_points
    indices = indices + indices_base
    indices = indices.view(-1)

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_siz * num_points, -1)[indices, :]
    feature = feature.view(batch_siz, num_points, k, num_feats)
    x = x.view(batch_siz, num_points, 1, num_feats).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


def CosVector(x, y):
    if(len(x) != len(y)):
        print("error input, x and y not in the same space")
        return
    result1 = 0.0
    result2 = 0.0
    result3 = 0.0
    for i in range(len(x)):
        result1 += x[i] * y[i]
        result2 += x[i] ** 2
        result3 += y[i] ** 2
    return result1/((result2 * result3) ** 0.5)


def compute_dis2(x, y):
    dis2 = (x[0]-y[0])**2 + (x[1]-y[1])**2 + (x[2]-y[2])**2
    return dis2


def main():
    print("load tools")


if __name__ == '__main__':
    main()
