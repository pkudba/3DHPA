import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from .pointnet_utils import PointNetEncoder, feature_transform_reguliarzer, PointNetEncoder_sm


class get_model(nn.Module):
    def __init__(self, k=40, normal_channel=True, softmax_on=False):
        super(get_model, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.softmax_on = softmax_on

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        if self.softmax_on:
            x = F.log_softmax(x, dim=1)
        return x, trans_feat


class get_model_sm(nn.Module):
    def __init__(self, k=40, normal_channel=True, softmax_on=False):
        super(get_model_sm, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder_sm(global_feat=True, feature_transform=True, channel=channel)
        self.fc3 = nn.Linear(256, k)
        self.softmax_on = softmax_on

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = self.fc3(x)
        if self.softmax_on:
            x = F.log_softmax(x, dim=1)
        return x, trans_feat

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss


class trans_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(trans_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, trans_feat):
        # loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        total_loss = mat_diff_loss * self.mat_diff_loss_scale
        return total_loss


if __name__ == "__main__":
    model = get_model(k=256, normal_channel=False).cuda()
    xyz = torch.rand(4, 3, 1000).cuda()
    output = model(xyz)
