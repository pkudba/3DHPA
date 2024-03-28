# coding: utf-8
"""
    PartNetPartDataset
"""

import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, random_split

DATA_FEATURES = ['part_valids', 'part_ids', 'part_pcs']


class PartNetPartDataset(data.Dataset):
    def __init__(self, data_dir, data_fn, category, data_features=DATA_FEATURES, level=3, max_num_part=20):
        # store parameters.
        self.data_dir = data_dir        # a data directory inside [path/to/codebase]/data/
        self.data_fn = data_fn          # a .npy data indexing file listing all data tuples to load
        self.category = category

        self.max_num_part = max_num_part
        self.max_pairs = max_num_part * (max_num_part - 1) / 2
        self.level = level

        # load data.
        self.data = np.load(os.path.join(self.data_dir, data_fn))

        # data features.
        self.data_features = data_features

        # load category semantic information.
        self.part_sems = []
        self.part_sem2id = dict()

    def get_part_count(self):
        return len(self.part_sems)
        
    def __str__(self):
        strout = '[PartNetPartDataset %s %d] data_dir: %s, data_fn: %s, max_num_part: %d' % \
                (self.category, len(self), self.data_dir, self.data_fn, self.max_num_part)
        return strout

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        shape_id = self.data[index]

        cur_data_fn = os.path.join(self.data_dir, 'shape_data/%s_level' % shape_id + self.level + '.npy')
        cur_data = np.load(cur_data_fn, allow_pickle=True).item()  # assume data is stored in separate .npz file.
        cur_contact_data_fn = os.path.join(self.data_dir, 'contact_points/pairs_with_contact_points_%s_level'
                                           % shape_id + self.level + '.npy')
        cur_contacts = np.load(cur_contact_data_fn, allow_pickle=True)  # P x P x 4

        cur_pts = cur_data['part_pcs']  # P x N x 3
        cur_pose = cur_data['part_poses']  # P x (3+7)
        cur_geo_part_ids = np.array(cur_data['geo_part_ids'])  # P
        cur_sym = cur_data['sym']  # P x 3
        cur_num_part = cur_pts.shape[0]

        data_feats = ()
        for feat in self.data_features:
            if feat == 'contact_points':  # P x P x 4
                out = np.zeros((self.max_num_part, self.max_num_part, 4), dtype=np.float32)
                out[:cur_num_part, :cur_num_part, :] = cur_contacts
                out = torch.from_numpy(out).float().unsqueeze(0)
                data_feats = data_feats + (out,)

            elif feat == 'part_pcs':  # P x N x 3
                # directly returning a None will let data loader
                # with collate_fn=collate_fn_with_none to ignore this data item
                if cur_num_part > self.max_num_part:
                    return None
                out = np.zeros((self.max_num_part, cur_pts.shape[1], 3), dtype=np.float32)
                out[:cur_num_part] = cur_pts
                out = torch.from_numpy(out).float().unsqueeze(0)
                data_feats = data_feats + (out,)

            elif feat == 'part_poses':  # P x (3 + 4)
                if cur_num_part > self.max_num_part:
                    return None
                out = np.zeros((self.max_num_part, 3 + 4), dtype=np.float32)
                out[:cur_num_part] = cur_pose
                out = torch.from_numpy(out).float().unsqueeze(0)
                data_feats = data_feats + (out,)

            elif feat == 'semantic_ids':  # P
                cur_part_ids = cur_data['part_ids']
                if cur_num_part > self.max_num_part:
                    return None
                out = np.zeros((self.max_num_part,), dtype=np.float32)
                out[:cur_num_part] = cur_part_ids
                out = torch.from_numpy(out).float().unsqueeze(0)
                data_feats = data_feats + (out,)

            elif feat == 'part_ids':
                if cur_num_part > self.max_num_part:
                    return None
                mapped_part_ids = cur_geo_part_ids + 1  # start from 1.
                out = np.zeros((self.max_num_part,), dtype=np.float32)
                out[:cur_num_part] = mapped_part_ids
                out = torch.from_numpy(out).float().unsqueeze(0)
                data_feats = data_feats + (out,)

            elif feat == 'part_valids':
                if cur_num_part > self.max_num_part:
                    return None
                out = np.zeros((self.max_num_part,), dtype=np.float32)
                out[:cur_num_part] = 1.
                out = torch.from_numpy(out).float().unsqueeze(0)  # return 1 for the first P parts, 0 for the empty rest
                data_feats = data_feats + (out,)

            elif feat == 'sym':  # symmetry: P x 3
                if cur_num_part > self.max_num_part:
                    return None
                out = np.zeros((self.max_num_part, cur_sym.shape[1]), dtype=np.float32)
                out[:cur_num_part] = cur_sym
                out = torch.from_numpy(out).float().unsqueeze(0)
                data_feats = data_feats + (out,)

            elif feat == 'shape_id':
                data_feats = data_feats + (shape_id,)

            elif feat == 'pairs':
                if cur_num_part > self.max_num_part:
                    return None
                valid_pair_matrix = np.ones((cur_num_part, cur_num_part))
                pair_matrix = np.zeros((self.max_num_part, self.max_num_part))
                pair_matrix[:cur_num_part, :cur_num_part] = valid_pair_matrix
                out = torch.from_numpy(pair_matrix).unsqueeze(0)
                data_feats = data_feats + (out,)
            
            elif feat == 'match_ids':  # repeated part start from 1.
                if cur_num_part > self.max_num_part:
                    return None
                out = np.zeros((self.max_num_part,), dtype=np.float32)
                mapped_part_ids = cur_geo_part_ids + 1  # start from 1.
                out[:cur_num_part] = mapped_part_ids
                map_id = 1
                for i in range(1, self.max_num_part + 1):
                    if i > mapped_part_ids.max():
                        break
                    idx = np.where(out == i)[0]
                    idx = torch.from_numpy(idx)
                    if len(idx) == 0:
                        continue
                    elif len(idx) == 1:
                        out[idx] = 0
                    else:
                        out[idx] = map_id
                        map_id += 1
                data_feats = data_feats + (out,)

            else:
                raise ValueError('ERROR: unknown feat type %s!' % feat)

        return data_feats


if __name__ == "__main__":
    dataset = PartNetPartDataset(data_dir="/data/pkudba/datasets/partnet/prepare_data", data_fn="Chair.train.npy",
                                 category="Chair", data_features=DATA_FEATURES, level='3', max_num_part=100)
    for data in dataset:
        aa = 1

