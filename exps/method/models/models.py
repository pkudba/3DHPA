# coding: utf-8
import copy
from typing import Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn

import math

from .SAmodule import selfAttention


class Predictor(nn.Module):

    def __init__(self, feat_dim, mode='relative'):
        super(Predictor, self).__init__()
        self.mode = mode
        
        self.mlp = nn.Linear(feat_dim, 1024)

        self.trans = nn.Sequential(
                nn.Linear(1024, 512),
                nn.Linear(512, 3)
            )
        self.quat = nn.Sequential(
                nn.Linear(1024, 512),
                nn.Linear(512, 4)
            )

    def forward(self, feat):
        feat = torch.relu(self.mlp(feat))
        if self.mode=='relative':
            trans = torch.tanh(self.trans(feat))  
        else:
            trans = torch.tanh(self.trans(feat))
        quat_bias = feat.new_tensor([[[1.0, 0.0, 0.0, 0.0]]])
        quat = self.quat(feat).add(quat_bias)

        quat = quat / (1e-12 + quat.pow(2).sum(dim=-1, keepdim=True)).sqrt()

        out = torch.cat([trans, quat], dim=-1)
        return out


class FinalPoseHead(nn.Module):

    def __init__(self, feat_dim):
        super(FinalPoseHead, self).__init__()

        self.mlp = nn.Linear(feat_dim, 128)

        self.trans = nn.Linear(128, 3)

        self.quat = nn.Linear(128, 4)
        self.quat.bias.data.zero_()

    def forward(self, feat):
        feat = torch.relu(self.mlp(feat))

        trans = torch.tanh(self.trans(feat))  

        quat_bias = feat.new_tensor([[[1.0, 0.0, 0.0, 0.0]]])
        quat = self.quat(feat).add(quat_bias)
        quat = quat / (1e-12 + quat.pow(2).sum(dim=-1, keepdim=True)).sqrt()

        out = torch.cat([trans, quat], dim=-1)
        return out



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        mlp = list()
        for l in range(num_layers):
            if l == 0:
                mlp.append(nn.Linear(input_dim, hidden_dim))
                mlp.append(nn.LayerNorm(hidden_dim))
            elif l == num_layers - 1:
                mlp.append(nn.Linear(hidden_dim, output_dim))
                mlp.append(nn.LayerNorm(output_dim))
            else:
                mlp.append(nn.Linear(hidden_dim, hidden_dim))
                mlp.append(nn.LayerNorm(hidden_dim))
            mlp.append(nn.ReLU(inplace=True))
        self.mlp = nn.ModuleList(mlp)

    def forward(self, x):
        for layer in self.mlp:
            x = layer(x)
        return x


class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = m.detach().clone().requires_grad_(False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x
    
    

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, return_intermediate=False, mode='fixed', args=None):
        super().__init__()
        self.args = args
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.mode = mode
        self.return_intermediate = return_intermediate

        dim_in = encoder_layer.d_model
        if self.mode=='base':
            dim_in = dim_in + 20
        if args.base_cat:
            dim_in = dim_in + args.feat_dim
        if args.pose_cat:
            dim_in = dim_in + 7
        if (args.noise_cat==1) & (self.mode!='base'):
            dim_in = dim_in + args.noise_dim
        if (args.ins_cat==1) & (self.mode != 'base'):
            assert (args.ins_cat_inter_only and args.ins_cat_intra_only) is not True
            dim_in = dim_in + 40

        if self.mode=='relative':
            mlp = MLP(dim_in, args.feat_dim, args.feat_dim, args.num_mlp)
            predictor = Predictor(args.feat_dim, mode='relative')
        else:
            mlp = MLP(dim_in, args.feat_dim, args.feat_dim, args.num_mlp)
            predictor = Predictor(args.feat_dim, mode='base')

        if not args.shared_pred:
            self.mlp = _get_clones(mlp, num_layers)
            self.predictor = _get_clones(predictor, num_layers)
        else:
            self.mlp = mlp
            self.predictor = predictor
        self.embedding = nn.Linear(512, 256)
        self.dim_in = dim_in
        self.skipcn0 = nn.Linear(2*args.feat_dim,args.feat_dim)
        self.skipcn1 = nn.Linear(2*args.feat_dim,args.feat_dim)
        self.skipcn2 = nn.Linear(2*args.feat_dim,args.feat_dim)

    def forward(self, match_codes, cross, src, init_pred, ins_codes, 
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                lens: Optional[Tensor] = None,
                gt_part_poses = None,
                normed_match = None,
                **kwargs):

        output = src
        intermediate = []
        num_part, batch_size, _ = src.size()

        if self.args.noise_cat or self.args.noise_cat_in_encoder:
            assert self.args.train_mon > 1 or self.args.eval_mon > 1
            random_noise = torch.normal(mean=0., std=1., size=(num_part, batch_size, self.args.noise_dim)).to(src.device)

        pred = init_pred
        
        noise2 = np.random.normal(loc=0.0, scale=1.0, size=[1, batch_size, self.args.noise_dim]).astype(np.float32)
        noise2 = torch.tensor(noise2, requires_grad = False).cuda()
        noise2 = noise2.repeat(2, 1, 1)
        init_hidden = torch.zeros(2, batch_size, self.args.hidden_gru - self.args.noise_dim, requires_grad=False).cuda()
        decoder_hidden = torch.cat([init_hidden, noise2], dim = -1)
        cross_output = output.clone()

        for idx, layer in enumerate(self.layers):
            if idx==3:
                output = self.skipcn2(torch.cat((output, record_out2), dim=-1))
            elif idx==4:
                output = self.skipcn1(torch.cat((output, record_out1), dim=-1))
            elif idx==5:
                output = self.skipcn0(torch.cat((output, record_out0), dim=-1))
            if self.mode=='relative':
                output = torch.cat((output, init_pred), dim=-1)
                if idx==0:
                    cross = torch.cat((cross, init_pred), dim=-1)
            if self.args.pose_cat_in_encoder:
                output = torch.cat((output, pred), dim=-1)
                if self.mode=='relative' and idx==0:
                    cross = torch.cat((cross, pred), dim=-1)
            if (self.args.noise_cat_in_encoder==1) & (self.mode!='base'):
                output = torch.cat((output, random_noise), dim=-1)
                if idx==0:
                    cross = torch.cat((cross, random_noise), dim=-1)
            if self.mode=='base':
                output = torch.cat((output, ins_codes), dim=-1)
            if (self.args.ins_cat_in_encoder==1) & (self.mode!='base'):
                cat = ins_codes
                output = torch.cat((output, cat), dim=-1)
                cross_output = torch.cat((cross_output, cat), dim=-1)
                if idx==0:
                    cross = torch.cat((cross, cat), dim=-1)

            if self.mode=='relative':
                output, cross_output = layer(match_codes, cross, output, cross_output, src_mask=mask,
                               src_key_padding_mask=src_key_padding_mask, pos=pos)

                if self.norm is not None:
                    output = self.norm(output)
                    cross_output = self.norm(cross_output)

                fuse = output
                feat = fuse.clone()
            
            else:
                output = layer(match_codes, cross, output, cross_output, src_mask=mask,
                               src_key_padding_mask=src_key_padding_mask, pos=pos)

                if self.norm is not None:
                    output = self.norm(output)
                feat = output.clone()
            
            if idx==0:
                record_out0 = feat.clone()
            elif idx==1:
                record_out1 = feat.clone()
            elif idx==2:
                record_out2 = feat.clone()

            if self.args.base_cat:
                feat = torch.cat((feat, src), dim=-1)
            if self.args.pose_cat:
                feat = torch.cat((feat, pred), dim=-1)
            if (self.args.noise_cat==1) & (self.mode!='base'):
                feat = torch.cat((feat, random_noise), dim=-1)
            if (self.args.ins_cat==1) & (self.mode!='base'):
                feat = torch.cat((feat, ins_codes), dim=-1)
            if self.mode=='base':
                feat = torch.cat((feat, ins_codes), dim=-1)

            if self.mode=='relative' and self.args.model_version=='autoreg2':
                if not self.args.shared_pred:
                    feat = self.mlp[idx](feat)
                    pred = self.predictor[idx](feat)
                else:
                    feat = self.mlp(feat)
                    input_pose = torch.zeros_like(gt_part_poses)
                    input_pose[1:,:,:] = gt_part_poses[:-1,:,:]
                    pred = self.predictor(feat, decoder_hidden, lens)
            else:
                if not self.args.shared_pred:
                    feat = self.mlp[idx](feat)
                    pred = self.predictor[idx](feat)
                else:
                    feat = self.mlp(feat)
                    if self.mode=='relative':
                        pred = self.predictor(feat)
                    else:
                        pred = self.predictor(feat)

            if self.return_intermediate:
                intermediate.append(pred)

            if self.args.pred_detach:
                with torch.no_grad():
                    pred = pred.detach()
        if self.return_intermediate:
            return torch.stack(intermediate), output
        else:
            return pred.unsqueeze(0), output
 


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=0, mode='fixed', args=None):
        super().__init__()
        self.args = args
        dim_in = d_model
        self.mode = mode
        self.dim_in = d_model
        self.nhead = nhead
        if self.mode=='relative':
            dim_in += 7
        if args.pose_cat_in_encoder:
            dim_in += 7
        if (self.args.noise_cat_in_encoder==1) & (self.mode!='base'):
            dim_in += args.noise_dim
        if (args.ins_cat_in_encoder==1) & (self.mode!='base'):
            assert (args.ins_cat_inter_only and args.ins_cat_intra_only) is not True
            dim_in += 40
        if self.mode=='base':
            dim_in += 20
        self.proj = nn.Linear(dim_in, d_model)
        self.proj_cross = nn.Linear(dim_in, d_model)
        self.proj_cross_src = nn.Linear(dim_in-14-args.noise_dim, d_model)
        self.new_proj_cross = nn.Linear(dim_in+256, d_model)

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn_cross = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.part_attn = selfAttention(self.nhead, d_model, d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.d_model = d_model

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, match_codes, 
                     cross, ori_src, cross_src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):

        if self.mode=='relative':
            src = self.proj(ori_src)
            cross_q = self.with_pos_embed(src, pos)
            cross = self.proj(cross)
            cross_q = self.with_pos_embed(src, pos)
            cross_k = self.with_pos_embed(cross, pos)
            src_psp, src_psp_attn_weights = self.self_attn_cross(cross_q, cross_k, value=cross, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)
            src = src + src_psp
            q = self.with_pos_embed(src, pos)
            k = self.with_pos_embed(src, pos)

            src2, src2_attn_weights = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
            src2 = torch.where(torch.isnan(src2), torch.full_like(src2, 0), src2)
        else:
            src = self.proj(ori_src)
            q = k = self.with_pos_embed(src, pos)
            src2, src2_attn_weights = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
            src2 = torch.where(torch.isnan(src2), torch.full_like(src2, 0), src2)

        if self.args.offset_attention==1:
            src = src + self.dropout1(src-src2)
        elif self.args.offset_attention==0:
            src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        if self.mode=='relative':
            return src, src
        return src

    def forward_pre(self, cross, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2, src2_attn_weights = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src2 = torch.where(torch.isnan(src2), torch.full_like(src2, 0), src2)
        if self.args.offset_attention==1:
            src = src + self.dropout1(src-src2)
        elif self.args.offset_attention==0:
            src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, match_codes, cross, src, cross_src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before==1:
            return self.forward_pre(cross, src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(match_codes, cross, src, cross_src, src_mask, src_key_padding_mask, pos)

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False, args=None):
        super().__init__()
        self.args = args
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

        dim_in = decoder_layer.d_model
        if args.pose_cat_in_decoder_pred:
            dim_in = dim_in + 7
        if args.noise_cat_in_decoder_pred:
            dim_in = dim_in + args.noise_dim

        pose_mlp = MLP(dim_in, args.feat_dim, args.feat_dim, args.num_mlp)
        pose_pred = Predictor(args.feat_dim)
        if self.args.cate_on:
            cate_mlp = MLP(dim_in, args.feat_dim, args.feat_dim, args.num_mlp)
            cate_pred = nn.Linear(args.feat_dim, 1)
        if not args.shared_pred:
            self.pose_mlp = _get_clones(pose_mlp, num_layers)
            self.pose_pred = _get_clones(pose_pred, num_layers)
            if self.args.cate_on:
                self.cate_mlp = _get_clones(cate_mlp, num_layers)
                self.cate_pred = _get_clones(cate_pred, num_layers)
        else:
            self.pose_mlp = pose_mlp
            self.pose_pred = pose_pred
            if self.args.cate_on:
                self.cate_mlp = cate_mlp
                self.cate_pred = cate_pred

    def forward(self, tgt, memory, ins_codes,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                **kwargs):
        num_part, batch_size, _ = tgt.size()

        if self.args.noise_cat_in_decoder_pred or self.args.noise_cat_in_decoder_trans:
            assert self.args.train_mon > 1 or self.args.eval_mon > 1
            random_noise = torch.normal(mean=0., std=1., size=(num_part, batch_size, self.args.noise_dim)).to(tgt.device)

        if self.args.pose_cat_in_memory:
            memory_poses = kwargs["memory_poses"]
        if self.args.noise_cat_in_memory:
            num_memory, _, len_memory = memory.size()
            memory_noise = torch.normal(mean=0., std=1., size=(num_memory, batch_size, self.args.noise_dim)).to(memory.device)

        output = tgt
        pose_inters = []
        cate_inters = []

        for idx, layer in enumerate(self.layers):
            if idx == 0:
                pred = tgt.new_zeros((num_part, batch_size, 7))

            if self.args.pose_cat_in_decoder_trans:
                output = torch.cat((output, pred), dim=-1)
            if self.args.noise_cat_in_decoder_trans:
                output = torch.cat((output, random_noise), dim=-1)

            if self.args.pose_cat_in_memory:
                memory = torch.cat((memory, memory_poses), dim=-1)
            if self.args.noise_cat_in_memory:
                memory = torch.cat((memory, memory_noise), dim=-1)

            if self.args.ins_cat_in_decoder:
                output = torch.cat((output, ins_codes), dim=-1)
                memory = torch.cat((memory, ins_codes), dim=-1)

            output, memory = layer(output, memory, tgt_mask=tgt_mask,
                                   memory_mask=memory_mask,
                                   tgt_key_padding_mask=tgt_key_padding_mask,
                                   memory_key_padding_mask=memory_key_padding_mask,
                                   pos=pos, query_pos=query_pos)

            if self.norm is not None:
                output = self.norm(output)
            feat = output.clone()

            if self.args.pose_cat_in_decoder_pred:
                feat = torch.cat((feat, pred), dim=-1)
            if self.args.noise_cat_in_decoder_pred:
                feat = torch.cat((feat, random_noise), dim=-1)

            if not self.args.shared_pred:
                pose_feat = self.pose_mlp[idx](feat)
                pose_pred = self.pose_pred[idx](pose_feat)
                if self.args.cate_on:
                    cate_feat = self.cate_mlp[idx](feat)
                    cate_pred = self.cate_pred[idx](cate_feat)
            else:
                pose_feat = self.pose_mlp(feat)
                pose_pred = self.pose_pred(pose_feat)
                if self.args.cate_on:
                    cate_feat = self.cate_mlp(feat)
                    cate_pred = self.cate_pred(cate_feat)

            if self.return_intermediate:
                pose_inters.append(pose_pred)
                if self.args.cate_on:
                    cate_inters.append(cate_pred)
                else:
                    cate_inters.append(pose_pred.new_zeros(pose_pred.size()))

            if self.args.pred_detach:
                with torch.no_grad():
                    pred = pose_pred.detach()

        if self.return_intermediate:
            return torch.stack(pose_inters), torch.stack(cate_inters)
        else:
            return pose_inters.unsqueeze(0), cate_inters.unsqueeze(0)
         

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=0, args=None):
        super().__init__()
        self.args = args
        dim_in = d_model
        if args.pose_cat_in_decoder_trans:
            dim_in += 7
        if args.noise_cat_in_decoder_trans:
            dim_in += args.noise_dim

        dim_memory = d_model
        if args.pose_cat_in_memory:
            dim_memory += 7
        if args.noise_cat_in_memory:
            dim_memory += args.noise_dim

        if args.ins_cat_in_decoder:
            dim_in += args.max_num_part * 2
            dim_memory += args.max_num_part * 2
        self.proj = nn.Linear(dim_in, d_model)
        self.memory_proj = nn.Linear(dim_memory, d_model)

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.d_model = d_model

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        tgt = self.proj(tgt)
        memory = self.memory_proj(memory)
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos), memory
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos), memory

