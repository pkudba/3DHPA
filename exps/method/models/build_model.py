# coding: utf-8
import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor

# No more decoder in the final version
from .models import MLP, Predictor, FinalPoseHead, TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer


class HPA(nn.Module):

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=4,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=0,
                 return_intermediate=True, mode='fixed', args=None
                 ):
        super().__init__()
        self.args = args

        self.mode = mode

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, mode=self.mode, args=args)
        
        encoder_norm = nn.LayerNorm(d_model) if normalize_before==1 else None
        if mode=='base':
            self.num_encoder_layers = 2
        else:
            self.num_encoder_layers = num_encoder_layers
        self.encoder = TransformerEncoder(encoder_layer, self.num_encoder_layers, encoder_norm,
                                          return_intermediate=return_intermediate, mode=self.mode, args=args)

        self.finalencoder = FinalPoseHead(14)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, cross, in_src, mask, gt_part_poses, base_pred, pos_embed, **kwargs):
        # B x P x C --> P x B x C
        src = in_src.permute(1, 0, 2)
        gt_part_poses = gt_part_poses.permute(1, 0, 2)
        lens = mask.sum(1)
        mask = ~mask.bool()  # B x P
        tgt = torch.zeros_like(src)

        num_part, batch_size, _ = src.size()

        # if self.args.ins_cat or self.args.ins_cat_in_encoder:
        part_ids = kwargs["part_ids"].long()
        if self.args.ins_version == "v1":
            # generate geometrical instance encode.
            match_h = part_ids.unsqueeze(2).repeat(1, 1, num_part)
            match_v = part_ids.unsqueeze(1).repeat(1, num_part, 1)
            match_codes = (match_h == match_v).float()  # [b, p, p]
            valid_mask = mask.unsqueeze(2).repeat(1, 1, num_part)
            match_codes[valid_mask] = 0.
            eye_mask = torch.eye(num_part).unsqueeze(0).repeat(batch_size, 1, 1).bool().to(part_ids.device)
            match_codes[eye_mask] = 0.
            # ins-inter encode.
            inter_codes = (part_ids - (~mask).long()).unsqueeze(2)
            if self.args.ins_cat_intra_only:
                ins_codes = match_codes.clone()
            elif self.args.ins_cat_inter_only:
                ins_codes = part_ids.new_zeros((batch_size, num_part, num_part)).scatter_(2, inter_codes, 1)
                ins_codes[valid_mask] = 0.
            else:
                # former is inter-code; later is intra-code when repeated.
                ins_codes = part_ids.new_zeros((batch_size, num_part, num_part * 2)).scatter_(2, inter_codes, 1)
                ins_codes[..., :num_part][valid_mask] = 0.
                ins_codes[..., num_part:] = match_codes
        elif self.args.ins_version == "v2":
            match_h = part_ids.unsqueeze(2).repeat(1, 1, num_part)
            match_v = part_ids.unsqueeze(1).repeat(1, num_part, 1)
            match_codes = (match_h == match_v).float()  # [b, p, p]
            valid_mask = mask.unsqueeze(2).repeat(1, 1, num_part)
            match_codes[valid_mask] = 0.
            cate_codes = torch.arange(num_part).unsqueeze(0).repeat(batch_size, 1).to(part_ids.device)
            if self.args.ins_cat_inter_only:
                ins_codes = part_ids.new_zeros((batch_size, num_part, num_part * 2)).scatter_(2, cate_codes.unsqueeze(2), 1)
                ins_codes[..., :num_part][valid_mask] = 0.
            elif self.args.ins_cat_intra_only:
                ins_codes = part_ids.new_zeros((batch_size, num_part, num_part * 2))
                ins_codes[..., num_part:] = match_codes
            else:
                ins_codes = part_ids.new_zeros((batch_size, num_part, num_part * 2)).scatter_(2, cate_codes.unsqueeze(2), 1)
                ins_codes = torch.flip(ins_codes, dims=[1])
                ins_codes[..., :num_part][valid_mask] = 0.
                ins_codes[..., num_part:] = match_codes
        else:
            raise NotImplementedError

        if num_part < self.args.max_num_part:
            num_res = self.args.max_num_part - num_part
            ins_codes_pad = ins_codes.new_zeros((batch_size, num_part, self.args.max_num_part * 2))
            ins_codes_pad[..., :num_part] = ins_codes[..., :num_part]
            ins_codes_pad[..., self.args.max_num_part:-num_res] = ins_codes[..., num_part:]
            ins_codes = ins_codes_pad.clone()
        ins_codes = ins_codes.transpose(0, 1).contiguous()

        class_codes = F.one_hot(part_ids, num_classes=20).transpose(0,1).contiguous() 
        idx_part_ids = part_ids.clone()
        idx_part_ids[idx_part_ids!=0] -= 1

        index = (idx_part_ids).unsqueeze(-1).repeat(1,1,in_src.shape[-1])
        index_preds = (idx_part_ids).unsqueeze(-1).repeat(1,1,7)
        base_src = torch.zeros_like(in_src)
        base_src = base_src.scatter_(dim=1, index=index, src=in_src)
        base_mask = ~(torch.any(base_src!=0, dim=-1))   

        init_pred = src.new_zeros((num_part, batch_size, 7))

        if self.mode=='base':
            preds, memory = self.encoder(match_codes, cross, base_src.permute(1,0,2), init_pred, class_codes, src_key_padding_mask=base_mask, pos=pos_embed, lens=lens, gt_part_poses=gt_part_poses, **kwargs)
            preds_clone = preds.clone()
            preds[-1] = torch.gather(preds_clone[-1].permute(1,0,2), 1, index_preds).permute(1,0,2)
            preds = preds.permute(2, 0, 1, 3)
        elif self.mode=='relative':
            preds, memory = self.encoder(match_codes, cross, src, base_pred, ins_codes, mask=None, src_key_padding_mask=mask, pos=pos_embed, lens=lens, gt_part_poses=gt_part_poses, **kwargs)
            preds = preds.permute(2, 0, 1, 3)
            
        final_preds = preds[:,:,:,:]
        return final_preds, memory

class EnDe(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=0,
                 return_intermediate=True, mode="decoder", args=None
                 ):
        super().__init__()
        self.args = args
        self.d_embed = 256
        self.mode = mode
        num_decoder_layers = num_encoder_layers

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, mode="relative", args=args)

        encoder_norm = nn.LayerNorm(d_model) if normalize_before==1 else None
        decoder_norm = nn.LayerNorm(d_model) if normalize_before==1 else None

        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm,
                                          return_intermediate=return_intermediate, mode='relative', args=args)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before, args=args)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                        return_intermediate=return_intermediate, args=args)

        self.finalencoder = FinalPoseHead(14)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def embed(self, tgt):
        p, b, _ = tgt.size()
        zeros = torch.zeros((p, b, 1)).to(tgt.device)
        embedding = tgt.repeat(1,1,85)
        return torch.cat((embedding, zeros),dim=-1)

    def forward(self, in_src, mask, gt_part_poses, base_pred, pos_embed, **kwargs):
        # B x P x C --> P x B x C
        src = in_src.permute(1, 0, 2)
        gt_part_poses = gt_part_poses.permute(1, 0, 2)
        lens = mask.sum(1)
        mask = ~mask.bool()  # B x P
        tgt = torch.zeros_like(src)

        num_part, batch_size, _ = src.size()

        if self.args.ins_cat or self.args.ins_cat_in_encoder:
            part_ids = kwargs["part_ids"].long()
            if self.args.ins_version == "v1":
                # generate geometrical instance encode.
                match_h = part_ids.unsqueeze(2).repeat(1, 1, num_part)
                match_v = part_ids.unsqueeze(1).repeat(1, num_part, 1)
                match_codes = (match_h == match_v).float()  # [b, p, p]
                valid_mask = mask.unsqueeze(2).repeat(1, 1, num_part)
                match_codes[valid_mask] = 0.
                eye_mask = torch.eye(num_part).unsqueeze(0).repeat(batch_size, 1, 1).bool().to(part_ids.device)
                match_codes[eye_mask] = 0.
                # ins-inter encode.
                inter_codes = (part_ids - (~mask).long()).unsqueeze(2)
                if self.args.ins_cat_intra_only:
                    ins_codes = match_codes.clone()
                elif self.args.ins_cat_inter_only:
                    ins_codes = part_ids.new_zeros((batch_size, num_part, num_part)).scatter_(2, inter_codes, 1)
                    ins_codes[valid_mask] = 0.
                else:
                    # former is inter-code; later is intra-code when repeated.
                    ins_codes = part_ids.new_zeros((batch_size, num_part, num_part * 2)).scatter_(2, inter_codes, 1)
                    ins_codes[..., :num_part][valid_mask] = 0.
                    ins_codes[..., num_part:] = match_codes
            elif self.args.ins_version == "v2":
                match_h = part_ids.unsqueeze(2).repeat(1, 1, num_part)
                match_v = part_ids.unsqueeze(1).repeat(1, num_part, 1)
                match_codes = (match_h == match_v).float()  # [b, p, p]
                valid_mask = mask.unsqueeze(2).repeat(1, 1, num_part)
                match_codes[valid_mask] = 0.
                cate_codes = torch.arange(num_part).unsqueeze(0).repeat(batch_size, 1).to(part_ids.device)
                if self.args.ins_cat_inter_only:
                    ins_codes = part_ids.new_zeros((batch_size, num_part, num_part * 2)).scatter_(2, cate_codes.unsqueeze(2), 1)
                    ins_codes[..., :num_part][valid_mask] = 0.
                elif self.args.ins_cat_intra_only:
                    ins_codes = part_ids.new_zeros((batch_size, num_part, num_part * 2))
                    ins_codes[..., num_part:] = match_codes
                else:
                    ins_codes = part_ids.new_zeros((batch_size, num_part, num_part * 2)).scatter_(2, cate_codes.unsqueeze(2), 1)
                    ins_codes[..., :num_part][valid_mask] = 0.
                    ins_codes[..., num_part:] = match_codes
            else:
                raise NotImplementedError

            if num_part < self.args.max_num_part:
                num_res = self.args.max_num_part - num_part
                ins_codes_pad = ins_codes.new_zeros((batch_size, num_part, self.args.max_num_part * 2))
                ins_codes_pad[..., :num_part] = ins_codes[..., :num_part]
                ins_codes_pad[..., self.args.max_num_part:-num_res] = ins_codes[..., num_part:]
                ins_codes = ins_codes_pad.clone()
            ins_codes = ins_codes.transpose(0, 1).contiguous()

        preds, memory = self.encoder(src, base_pred, ins_codes, mask=None, src_key_padding_mask=mask, pos=pos_embed, lens=lens, gt_part_poses=gt_part_poses, **kwargs)
        tgt_mask = torch.triu(torch.ones(num_part,num_part),diagonal=1).bool().to(part_ids.device)

        if self.training:
            generate_poses = torch.zeros_like(gt_part_poses[:,:,:3])
            generate_poses[1:,:,:] = gt_part_poses[:-1,:,:3]
            gt_part_embedding = self.embed(generate_poses)
            output, _ = self.decoder(gt_part_embedding, memory, ins_codes, tgt_mask=tgt_mask, memory_mask=None,
                                    tgt_key_padding_mask=mask,
                                    memory_key_padding_mask=mask)
        else:
            generate_poses = torch.zeros_like(gt_part_poses[:,:,:3])
            gt_part_embedding = self.embed(generate_poses)
            output, _ = self.decoder(gt_part_embedding, memory, ins_codes, tgt_mask=tgt_mask, memory_mask=None,
                                        tgt_key_padding_mask=mask,
                                        memory_key_padding_mask=mask)

        final_preds = torch.cat((output[:,:,:,0:3],preds[:,:,:,3:]),dim=-1)
        final_preds = final_preds[3:,:,:,:]
        return output.permute(2, 0, 1, 3), _


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def build_hpa(args, mode):
    return HPA(
        d_model=args.feat_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        normalize_before=args.pre_norm,
        return_intermediate=True,
        mode=mode,
        args=args
    )

def build_ende(args, mode):
    return EnDe(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        normalize_before=args.pre_norm,
        return_intermediate=True,
        mode=mode,
        args=args
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
