# -*- encoding: utf-8 -*-
'''
@File  :  sol_model1.py
@Time  :  2023/11/24 15:46:57
@Author :  Jiuchuang Yuan 
@Contact :  jiuchuang.yuan@xtalpi.com
'''

import torch
from torch import nn
import torch.nn.functional as F

# The code for the graphgps_layer is sourced from the GraphGPS repository. https://github.com/rampasek/GraphGPS/tree/main/graphgps
from .graphgps_layer.gps_layer import GPSLayer

class SinglePropKANModel(nn.Module): 
    def __init__(self, 
                node_features_dim=65,
                edge_features_dim=13,
                core_layer_num=6,
                head_out_dim=64, 
                kan_in_dim=8,
                net_heads=4,
                dropout=0.5
                ):
        super().__init__()
        from kan import KAN

        self.automatic_optimization= False
        self.y_scale = False
        self.node_embedding = nn.Linear(node_features_dim, head_out_dim)
        self.edge_embedding = nn.Linear(edge_features_dim, head_out_dim)

        layers = []
        for _ in range(core_layer_num):
            layers.append(GPSLayer(head_out_dim, 'CustomGatedGCN', 'Transformer', net_heads, pna_degrees=None, equivstable_pe=False, dropout=dropout,
                     attn_dropout=dropout, layer_norm=False, batch_norm=True,bigbird_cfg=None))
        self.scale_fc = nn.Linear(head_out_dim, kan_in_dim)
        self.core_layers = torch.nn.Sequential(*layers)

        self.out_kan = KAN(width=[kan_in_dim,5,1], grid=3, k=3, seed=42)
        
    def forward(self, g_mol, return_fea=False):
        g_mol.x = torch.tanh(self.node_embedding(g_mol.x))
        g_mol.edge_attr = torch.tanh(self.edge_embedding(g_mol.edge_attr))
        
        g_mol = self.core_layers(g_mol)
        g_mol.x = self.scale_fc(g_mol.x)
        fea_mol = self.pooling(g_mol)
    
        c = fea_mol.clone()
        c = self.out_kan(c)
        out = F.softplus(c.squeeze(1))

        if return_fea:
            return out, fea_mol
        else:
            return out

    def pooling(self, g):
        if hasattr(g, 'mask'):
            summed_fea = [torch.sum(g.x[torch.where(g.batch == case_idx)] * g.mask[torch.where(g.batch == case_idx)].unsqueeze(1), dim=0, keepdim=True) for case_idx in g.batch.unique()]
        else:
            summed_fea = [torch.sum(g.x[torch.where(g.batch == case_idx)], dim=0, keepdim=True) for case_idx in g.batch.unique()]
        return torch.cat(summed_fea, dim=0)

