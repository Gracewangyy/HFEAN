import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import make_coord
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)


class LIIF(nn.Module):

    def __init__(self, in_features, local_ensemble=True, feat_unfold=True, cell_decode=True):
        super().__init__()

        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        self.imnet = MLP(in_dim=in_features * 9 + 2 + 2, hidden_list=[256, 256, 256, 256, 256], out_dim=in_features)

    def gen_feat(self, feat):
        self.feat = feat
        return self.feat

    def query_rgb(self, coord, cell=None):  # coord: hr_coord
        feat = self.feat  # from RDN

        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                                mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2  # 1 / H
        ry = 2 / feat.shape[-1] / 2  # 1 / W

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda().permute(2, 0, 1).unsqueeze(0).expand(
            feat.shape[0], 2, *feat.shape[-2:]).to(device)

        preds = []
        areas = []
        for vx in vx_lst:  # neighbor points: [-1, -1], [-1, 1], [1, -1], [1, 1]
            for vy in vy_lst:
                coord_ = coord.clone().to(device)  # [1, 48*48, 2]

                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(  # .unfold
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)  # [1, 48*48, 3*9]

                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)  # [1, 48*48, 2]

                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]  # * 48
                rel_coord[:, :, 1] *= feat.shape[-1]  # * 48

                inp = torch.cat([q_feat, rel_coord], dim=-1)  # [1, 48*48, 3*9+2]


                if self.cell_decode:
                    rel_cell = cell.clone().to(device)
                    rel_cell[:, :, 0] *= feat.shape[-2]  # [-1, 1] / crop_hr_h * 48
                    rel_cell[:, :, 1] *= feat.shape[-1]  # [-1, 1] / crop_hr_w * 48
                    inp = torch.cat([inp, rel_cell], dim=-1)  # inp (16,48*48,580)

                bs, q = coord.shape[:2]
                inp = inp.view(bs * q, -1)
                pred = self.imnet(inp).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)  # area weight: s/S
        if self.local_ensemble:
            t = areas[0]
            areas[0] = areas[3]
            areas[3] = t
            t = areas[1]
            areas[1] = areas[2]
            areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)

        return ret

    def forward(self, feat, coord, cell):
        self.gen_feat(feat)
        ret = self.query_rgb(coord, cell)

        return ret
