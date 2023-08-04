import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from .INA import *
from .encoder_hornet import hornet_spa
from .FEM import *

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret.to(device)


def Gaussian_filter(tensor, kernel_size=5, sigma=0):
 
    if kernel_size % 2 == 0 or kernel_size < 3:
        raise ValueError("Kernel size must be a positive odd integer greater than or equal to 3.")

    if sigma == 0:
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8

    # convert tensor to numpy
    numpy_array = tensor.detach().cpu().numpy()

    # Gaussian_filter
    for i in range(numpy_array.shape[0]):  # iterate over the batch size
        image = numpy_array[i, 0, :, :]  # Retrieve current image and compress the channel

        # Gaussian_blur
        blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

        # filtered result to the original array
        numpy_array[i, 0, :, :] = blurred_image

    # convert numpy to tensor
    blurred_tensor = torch.from_numpy(numpy_array)
    blur_lp = torch.tensor(blurred_tensor).to(device)
    blur_hp = tensor - blur_lp

    return blur_hp


class FFCNET(nn.Module):
    def __init__(self, config1, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.config1 = config1
        spectral_num = config1[config1["train_dataset"]]["spectral_bands"]
        scale_factor =  config1[config1["train_dataset"]]["factor"]
        self.spectral_num = spectral_num

        # Pre-align Module
        self.conv_PS = nn.Conv2d(4, 16*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.PS = nn.PixelShuffle(4)

        dim = 64
        out_dim_spa = 16
        self.hornet_feat0 = hornet_spa(in_chans=spectral_num + 1, base_dim=out_dim_spa, depths=[2, 3, 3, 2])
        self.hornet_feat1 = hornet_spa(in_chans=spectral_num + 1, base_dim=out_dim_spa, depths=[2, 3, 3, 2])
        self.hornet_feat2 = hornet_spa(in_chans=spectral_num + 1, base_dim=out_dim_spa, depths=[2, 3, 3, 2])

        # FEM split channels
        self.split_01 = FFC(in_channels=16, out_channels=16, ratio_gin=0.5, ratio_gout=0.5)
        self.split_02 = FFC(in_channels=16, out_channels=16, ratio_gin=0.5, ratio_gout=0.5)
        self.split_03 = FFC(in_channels=16, out_channels=16, ratio_gin=0.5, ratio_gout=0.5)
        self.split_11 = FFC(in_channels=16, out_channels=16, ratio_gin=0.5, ratio_gout=0.5)
        self.split_12 = FFC(in_channels=16, out_channels=16, ratio_gin=0.5, ratio_gout=0.5)
        self.split_13 = FFC(in_channels=16, out_channels=16, ratio_gin=0.5, ratio_gout=0.5)
        self.split_21 = FFC(in_channels=16, out_channels=16, ratio_gin=0.5, ratio_gout=0.5)
        self.split_22 = FFC(in_channels=16, out_channels=16, ratio_gin=0.5, ratio_gout=0.5)
        self.split_23 = FFC(in_channels=16, out_channels=16, ratio_gin=0.5, ratio_gout=0.5)

        # RCAB attention
        self.FFC_RCAB_0 = Refine(n_feat=16, out_channel=16)
        self.FFC_RCAB_1 = Refine(n_feat=16, out_channel=16)
        self.FFC_RCAB_2 = Refine(n_feat=16, out_channel=16)
        self.FFC_RCAB_01 = Refine(n_feat=16, out_channel=16)
        self.FFC_RCAB_02 = Refine(n_feat=16, out_channel=16)
        self.FFC_RCAB_03 = Refine(n_feat=16, out_channel=16)
        self.FFC_RCAB_11 = Refine(n_feat=16, out_channel=16)
        self.FFC_RCAB_12 = Refine(n_feat=16, out_channel=16)
        self.FFC_RCAB_13 = Refine(n_feat=16, out_channel=16)
        self.FFC_RCAB_21 = Refine(n_feat=16, out_channel=16)
        self.FFC_RCAB_22 = Refine(n_feat=16, out_channel=16)
        self.FFC_RCAB_23 = Refine(n_feat=16, out_channel=16)

        self.conv0 = nn.Conv2d(48, 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1 = nn.Conv2d(48, 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(48, 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_hp_final = nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1, bias=False)

        # INA Module
        self.conv_liif = nn.Conv2d(48, 16, kernel_size=1, stride=1, padding=0, bias=False)
        self.liif = LIIF(in_features=16, local_ensemble=True, feat_unfold=True, cell_decode=True)
        

    def forward(self, lrms,pan ):

        B, C, H, side_len = pan.shape
        coord = make_coord(shape=(side_len, side_len), ranges=None, flatten=True).unsqueeze(0)

        coord = coord.repeat(B, 1, 1)
        cell = torch.ones_like(coord)

        cell[:, :, 0] *= 2 / side_len
        cell[:, :, 1] *= 2 / side_len

        # Preliminary Alignment Module (Pre-align)
        lrms_conv_result = self.conv_PS(lrms)
        lrms_up = self.PS(lrms_conv_result)

        # Gaussian filters, with kernel sizes of 5, 27, 41
        pan_hp0 = Gaussian_filter(pan, kernel_size=5, sigma=1.5)
        ms_hp0 = Gaussian_filter(lrms_up, kernel_size=5, sigma=1.5)
        pan_hp1 = Gaussian_filter(pan, kernel_size=27, sigma=2)
        ms_hp1 = Gaussian_filter(lrms_up, kernel_size=27, sigma=2)
        pan_hp2 = Gaussian_filter(pan, kernel_size=41, sigma=2.8)
        ms_hp2 = Gaussian_filter(lrms_up, kernel_size=41, sigma=2.8)

        feat_0 = [pan_hp0, ms_hp0]
        feat_0 = torch.cat(feat_0, dim=1)
        feat_0 = self.hornet_feat0(feat_0)

        feat_1 = [pan_hp1, ms_hp1]
        feat_1 = torch.cat(feat_1, dim=1)
        feat_1 = self.hornet_feat1(feat_1)

        feat_2 = [pan_hp2, ms_hp2]
        feat_2 = torch.cat(feat_2, dim=1)
        feat_2 = self.hornet_feat2(feat_2)

        # RCAB attention
        feat_RCAB_0 = self.FFC_RCAB_0(feat_0)
        feat_RCAB_1 = self.FFC_RCAB_1(feat_1)
        feat_RCAB_2 = self.FFC_RCAB_2(feat_2)

        # Feature Enhancement Module (FEM)
        # first branch FEM_00
        l2l_out00, l2g_out00, g2l_out00, g2g_out00 = self.split_01(feat_RCAB_0)  
        global_feat_00 = l2g_out00 + g2g_out00
        local_feat_00 = l2l_out00 + g2l_out00
        feat_cat_00 = torch.cat((global_feat_00, local_feat_00), dim=1)
        # RCAB attention
        feat_RCAB_01 = self.FFC_RCAB_01(feat_cat_00)

        # first branch FEM_01
        l2l_out01, l2g_out01, g2l_out01, g2g_out01 = self.split_02(feat_RCAB_01)
        global_feat_01 = l2g_out01 + g2g_out01
        local_feat_01 = l2l_out01 + g2l_out01
        feat_cat_01 = torch.cat((global_feat_01, local_feat_01), dim=1)
        # RCAB attention
        feat_RCAB_02 = self.FFC_RCAB_02(feat_cat_01)

        # res concat
        feat_all_0 = [feat_RCAB_0, feat_RCAB_01, feat_RCAB_02]
        feat_all_0 = torch.cat(feat_all_0, dim=1)
        feat_all_0 = self.conv0(feat_all_0)

        # second branch FEM_10
        l2l_out10, l2g_out10, g2l_out10, g2g_out10 = self.split_11(feat_RCAB_1)
        global_feat_10 = l2g_out10 + g2g_out10
        local_feat_10 = l2l_out10 + g2l_out10
        feat_cat_10 = torch.cat((global_feat_10, local_feat_10), dim=1)
        # RCAB attention
        feat_RCAB_11 = self.FFC_RCAB_11(feat_cat_10)

        # second branch FEM_11
        l2l_out11, l2g_out11, g2l_out11, g2g_out11 = self.split_12(feat_RCAB_11)
        global_feat_11 = l2g_out11 + g2g_out11
        local_feat_11 = l2l_out11 + g2l_out11
        feat_cat_11 = torch.cat((global_feat_11, local_feat_11), dim=1)
        # RCAB attention
        feat_RCAB_12 = self.FFC_RCAB_12(feat_cat_11)

        # res concat
        feat_all_1 = [feat_RCAB_1, feat_RCAB_11, feat_RCAB_12]
        feat_all_1 = torch.cat(feat_all_1, dim=1)
        feat_all_1 = self.conv1(feat_all_1)

        # third branch FEM_20
        l2l_out20, l2g_out20, g2l_out20, g2g_out20 = self.split_21(feat_RCAB_2) 
        global_feat_20 = l2g_out20 + g2g_out20
        local_feat_20 = l2l_out20 + g2l_out20
        feat_cat_20 = torch.cat((global_feat_20, local_feat_20), dim=1)
        # RCAB attention
        feat_RCAB_21 = self.FFC_RCAB_21(feat_cat_20)

        # third branch FEM_21
        l2l_out21, l2g_out21, g2l_out21, g2g_out21 = self.split_22(feat_RCAB_21)
        global_feat_21 = l2g_out21 + g2g_out21
        local_feat_21 = l2l_out21 + g2l_out21
        feat_cat_21 = torch.cat((global_feat_21, local_feat_21), dim=1)
        # RCAB attention
        feat_RCAB_22 = self.FFC_RCAB_22(feat_cat_21)

        # res concat
        feat_all_2 = [feat_RCAB_2, feat_RCAB_21, feat_RCAB_22]
        feat_all_2 = torch.cat(feat_all_2, dim=1)
        feat_all_2 = self.conv2(feat_all_2)

        feat_liif = [feat_all_0, feat_all_1, feat_all_2]
        feat_liif = torch.cat(feat_liif, dim=1)

        # Implicit Neural Alignment Mdule (INA)
        feat_all = self.conv_liif(feat_liif)
        feat_out = self.liif(feat_all, coord=coord, cell=cell)
        feat_out1 = feat_out.permute(0, 2, 1).view(feat_out.shape[0], -1, side_len, side_len)
        feat_hp = self.conv_hp_final(feat_out1)

        # HMS results
        output = feat_hp + lrms_up

        return {"pred": output}


