import torch
import torch.nn as nn
import torch.nn.functional as F


class CALayer(nn.Module):
    def __init__(self, channel, reduction):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),nn.ReLU(inplace=True), nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),nn.Sigmoid())
        self.process = nn.Sequential(nn.Conv2d(channel, channel, 3, stride=1, padding=1),nn.ReLU(), nn.Conv2d(channel, channel, 3, stride=1, padding=1))

    def forward(self, x):
        y = self.process(x)
        y = self.avg_pool(y)
        z = self.conv_du(y)
        return z * y + x

class Refine(nn.Module):
    def __init__(self,n_feat,out_channel):
        super(Refine, self).__init__()
        self.conv_in = nn.Conv2d(n_feat, n_feat, 3, stride=1, padding=1)
        self.process = nn.Sequential(CALayer(n_feat,4))
        self.conv_last = nn.Conv2d(in_channels=n_feat, out_channels=out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):

        out = self.conv_in(x)
        out = self.process(out)
        out = self.conv_last(out)

        return out


class NonLocalBlock(nn.Module):
    def __init__(self, channel):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.LeakyReLU = nn.LeakyReLU(0.1, inplace=False)
        self.conv_transfer = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, c, -1)
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0,2,1).contiguous().view(b,self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)

        # global part，leaky relu
        global1 = self.LeakyReLU(mask)    # (4,32,128,128)
        global2 = self.conv_transfer(global1)  # (4,32,128,128)
        global3 = self.LeakyReLU(global2)

        out = global3 + x

        return out


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups
        self.pre0 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.pre1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.pre2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.panF_pha_conv = nn.Conv2d(in_channels=in_channels*2, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.amp_fuse = nn.Sequential(nn.Conv2d(in_channels=in_channels*2, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.LeakyReLU(0.1, inplace=False),nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        self.pha_fuse = nn.Sequential(nn.Conv2d(in_channels=in_channels*3, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False), nn.LeakyReLU(0.1, inplace=False),nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False))
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2, out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):

        batch, c, h, w = x.size()
        r_size = x.size()

        # FFT and conv1*1
        _, _, H, W = x.shape
        xF = torch.fft.rfft2(self.pre0(x) + 1e-8, norm='backward')

        xF_amp = torch.abs(xF)
        xF_pha = torch.angle(xF)

        # synthesize the phase and amplitude 
        amp_fuse = xF_amp
        pha_fuse = xF_pha

        real = amp_fuse * torch.cos(pha_fuse) + 1e-8
        imag = amp_fuse * torch.sin(pha_fuse) + 1e-8
        out = torch.complex(real, imag) + 1e-8
        output = torch.abs(torch.fft.irfft2(out, s=(H, W), norm='backward'))

        return output

class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv_pan = nn.Conv2d(in_channels*2, out_channels // 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_ms = nn.Conv2d(in_channels*2, out_channels // 2, kernel_size=1, stride=1, padding=0, bias=False)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)  # out_channels=32, out_channels // 2 = 16
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s_h = h // split_no
            split_s_w = w // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s_h, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s_w, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output


# FFC
class FFC(nn.Module):

    def __init__(self, in_channels, out_channels,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride
        #self.conv1 = nn.Conv2d(5, 16, kernel_size=3, stride=1, padding=1, bias=False)

        in_cg = int(in_channels * ratio_gin)  # 32
        in_cl = in_channels - in_cg   # 32
        out_cg = int(out_channels * ratio_gout)   # 32
        out_cl = out_channels - out_cg   # 32

        self.in_channels = in_channels

        self.ratio_gin = ratio_gin   # 0.5
        self.ratio_gout = ratio_gout   # 0.5

        module1 = nn.Conv2d
        kernel_size1 = 3
        padding1 = 1
        self.convl2l = module1(in_cl, out_cl, kernel_size1,
                              stride, padding1, dilation, groups, bias)
        module2 = nn.Conv2d
        kernel_size2 = 1
        self.convl2g = module2(in_cl, out_cg, kernel_size2,
                              stride, padding, dilation, groups, bias)
        self.non_local = NonLocalBlock(channel=in_cl)

        module3 = nn.Conv2d
        kernel_size3 = 3
        padding3 = 1
        self.convg2l = module3(in_cg, out_cl, kernel_size3,
                              stride, padding3, dilation, groups, bias)

        # SpectralTransform
        module4 = SpectralTransform
        self.convg2g = module4(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu)

    def forward(self, x):

        x_l = x[:, :self.in_channels//2, :, :]
        x_g = x[:, self.in_channels//2:, :, :]

        # local to local
        l2l_out = self.convl2l(x_l)
        self.act_l2l = nn.LeakyReLU(0.1, inplace=False)
        l2l_out = self.act_l2l(l2l_out)    # (4,32,128,128)

        # local to global
        l2g_out = self.convl2g(x_l)   # (4,32,128,128)
        l2g_out = self.non_local(l2g_out)

        # global to local
        g2l_out = self.convg2l(x_g)
        self.act_g2l = nn.LeakyReLU(0.1, inplace=False)
        g2l_out = self.act_g2l(g2l_out)    # (4,32,128,128)

        # global to global --- spectrum
        g2g_out = self.convg2g(x_g)

        return l2l_out, l2g_out, g2l_out, g2g_out