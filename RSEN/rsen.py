import torch
import torch.nn as nn
from einops import rearrange
import numbers

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



class FSAS(nn.Module):
    def __init__(self, dim, bias):
        super(FSAS, self).__init__()

        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')

        self.patch_size = 8

    def forward(self, x):
        hidden = self.to_hidden(x)

        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)

        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())

        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)

        out = self.norm(out)

        output = v * out
        output = self.project_out(output)

        return output

class TransformerBlock(nn.Module):
    def __init__(self, dim, bias=False, LayerNorm_type='WithBias'):
        super(TransformerBlock, self).__init__()


        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = FSAS(dim, bias)



    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        b,c,h,w = x.size()
        x = rearrange(x, ' b c h w -> b (h w) c', h=h, w=w)

        return x

class F_ext(nn.Module):
    def __init__(self, in_nc=3, nf=64):
        super(F_ext, self).__init__()
        stride = 1
        pad = 1
        self.pad = nn.ZeroPad2d(1)
        self.conv1 = nn.Conv2d(in_nc, nf, 3, stride, pad, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, stride, pad, bias=True)
        self.conv3 = nn.Conv2d(nf, nf, 3, stride, pad, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):

        conv1_out = self.act(self.conv1(x))
        conv2_out = self.act(self.conv2(conv1_out))
        conv3_out = self.act(self.conv3(conv2_out))
        out = torch.mean(conv3_out, dim=[2, 3], keepdim=False)

        return out


class SDNet(nn.Module):
    def __init__(self, dim, bias):
        super(SDNet, self).__init__()
        self.dim = dim
        self.F_ext_net = F_ext(in_nc=4, nf=64)
        self.Conv_0 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,  dilation=1, bias=True)
        "1"
        self.Conv_1 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_2 = nn.ReLU(inplace=True)
        self.Conv_3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_5 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_6 = nn.ReLU(inplace=True)
        self.Conv_7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_9 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_10 = nn.ReLU(inplace=True)
        self.Conv_11 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_13 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_14 = nn.ReLU(inplace=True)
        self.Conv_15 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)



        self.Conv_17 = torch.nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=0, dilation=1, bias=True)

        "2"
        self.Conv_18 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_19 = nn.ReLU(inplace=True)
        self.Conv_20 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_22 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_23 = nn.ReLU(inplace=True)
        self.Conv_24 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_26 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_27 = nn.ReLU(inplace=True)
        self.Conv_28 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)


        self.Conv_30 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_31 = nn.ReLU(inplace=True)
        self.Conv_32 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)


        self.Conv_34 = torch.nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=0, dilation=1, bias=True)
        self.after_conv1 = TransformerBlock(dim=int(dim), bias=bias)
        self.prompt_scale0 = nn.Linear(64, self.dim, bias=True)


        # self.prompt_scale2 = nn.Linear(64, int(dim * 2 ** 3), bias=True)
        "3"
        self.Conv_35 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_36 = nn.ReLU(inplace=True)
        self.Conv_37 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_39 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_40 = nn.ReLU(inplace=True)
        self.Conv_41 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_43 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_44 = nn.ReLU(inplace=True)
        self.Conv_45 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_47 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_48 = nn.ReLU(inplace=True)
        self.Conv_49 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)


        self.Conv_51 = torch.nn.Conv2d(256, 512,  kernel_size=2, stride=2, padding=0, dilation=1, bias=True)
        self.after_conv2 = TransformerBlock(dim=int(dim * 2), bias=bias)
        self.prompt_scale1 = nn.Linear(64, self.dim * 2, bias=True)
        "4"
        self.Conv_52 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_53 = nn.ReLU(inplace=True)
        self.Conv_54 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_56 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_57 = nn.ReLU(inplace=True)
        self.Conv_58 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_60 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_61 = nn.ReLU(inplace=True)
        self.Conv_62 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_64 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_65 = nn.ReLU(inplace=True)
        self.Conv_66 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        '5'
        self.ConvTranspose_69 =torch.nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0, dilation=1, bias=True)
        self.Conv_70 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_71 = nn.ReLU(inplace=True)
        self.Conv_72 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_74 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_75 = nn.ReLU(inplace=True)
        self.Conv_76 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_78 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_79 = nn.ReLU(inplace=True)
        self.Conv_80 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_82 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_83 = nn.ReLU(inplace=True)
        self.Conv_84 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)


        '6'
        self.ConvTranspose_87 = torch.nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, dilation=1,bias=True)
        self.Conv_88 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_89 = nn.ReLU(inplace=True)
        self.Conv_90 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_92 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_93 = nn.ReLU(inplace=True)
        self.Conv_94 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_96 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_97 = nn.ReLU(inplace=True)
        self.Conv_98 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_100 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_101 = nn.ReLU(inplace=True)
        self.Conv_102 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)


        '7'
        self.ConvTranspose_105 = torch.nn.ConvTranspose2d(128,64, kernel_size=2, stride=2, padding=0, dilation=1, bias=True)
        self.Conv_106 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_107 = nn.ReLU(inplace=True)
        self.Conv_108 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_110 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_111 = nn.ReLU(inplace=True)
        self.Conv_112 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_114 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_115 = nn.ReLU(inplace=True)
        self.Conv_116 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_118 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.ReLU_119 = nn.ReLU(inplace=True)
        self.Conv_120 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)

        self.Conv_123 = torch.nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)


    def forward(self, input,mask):
        x = input
        xi = torch.cat((mask , input), dim = 1)
        prompt = self.F_ext_net(xi)
        scale0 = self.prompt_scale0(prompt)
        scale1 = self.prompt_scale1(prompt)

        "1"
        temp4 = self.Conv_3(self.ReLU_2(self.Conv_1(self.Conv_0(x))))
        Add_4 = temp4 + self.Conv_0(x)
        temp8 = self.Conv_7(self.ReLU_6(self.Conv_5(Add_4)))
        Add_8 = temp8 + Add_4
        temp12 = self.Conv_11(self.ReLU_10(self.Conv_9(Add_8)))
        Add_12 = temp12  + Add_8
        temp16 = self.Conv_15(self.ReLU_14(self.Conv_13(Add_12)))
        Add_16 = temp16 + Add_12

        "2"
        temp21 = self.Conv_20(self.ReLU_19(self.Conv_18(self.Conv_17(Add_16))))
        Add_21 = temp21 + self.Conv_17(Add_16)
        temp25 = self.Conv_24(self.ReLU_23(self.Conv_22(Add_21)))
        Add_25 = temp25 + Add_21
        temp29 = self.Conv_28(self.ReLU_27(self.Conv_26(Add_25)))
        Add_29 = temp29 + Add_25
        temp33 = self.Conv_32(self.ReLU_31(self.Conv_30(Add_29)))
        Add_33 = temp33 + Add_29

        "3"

        prompt1 = self.after_conv1(self.Conv_34(Add_33)) * scale0.view(-1, 1, self.dim) + scale0.view(-1, 1, self.dim) + self.after_conv1(self.Conv_34(Add_33))
        h1, w1 = self.Conv_34(Add_33).shape[-2:]
        prompt1 = rearrange(prompt1, 'b (h w) c -> b c h w', h=h1, w=w1)
        temp38 = self.Conv_37(self.ReLU_36(self.Conv_35(prompt1)))
        Add_38 = temp38 + prompt1
        temp42 = self.Conv_41(self.ReLU_40(self.Conv_39(Add_38)))
        Add_42 = temp42 + Add_38
        temp46 = self.Conv_45(self.ReLU_44(self.Conv_43(Add_42)))
        Add_46 = temp46 + Add_42
        temp50 = self.Conv_49(self.ReLU_48(self.Conv_47(Add_46)))
        Add_50 = temp50 + Add_46

        "4"

        prompt2 = self.after_conv2(self.Conv_51(Add_50)) * scale1.view(-1, 1, self.dim *  2)  + scale1.view(-1, 1, self.dim * 2) + self.after_conv2(self.Conv_51(Add_50))
        h2, w2 = self.Conv_51(Add_50).shape[-2:]
        prompt2 = rearrange(prompt2, 'b (h w) c -> b c h w', h=h2, w=w2)
        temp55 = self.Conv_54(self.ReLU_53(self.Conv_52(prompt2)))
        Add_55 = temp55 + prompt2
        temp59 = self.Conv_58(self.ReLU_57(self.Conv_56(Add_55)))
        Add_59 = temp59 + Add_55
        temp63 = self.Conv_62(self.ReLU_61(self.Conv_60(Add_59)))
        Add_63 = temp63 + Add_59
        temp67 = self.Conv_66(self.ReLU_65(self.Conv_64(Add_63)))
        Add_67 = temp67 + Add_63


        Add_68 = Add_67 + prompt2

        "5"
        temp73 = self.Conv_72(self.ReLU_71(self.Conv_70(self.ConvTranspose_69(Add_68))))
        Add_73 = temp73 + self.ConvTranspose_69(Add_68)
        temp77 = self.Conv_76(self.ReLU_75(self.Conv_74(Add_73)))
        Add_77 = temp77 + Add_73
        temp81 = self.Conv_80(self.ReLU_79(self.Conv_78(Add_77)))
        Add_81 = temp81 + Add_77
        temp85 = self.Conv_84(self.ReLU_83(self.Conv_82(Add_81)))
        Add_85 = temp85 + Add_81

        Add_86 = Add_85 + prompt1

        "6"
        temp91 = self.Conv_90(self.ReLU_89(self.Conv_88(self.ConvTranspose_87(Add_86))))
        Add_91 = temp91 + self.ConvTranspose_87(Add_86)
        temp95 = self.Conv_94(self.ReLU_93(self.Conv_92(Add_91)))
        Add_95 = temp95 + Add_91
        temp99 = self.Conv_98(self.ReLU_97(self.Conv_96(Add_95)))
        Add_99 = temp99 + Add_95
        temp103 = self.Conv_102(self.ReLU_101(self.Conv_100(Add_99)))
        Add_103 = temp103 + Add_99

        Add_104 = Add_103 + self.Conv_17(Add_16)

        "7"
        temp109 = self.Conv_108(self.ReLU_107(self.Conv_106(self.ConvTranspose_105(Add_104))))
        Add_109 = temp109 + self.ConvTranspose_105(Add_104)
        temp113 = self.Conv_112(self.ReLU_111(self.Conv_110(Add_109)))
        Add_113 = temp113 + Add_109
        temp117 = self.Conv_116(self.ReLU_115(self.Conv_114(Add_113)))
        Add_117 = temp117 + Add_113
        temp121 = self.Conv_120(self.ReLU_119(self.Conv_118(Add_117)))
        Add_121 = temp121 + Add_117


        Add_122 = Add_121 + self.Conv_0(x)
        out = self.Conv_123(Add_122)

        return out

