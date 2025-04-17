import torch.nn as nn
import math
from .Swin import Swintransformer
import cv2
import os
import torch
import torch.nn.functional as F
from .Res2Net_v1b import res2net50_v1b_26w_4s
from torch.nn.parameter import Parameter

Act = nn.ReLU
def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
            if m.weight is None:
                pass
            elif m.bias is not None:
                nn.init.zeros_(m.bias)
            else:
                nn.init.ones_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.ReLU6, nn.Upsample, Parameter, nn.AdaptiveAvgPool2d, nn.Sigmoid,Swintransformer,nn.PReLU)):
            pass
        elif isinstance(m, (RFEfussion,ReverseStage,GRA,CAB,CALayer,RFM1,SpatialAttentionModule,CFR,NeighborConnectionDecoder)):
            pass
        else:
            m.initialize()
            # pass

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

    def initialize(self):
        weight_init(self)

def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
    
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x
        return res
    
class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttentionModule, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 使用输入特征的最大池化和平均池化特征
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)

        x = self.conv1(x)
        return self.sigmoid(x)

class SAM(nn.Module):
    def __init__(self, ch_in=32, reduction=16):
        super(SAM, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.spatial_attention = SpatialAttentionModule() # 空间注意力
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )
        self.fc_wight = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, 1, bias=False),
            nn.Sigmoid()
        )
        
    def initialize(self):
        weight_init(self)


    def forward(self, x_h, x_l):
        #print('x_h shape, x_l shape,',x_h.shape, x_l.shape)
        b, c, _, _ = x_h.size()
        #print('self.avg_pool(x_h)',self.avg_pool(x_h).shape)
        y_h = self.avg_pool(x_h).view(b, c) # squeeze操作
        #print('***this is Y-h shape',y_h.shape)
        h_weight=self.fc_wight(y_h)
        #print('h_weight',h_weight.shape,h_weight) ##(batch,1)
        y_h = self.fc(y_h).view(b, c, 1, 1) # FC获取通道注意力权重，是具有全局信息的
        #print('y_h.expand_as(x_h)',y_h.expand_as(x_h).shape)
        x_fusion_h=x_h * y_h.expand_as(x_h)
        x_fusion_h=torch.mul(x_fusion_h, h_weight.view(b, 1, 1, 1))
##################----------------------------------
        b, c, _, _ = x_l.size()
        y_l = self.avg_pool(x_l).view(b, c) # squeeze操作
        l_weight = self.fc_wight(y_l)
        #print('l_weight',l_weight.shape,l_weight)
        y_l = self.fc(y_l).view(b, c, 1, 1) # FC获取通道注意力权重，是具有全局信息的
        #print('***this is y_l shape', y_l.shape)
        x_fusion_l=x_l * y_l.expand_as(x_l)
        x_fusion_l = torch.mul(x_fusion_l, l_weight.view(b, 1, 1, 1))
#################-------------------------------
        #print('x_fusion_h shape, x_fusion_l shape,h_weight shape',x_fusion_h.shape,x_fusion_l.shape,h_weight.shape)
        x_fusion=x_fusion_h+x_fusion_l

        # 应用空间注意力 
        #————————————————————————新增
        spatial_attention = self.spatial_attention(x_fusion)
        x_fusion = x_fusion * spatial_attention

        return x_fusion # 注意力作用每一个通道上


def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area  
    return supp_feat

class RFEfussion(nn.Module):
    '''
    先验融合模块 
    '''
    def __init__(self,chanel=3):
        super(RFEfussion,self).__init__()
        self.chanel=chanel

        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.alpha_conv=nn.Sequential(
                                    nn.Conv2d(7, 3, kernel_size=1, stride=1, padding=0, bias=False),
                                    nn.ReLU())
        self.beta_conv=nn.Sequential(
                nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            )
        # self.alpha1= 0.5
        # self.beta1= 0.3
        # self.gamma = 0.1

  
    def forward(self,img,mask):
        x=img.size(2)
        y=img.size(3)
        supp_feat=Weighted_GAP(img,mask)
        # print(supp_feat.shape)  shape (1, 3, 1, 1)
        # feat1=img+mask
        supp_feat_bin = supp_feat.expand(-1, -1, x, y)
        merge_feat_bin = torch.cat([img, supp_feat_bin, mask], 1)
        # print(merge_feat_bin.shape) shape(1, 9, 2592, 1944)
        merge_feat_bin=self.alpha_conv(merge_feat_bin)+img
        # print(merge_feat_bin.shape)
        merge_feat_bin = self.beta_conv(merge_feat_bin) + merge_feat_bin


        return merge_feat_bin

class RFM1(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFM1, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1) , dilation=1),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0), dilation=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, (1, 3), padding=((0, 2) ), dilation=2),
            BasicConv2d(out_channel, out_channel, (3, 1), padding=((2, 0) ), dilation=2)
        )

        self.branch3 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, (1, 3), padding=((0, 3) ), dilation=3),
            BasicConv2d(out_channel, out_channel, (3, 1), padding=((3, 0) ), dilation=3)
        )

        self.branch4 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, (1, 3), padding=((0, 4) ), dilation=4),
            BasicConv2d(out_channel, out_channel, (3, 1), padding=((4, 0) ), dilation=4)
        )

        self.conv = nn.Conv2d(in_channel, out_channel, 1)

        self.conv_cat = nn.Conv2d(out_channel*4, out_channel, 3, padding=1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(self.conv(x) + x1)
        x3 = self.branch3(self.conv(x) + x2)
        x4 = self.branch4(self.conv(x) + x3)
        x_cat = self.conv_cat(torch.cat((x0, x2, x3, x4), dim=1))

        x = self.relu(x1 + x_cat)
        return x

    def initialize(self):
        weight_init(self)

class DB2(nn.Module):
    def __init__(self,inplanes,outplanes):
        super(DB2,self).__init__()
        self.short_cut = nn.Conv2d(outplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Sequential(
            nn.Conv2d(inplanes+outplanes,outplanes,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(outplanes,outplanes,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(outplanes,outplanes,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(outplanes,outplanes,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )

    def forward(self,x,z): 
        z = F.interpolate(z,size=x.size()[2:],mode='bilinear',align_corners=True)
        p = self.conv(torch.cat((x,z),1))
        sc = self.short_cut(z)
        p  = p+sc
        p2 = self.conv2(p)
        p  = p+p2
        return p,p
    
    def initialize(self):
        weight_init(self)

class MTD1(nn.Module):
    def __init__(self,inplanes,outplanes):
        super(MTD1,self).__init__()
        self.squeeze1 = nn.Sequential(  
                    nn.Conv2d(inplanes, outplanes,kernel_size=1,stride=1,padding=0), 
                    nn.BatchNorm2d(64), 
                    nn.ReLU(inplace=True)
                )
        self.squeeze2 = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3,stride=1,dilation=2,padding=2), 
                nn.BatchNorm2d(64), 
                nn.ReLU(inplace=True)
                )
        #self.d2 = DB2(512,64)
        self.short_cut = nn.Conv2d(outplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.conv = nn.Sequential(
            nn.Conv2d(inplanes+outplanes,outplanes,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(outplanes,outplanes,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(outplanes,outplanes,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(outplanes,outplanes,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outplanes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, y):
        x = self.squeeze2(self.squeeze1(x))   
        x = F.interpolate(x,size=y.size()[2:],mode='bilinear',align_corners=True)
        p = self.conv(torch.cat((x,y),1))
        sc = self.short_cut(x)
        p  = p+sc
        p2 = self.conv2(p)
        p  = p+p2
        return p

    def initialize(self):
        weight_init(self)

class MTD2(nn.Module):
    def __init__(self) -> None:
        super(MTD2,self).__init__()

        self.md1 = MTD1(64,64)

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self,s,r,up):
        up = F.interpolate(up,size=s.size()[2:],mode='bilinear',align_corners=True)
        sr = self.conv3x3(s+r)
        out  =self.md1(sr,up)
        return out
    def initialize(self):
        weight_init(self)

class decodersr(nn.Module):
    def __init__(self) -> None:
        super(decodersr,self).__init__()
        self.sqz_r5 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=3,stride=1,dilation=1,padding=1), 
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True)
            )
        self.cov1 = nn.Sequential(
            BasicConv2d(256, 128, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(128, 128, kernel_size=(3, 1), padding=(1, 0))
        )
        self.cov2 = nn.Sequential(
            BasicConv2d(512, 256, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(256, 256, kernel_size=(3, 1), padding=(1, 0))
        )

        self.mtd1=MTD1(512,64)
        self.mtd1_2=MTD1(128,64)
        self.mtd1_3=MTD1(256,64)
        self.mtd2=MTD2()
        self.d5 = DB2(128,64)
        self.d6 = DB2(64,64)
        
    def forward(self,s1,s2,s3,s4,r2,r3,r4,r5):
        r5 = F.interpolate(r5,size = s2.size()[2:],mode='bilinear',align_corners=True) 
        s1 = F.interpolate(s1,size = r4.size()[2:],mode='bilinear',align_corners=True) 

        s3_=self.mtd1(s4,s3)

        s2=self.cov1(s2)
        s1_=self.mtd1_2(s2,s1)

        r5=self.cov2(r5)
        r4_=self.mtd1_3(r5,r4)

        fu_feature_r4=self.mtd2(s1_,r4_,s3_)

        r3_,_ = self.d5(r3,fu_feature_r4)
        r2_,_ = self.d6(r2,r3_)

        return r2_
        
    def initialize(self):
        weight_init(self)
        

class Spade(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super(Spade, self).__init__()
        self.param_free_norm = nn.BatchNorm2d(out_channels, affine=False)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(True)
        )
        self.mlp_gamma = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x, edge):
        normalized = self.param_free_norm(x)

        edge = F.interpolate(edge, size=x.size()[2:], mode='bilinear')
        actv = self.mlp_shared(edge)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1 + gamma) + beta
        return out

    def initialize(self):
        weight_init(self)

class GRA(nn.Module):
    def __init__(self, channel, subchannel):
        super(GRA, self).__init__()
        self.group = channel//subchannel
        self.conv = nn.Sequential(
            nn.Conv2d(channel + self.group, channel, 3, padding=1), nn.ReLU(True),
        )
        self.score = nn.Conv2d(channel, 1, 3, padding=1)

    def forward(self, x, y):
        if self.group == 1:
            x_cat = torch.cat((x, y), 1)
        elif self.group == 2:
            xs = torch.chunk(x, 2, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y), 1)
        elif self.group == 4:
            xs = torch.chunk(x, 4, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y), 1)
        elif self.group == 8:
            xs = torch.chunk(x, 8, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y), 1)
        elif self.group == 16:
            xs = torch.chunk(x, 16, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y,
            xs[8], y, xs[9], y, xs[10], y, xs[11], y, xs[12], y, xs[13], y, xs[14], y, xs[15], y), 1)
        elif self.group == 32:
            xs = torch.chunk(x, 32, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y,
            xs[8], y, xs[9], y, xs[10], y, xs[11], y, xs[12], y, xs[13], y, xs[14], y, xs[15], y,
            xs[16], y, xs[17], y, xs[18], y, xs[19], y, xs[20], y, xs[21], y, xs[22], y, xs[23], y,
            xs[24], y, xs[25], y, xs[26], y, xs[27], y, xs[28], y, xs[29], y, xs[30], y, xs[31], y), 1)
        else:
            raise Exception("Invalid Channel")

        x = x + self.conv(x_cat)
        y = y + self.score(x)

        return x, y

class ReverseStage(nn.Module):
    def __init__(self, channel):
        super(ReverseStage, self).__init__()
        self.weak_gra = GRA(channel, channel)
        self.medium_gra = GRA(channel, 8)
        self.strong_gra = GRA(channel, 1)

    def forward(self, x, y):
        # reverse guided block
        y = -1 * (torch.sigmoid(y)) + 1

        # three group-reversal attention blocks
        x, y = self.weak_gra(x, y)
        x, y = self.medium_gra(x, y)
        _, y = self.strong_gra(x, y)

        return y

class CFR(nn.Module):
    def __init__(self) -> None:
        super(CFR, self).__init__()
        self.ra21_conv = BasicConv2d(64, 1, kernel_size=3, padding=1)
        
        self.r1_conv = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra1_conv= BasicConv2d(128, 64, kernel_size=3, padding=1)

    def forward(self,r2,r1):
        r2=F.interpolate(r2, size=r1.size()[2:], mode='bilinear')
        r2= self.ra21_conv(r2)
        r_2_weight = r2.expand(-1, r1.size()[1], -1, -1)
        r_2_out = r_2_weight * r1
        r1=self.r1_conv(r1)
        r_1= torch.cat((r1, r_2_out), 1)
        r_1_out = self.ra1_conv(r_1)

        return r_1_out

class NeighborConnectionDecoder(nn.Module):
    def __init__(self, channel):
        super(NeighborConnectionDecoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d((2 * channel), (2 * channel), 3, padding=1)
        self.conv_concat2 = BasicConv2d((2 * channel), (2 * channel), 3, padding=1)
        self.conv_concat3 = BasicConv2d((3 * channel), (3 * channel), 3, padding=1)
        self.conv4 = BasicConv2d((3 * channel), (3 * channel), 3, padding=1)
        self.conv5 = nn.Conv2d((3 * channel), 1, 1)

    def forward(self, x1, x2, x3):
        x3=self.upsample(x3)
        x2=self.upsample(x2)
        x1_1 = x1
        x2_1 = (self.conv_upsample1(self.upsample(x1)) * x2)
        x3_1 = ((self.conv_upsample2(x2_1) * self.conv_upsample3(x2)) * x3)
        x2_2 = torch.concat((x2_1, self.conv_upsample4(self.upsample(x1_1))), dim=1)
        x2_2 = self.conv_concat2(x2_2)
        x3_2 = torch.concat((x3_1, self.conv_upsample5(x2_2)), dim=1)
        x3_2 = self.conv_concat3(x3_2)
        x = self.conv4(x3_2)
        x = self.conv5(x)
        return x

class goodnet(nn.Module):
    def __init__(self) -> None:
        super(goodnet,self).__init__()
        self.RFEfussion=RFEfussion()

        self.swin2=Swintransformer(448)
        self.ResNet=res2net50_v1b_26w_4s()
        self.swin2.load_state_dict(torch.load('/home/fabian/BRL/zhoushanfeng/2024GoodNet/model_paths/swin_base_patch4_window7_224_22k.pth')['model'],strict=False)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)

        
        self.BasicConv1 = nn.Sequential(
            BasicConv2d(256, 128, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(128, 64, kernel_size=(3, 1), padding=(1, 0))
        )
        self.BasicConv2= nn.Sequential(
            BasicConv2d(512, 256, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(256, 128, kernel_size=(3, 1), padding=(1, 0))
        )
        self.BasicConv3 = nn.Sequential(
            BasicConv2d(1024, 512, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(512, 256, kernel_size=(3, 1), padding=(1, 0))
        )
        self.BasicConv4=nn.Sequential(
            BasicConv2d(2048, 1024, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(1024, 512, kernel_size=(3, 1), padding=(1, 0))
        )
        self.msf_conv_cat = BasicConv2d(64 * 4, 64, kernel_size=3, padding=1)
        self.msf_linear = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        self.linearr1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.linearr5 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr6 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr7 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr8 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.RFM1 = RFM1(256, 64)
        self.RFM2 = RFM1(512, 64)
        self.RFM3 = RFM1(1024, 64)
        self.RFM4 = RFM1(2048, 64)
        
        self.cfr1=CFR()   
        self.cfr2=CFR()
        self.cfr3=CFR()
        self.cfr4=CFR()

        self.SAM1=SAM(64)
        self.SAM2=SAM(64)
        self.SAM3=SAM(64)
        self.SAM0=SAM(64)

        self.out_SAM0 = nn.Conv2d(64, 1, 1)
        self.out_SAM1 = nn.Conv2d(64, 1, 1)
        self.out_SAM2 = nn.Conv2d(64, 1, 1)
        self.out_SAM3 = nn.Conv2d(64, 1, 1)
        self.out_SAM4 = nn.Conv2d(64, 1, 1)


        self.ra11_conv = BasicConv2d(64, 1, kernel_size=3, padding=1)
        self.ra21_conv = BasicConv2d(64, 1, kernel_size=3, padding=1)
        self.ra31_conv = BasicConv2d(64, 1, kernel_size=3, padding=1)
        self.ra41_conv = BasicConv2d(64, 1, kernel_size=3, padding=1)

        self.ra1_conv = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra4_conv = BasicConv2d(64, 64, kernel_size=3, padding=1)

        self.decodersr=decodersr()
        self.linear1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linear2 = nn.Conv2d(64, 1 , kernel_size=3, stride=1, padding=1)
        self.flat32=nn.Conv2d(64, 32 , kernel_size=3, stride=1, padding=1)
        self.NCD = NeighborConnectionDecoder(64)
        self.RS1 = ReverseStage(32)
        self.RS2 = ReverseStage(32)
        self.RS3 = ReverseStage(32)
        
        self.initialize()
    
    def forward(self,img,mask):
        
        x=self.RFEfussion(img,mask)
        #3个尺度大小的图片
        
        shape = x.size()[2:]
        x2=F.interpolate(x, size=(448,448), mode='bilinear',align_corners=True)
# s1 torch.Size([1, 128, 112, 112])
# s2 torch.Size([1, 256, 56, 56])
# s3 torch.Size([1, 512, 28, 28])
# s4 torch.Size([1, 512, 28, 28])    
        s1,s2,s3,s4 = self.swin2(x2)
        x_resnet = self.ResNet.conv1(x)
        x_resnet = self.ResNet.bn1(x_resnet)
        x_resnet = self.ResNet.relu(x_resnet)
        x_resnet = self.ResNet.maxpool(x_resnet)


# layer1 torch.Size([1, 256, 256, 256])
# layer2 torch.Size([1, 512, 128, 128])
# layer3 torch.Size([1, 1024, 64, 64])
# layer4 torch.Size([1, 2048, 32, 32])
        layer1 = self.ResNet.layer1(x_resnet)
        layer2 = self.ResNet.layer2(layer1)
        layer3 = self.ResNet.layer3(layer2)
        layer4 = self.ResNet.layer4(layer3)

        # print("layer1",layer1.shape)  
        # print("layer2",layer2.shape)      
        # print("layer3",layer3.shape)      
        # print("layer4",layer4.shape)   
  
#   s4s4_msf torch.Size([1, 64, 22, 22])
# s3_msf torch.Size([1, 64, 24, 24])
      
        rme1, rme2, rme3, rme4 = self.RFM1(layer1), self.RFM2(layer2), self.RFM3(layer3), self.RFM4(layer4)
        
        cfr3=self.cfr3(rme4,rme3)
        cfr2=self.cfr2(cfr3,rme2)       
        cfr1=self.cfr1(cfr2,rme1)

        guider1 = cfr1
        guider2 = F.interpolate(cfr2, scale_factor=2, mode='bilinear')
        guider3 = F.interpolate(cfr3, scale_factor=4, mode='bilinear')
        guider4 = F.interpolate(rme4, scale_factor=8, mode='bilinear')

        # spade4 = self.spade4(guider4 , msf_map)
        # spade3 = self.spade3(guider3 , msf_map)
        # spade2 = self.spade2(guider2 , msf_map)
        # spade1 = self.spade1(guider1 , msf_map)

# spade4 torch.Size([1, 64, 256, 256])
# spade3 torch.Size([1, 64, 256, 256])
# spade2 torch.Size([1, 64, 256, 256])
# spade1 torch.Size([1, 64, 256, 256])
        r1, r2, r3, r4 = self.BasicConv1(layer1), self.BasicConv2(layer2), self.BasicConv3(layer3), self.BasicConv4(layer4)
        # self.RME1_1 = RF2B(256, 64)
        # self.RME2_2 = RF2B(512, 128)
        # self.RME3_3 = RF2B(1024, 256)
        # self.RME4_4 = RF2B(2048, 512)
        predsr = self.decodersr(s1,s2,s3,s4,r1,r2,r3,r4)
        #pred23 torch.Size([2, 64, 256, 256])
        # print("pred23",pred23.shape)
        map_sr=self.linear1(predsr)
        pred_sr = F.interpolate(map_sr, size=shape, mode='bilinear') 

        SA_3 = self.SAM3(guider4,guider3)
        SA_2 = self.SAM2(SA_3,guider2)
        SA_1 = self.SAM1(SA_2,guider1)

        # SA_0 = self.SAM0(SA_1,predsr)

        map_24 = self.out_SAM3(SA_3)
        # rme4 torch.Size([2, 64, 32, 32])
        # map_24 torch.Size([2, 1, 32, 32])
        # SA_3 torch.Size([2, 64, 256, 256])
        # print("rme4",rme4.shape)
        # print("map_24",map_24.shape)
        # print("SA_3",SA_3.shape)
        # out21 torch.Size([2, 1, 256, 256])
        map_23 = self.out_SAM2(SA_2)+ map_24
        map_22 = self.out_SAM1(SA_1) + map_23
        # map_21 = self.out_SAM0(SA_0) +map_22
        # print("out21",map_21.shape)
        #out21 torch.Size([2, 1, 256, 256])
        # out_21 = F.interpolate(map_21, size=shape, mode='bilinear')
        out_22 = F.interpolate(map_22, size=shape, mode='bilinear')
        out_23 = F.interpolate(map_23, size=shape, mode='bilinear')
        out_24 = F.interpolate(map_24, size=shape, mode='bilinear')

# layer1 torch.Size([1, 256, 256, 256])
# layer2 torch.Size([1, 512, 128, 128])
# layer3 torch.Size([1, 1024, 64, 64])
# layer4 torch.Size([1, 2048, 32, 32]) 
        S_g = self.NCD(SA_2, SA_1, predsr)
        S_g=F.interpolate(S_g, scale_factor=2, mode='bilinear')
        

         

        # print("map_1",map_1.shape)
        # print("map_2",map_2.shape)
        # print("map_3",map_3.shape)
        # print("map_4",map_4.shape)


        return S_g,pred_sr,out_22,out_23,out_24

    def initialize(self):
        weight_init(self)
        
