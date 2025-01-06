import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models.resnet as resnet
from torchvision.models.vgg import VGG


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits



class PSPModule(nn.Module):
    def __init__(self, features=512, out_features=1024):
        super().__init__()
        self.stages1 = self._make_stage(features, 1)
        self.stages2 = self._make_stage(features, 2)
        self.stages3 = self._make_stage(features, 3)
        self.stages4 = self._make_stage(features, 6)
        self.bottleneck = nn.Conv2d(out_features, out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, 128, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors1 = F.upsample(input=self.stages1(feats), size=(h, w), mode='bilinear')
        priors2 = F.upsample(input=self.stages2(feats), size=(h, w), mode='bilinear')
        priors3 = F.upsample(input=self.stages3(feats), size=(h, w), mode='bilinear')
        priors4 = F.upsample(input=self.stages4(feats), size=(h, w), mode='bilinear')
        map = torch.cat((feats, priors4, priors3, priors2, priors1), 1)
        # bottle = self.bottleneck(priors)
        # return self.relu(bottle)
        return map

class PSPNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(PSPNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 512 // factor)
        self.PSP = PSPModule(512, out_features=1024)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.PSP(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class FCN8s(nn.Module):

    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1     = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2     = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4     = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5     = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']  # size=(N, 512, x.H/32, x.W/32)
        x4 = output['x4']  # size=(N, 512, x.H/16, x.W/16)
        x3 = output['x3']  # size=(N, 256, x.H/8,  x.W/8)

        score = self.relu(self.deconv1(x5))               # size=(N, 512, x.H/16, x.W/16)
        score = self.bn1(score + x4)                      # element-wise add, size=(N, 512, x.H/16, x.W/16)
        score = self.relu(self.deconv2(score))            # size=(N, 256, x.H/8, x.W/8)
        score = self.bn2(score + x3)                      # element-wise add, size=(N, 256, x.H/8, x.W/8)
        score = self.bn3(self.relu(self.deconv3(score)))  # size=(N, 128, x.H/4, x.W/4)
        score = self.bn4(self.relu(self.deconv4(score)))  # size=(N, 64, x.H/2, x.W/2)
        score = self.bn5(self.relu(self.deconv5(score)))  # size=(N, 32, x.H, x.W)
        score = self.classifier(score)                    # size=(N, n_class, x.H/1, x.W/1)
        # score = self.softmax(score)

        return score  # size=(N, n_class, x.H/1, x.W/1)

class VGGNet(VGG):
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]

        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        if remove_fc:  # delete redundant fully-connected layer params, can save memory
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}

        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d"%(idx+1)] = x

        return output

ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# cropped version from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)





'''======================================   old   ======================================'''
class ResBlock(nn.Module):
    def __init__(self, inch, outch, stride=1, dilation=1):
        # Residual Block
        # inch: input feature channel
        # outch: output feature channel
        # stride: the stride of  convolution layer
        super(ResBlock, self).__init__()
        assert (stride == 1 or stride == 2)

        self.conv1 = nn.Conv2d(inch, outch, 3, stride, padding=dilation, bias=False,
                               dilation=dilation)
        self.bn1 = nn.BatchNorm2d(outch)
        self.conv2 = nn.Conv2d(outch, outch, 3, 1, padding=dilation, bias=False,
                               dilation=dilation)
        self.bn2 = nn.BatchNorm2d(outch)

        if inch != outch:
            self.mapping = nn.Sequential(
                nn.Conv2d(inch, outch, 1, stride, bias=False),
                nn.BatchNorm2d(outch)
            )
        else:
            self.mapping = None

    def forward(self, x):
        y = x
        if not self.mapping is None:
            y = self.mapping(y)

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))

        out += y
        out = F.relu(out, inplace=True)

        return out




class encoder_basic(nn.Module):
    def __init__(self):
        super(encoder_basic, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.b1_1 = ResBlock(64, 64, 1)
        self.b1_2 = ResBlock(64, 64, 1)

        self.b2_1 = ResBlock(64, 128, 2)
        self.b2_2 = ResBlock(128, 128, 1)

        self.b3_1 = ResBlock(128, 256, 2)
        self.b3_2 = ResBlock(256, 256, 1)

        self.b4_1 = ResBlock(256, 512, 2)
        self.b4_2 = ResBlock(512, 512, 1)

    def forward(self, im):
        x1 = F.relu(self.bn1(self.conv1(im)), inplace=True)
        x2 = self.b1_2(self.b1_1(self.maxpool(x1)))
        x3 = self.b2_2(self.b2_1(x2))
        x4 = self.b3_2(self.b3_1(x3))
        x5 = self.b4_2(self.b4_1(x4))
        return x1, x2, x3, x4, x5


class decoder_basic(nn.Module):
    def __init__(self):
        super(decoder_basic, self).__init__()
        self.conv1 = nn.Conv2d(512 + 256 + 128, 512, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv1_1 = nn.Conv2d(512, 21, 3, 1, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(21)
        self.conv2 = nn.Conv2d(64 + 21, 21, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(21)
        self.conv3 = nn.Conv2d(21, 3, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(3)
        self.conv4 = nn.Conv2d(3, 3, 3, 1, 1, bias=False)
        self.sf = nn.Softmax(dim=1)

    def forward(self, im, x1, x2, x3, x4, x5):
        _, _, nh, nw = x3.size()
        x5 = F.interpolate(x5, [nh, nw], mode='bilinear')
        x4 = F.interpolate(x4, [nh, nw], mode='bilinear')
        y1 = F.relu(self.bn1(self.conv1(torch.cat([x3, x4, x5], dim=1))), inplace=True)
        y1 = F.relu(self.bn1_1(self.conv1_1(y1)), inplace=True)

        _, _, nh, nw = x2.size()
        y1 = F.interpolate(y1, [nh, nw], mode='bilinear')
        y1 = torch.cat([y1, x2], dim=1)
        y2 = F.relu(self.bn2(self.conv2(y1)), inplace=True)

        _, _, nh, nw = x1.size()
        y2 = F.interpolate(y2, [nh, nw], mode='bilinear')
        y3 = F.relu(self.bn3(self.conv3(y2)), inplace=True)

        y4 = self.sf(self.conv4(y3))

        _, _, nh, nw = im.size()
        y4 = F.interpolate(y4, [nh, nw], mode='bilinear')

        # pred = -torch.log(torch.clamp(y4, min=1e-8))
        # pred = self.sf(y4)
        pred = y4

        return pred

    def img_show(self, im, x1, x2, x3, x4, x5):
        _, _, nh, nw = x3.size()
        x5 = F.interpolate(x5, [nh, nw], mode='bilinear')
        x4 = F.interpolate(x4, [nh, nw], mode='bilinear')
        y1 = F.relu(self.bn1(self.conv1(torch.cat([x3, x4, x5], dim=1))), inplace=True)
        y1 = F.relu(self.bn1_1(self.conv1_1(y1)), inplace=True)

        _, _, nh, nw = x2.size()
        y1 = F.interpolate(y1, [nh, nw], mode='bilinear')
        y1 = torch.cat([y1, x2], dim=1)
        y2 = F.relu(self.bn2(self.conv2(y1)), inplace=True)

        _, _, nh, nw = x1.size()
        y2 = F.interpolate(y2, [nh, nw], mode='bilinear')
        y3 = F.relu(self.bn3(self.conv3(y2)), inplace=True)

        y4 = self.conv4(y3)
        _, _, nh, nw = im.size()
        y4 = F.interpolate(y4, [nh, nw], mode='bilinear')


        return y4


class encoderDilation(nn.Module):
    def __init__(self):
        super(encoderDilation, self).__init__()

        ## IMPLEMENT YOUR CODE HERE
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False, )
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.b1_1 = ResBlock(64, 64, 1)
        self.b1_2 = ResBlock(64, 64, 1)

        self.b2_1 = ResBlock(64, 128, 2)
        self.b2_2 = ResBlock(128, 128, 1)

        self.b3_1 = ResBlock(128, 256, 1, 2)
        self.b3_2 = ResBlock(256, 256, 1, 2)

        self.b4_1 = ResBlock(256, 512, 1, 4)
        self.b4_2 = ResBlock(512, 512, 1, 4)

    def forward(self, im):
        ## IMPLEMENT YOUR CODE HERE
        x1 = F.relu(self.bn1(self.conv1(im)), inplace=True)
        x2 = self.b1_2(self.b1_1(self.maxpool(x1)))
        x3 = self.b2_2(self.b2_1(x2))
        x4 = self.b3_2(self.b3_1(x3))
        x5 = self.b4_2(self.b4_1(x4))

        return x1, x2, x3, x4, x5


class decoderDilation(nn.Module):
    def __init__(self, isSpp=False):
        super(decoderDilation, self).__init__()

        ## IMPLEMENT YOUR CODE HERE
        self.conv1 = nn.Conv2d(512 + 256 + 128, 512, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv1_1 = nn.Conv2d(512, 21, 3, 1, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(21)
        self.conv2 = nn.Conv2d(64 + 21, 21, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(21)
        self.conv3 = nn.Conv2d(21, 3, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(3)
        self.conv4 = nn.Conv2d(3, 3, 3, 1, 1, bias=False)
        self.sf = nn.Softmax(dim=1)

    def forward(self, im, x1, x2, x3, x4, x5):
        ## IMPLEMENT YOUR CODE HERE
        _, _, nh, nw = x3.size()
        # x5 = F.interpolate(x5, [nh, nw], mode='bilinear' )
        # x4 = F.interpolate(x4, [nh, nw], mode='bilinear' )
        y1 = F.relu(self.bn1(self.conv1(torch.cat([x3, x4, x5], dim=1))), inplace=True)
        y1 = F.relu(self.bn1_1(self.conv1_1(y1)), inplace=True)

        _, _, nh, nw = x2.size()
        y1 = F.interpolate(y1, [nh, nw], mode='bilinear')
        y1 = torch.cat([y1, x2], dim=1)
        y2 = F.relu(self.bn2(self.conv2(y1)), inplace=True)

        _, _, nh, nw = x1.size()
        y2 = F.interpolate(y2, [nh, nw], mode='bilinear')
        y3 = F.relu(self.bn3(self.conv3(y2)), inplace=True)

        y4 = self.sf(self.conv4(y3))

        _, _, nh, nw = im.size()
        y4 = F.interpolate(y4, [nh, nw], mode='bilinear')

        #pred = -torch.log(torch.clamp(y4, min=1e-8))
        #pred = self.sf(y4)
        pred = y4

        return pred





class encoderSPP(nn.Module):
    def __init__(self):
        super(encoderSPP, self).__init__()

        ## IMPLEMENT YOUR CODE HERE
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False, )
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.b1_1 = ResBlock(64, 64, 1)
        self.b1_2 = ResBlock(64, 64, 1)

        self.b2_1 = ResBlock(64, 128, 2)
        self.b2_2 = ResBlock(128, 128, 1)

        self.b3_1 = ResBlock(128, 256, 1, 2)
        self.b3_2 = ResBlock(256, 256, 1, 2)

        self.b4_1 = ResBlock(256, 512, 1, 4)
        self.b4_2 = ResBlock(512, 512, 1, 4)

        self.psp = PSPModule(512, 1024)

    def forward(self, im):
        ## IMPLEMENT YOUR CODE HERE
        x1 = F.relu(self.bn1(self.conv1(im)), inplace=True)
        x2 = self.b1_2(self.b1_1(self.maxpool(x1)))
        x3 = self.b2_2(self.b2_1(x2))
        x4 = self.b3_2(self.b3_1(x3))
        x_py = self.b4_2(self.b4_1(x4))
        x5 = self.psp(x_py)

        return x1, x2, x3, x4, x5


class decoderSPP(nn.Module):
    def __init__(self, isSpp=False):
        super(decoderSPP, self).__init__()

        ## IMPLEMENT YOUR CODE HERE
        self.conv1 = nn.Conv2d(1024 + 256 + 128, 512, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv1_1 = nn.Conv2d(512, 21, 3, 1, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(21)
        self.conv2 = nn.Conv2d(64 + 21, 21, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(21)
        self.conv3 = nn.Conv2d(21, 3, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(3)
        self.conv4 = nn.Conv2d(3, 3, 3, 1, 1, bias=False)
        self.sf = nn.Softmax(dim=1)

    def forward(self, im, x1, x2, x3, x4, x5):
        ## IMPLEMENT YOUR CODE HERE
        _, _, nh, nw = x3.size()
        y1 = F.relu(self.bn1(self.conv1(torch.cat([x3, x4, x5], dim=1))), inplace=True)
        y1 = F.relu(self.bn1_1(self.conv1_1(y1)), inplace=True)

        _, _, nh, nw = x2.size()
        y1 = F.interpolate(y1, [nh, nw], mode='bilinear')
        y1 = torch.cat([y1, x2], dim=1)
        y2 = F.relu(self.bn2(self.conv2(y1)), inplace=True)

        _, _, nh, nw = x1.size()
        y2 = F.interpolate(y2, [nh, nw], mode='bilinear')
        y3 = F.relu(self.bn3(self.conv3(y2)), inplace=True)

        y4 = self.sf(self.conv4(y3))

        _, _, nh, nw = im.size()
        y4 = F.interpolate(y4, [nh, nw], mode='bilinear')

        #pred = -torch.log(torch.clamp(y4, min=1e-8))
        #pred = self.sf(y4)
        pred = y4

        return pred


def loadPretrainedWeight(network, isOutput=False):
    paramList = []
    resnet18 = resnet.resnet18(pretrained=True)
    for param in resnet18.parameters():
        paramList.append(param)

    cnt = 0
    for param in network.parameters():
        if paramList[cnt].size() == param.size():
            param.data.copy_(paramList[cnt].data)
            # param.data.copy_(param.data )
            if isOutput:
                print(param.size())
        else:
            print(param.shape, paramList[cnt].shape)
            break
        cnt += 1