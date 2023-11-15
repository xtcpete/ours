import torch.nn as nn
import torch.nn.functional as F

# Feature pooling
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

"""
out_planes number of feature map
capture spatial info such as immediate neighborhood of each pixel but small enough to be computational efficient
if the padding is when we want learning strictly tied to the input image content without any influence from padding
"""
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()

        # if stride is not 1, we want to downsample the fm by a factor of stride
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.conv2 = conv3x3(planes, planes)

        # batch normalize
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)

        # activition
        self.relu = nn.ReLU(inplace=True)

        # if stride is not 1, we need to downsample the input 'x' before adding it to the output of conv2
        if stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, planes, stride=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):

        y = x

        # pass the input through conv1, bn1 and relu.  y has planes number of channels
        y = self.relu(self.bn1(self.conv1(y)))

        # pass the feature map through second layer to extract features and create new ones
        y = self.bn2(self.conv2(y))

        # downsample x as needed to x + y
        if self.downsample is not None:
            x = self.downsample(x)

        # element-wise addition.  Skip connection
        return self.relu(x + y)
    
class ResNetFPN_8_2(nn.Module):

    """
    ResNet + FPN, output resolution are 1/8 and 1/2
    """

    def __init__(self, config):
        super().__init__()

        block = BasicBlock
        initial_dim = config['initial_dim']
        block_dims = config['block_dims']

        self.in_planes = initial_dim

        """Networks"""
        # the input image is grayscale
        self.conv1 = nn.Conv2d(3, initial_dim, kernel_size=7, stride=2, padding=3, bias=False) # srhink the spatial dimension by 1/2
        self.bn1 = nn.BatchNorm2d(initial_dim)
        self.relu = nn.ReLU(inplace=True)

        # reduce the spatial dimension of the input by half, stride 1, increase the channels
        self.layer1 = self._make_layer(block, block_dims[0], stride=1) # not shrink
        self.layer2 = self._make_layer(block, block_dims[1], stride=2) # shrink the spatial dimension by 1/2
        self.layer3 = self._make_layer(block, block_dims[2], stride=2) # shrink again by 1/2

        # FPN upsample
        self.layer3_outconv = conv1x1(block_dims[2], block_dims[2]) # channel to block_dims[2]

        self.layer2_outconv = conv1x1(block_dims[1], block_dims[2]) # channel to block_dims[2]
        self.layer2_outconv2 = nn.Sequential(
            conv3x3(block_dims[2], block_dims[2]),
            nn.BatchNorm2d(block_dims[2]),
            nn.LeakyReLU(),
            conv3x3(block_dims[2], block_dims[1])
        ) # channel to block_dims[1]

        self.layer1_outconv = conv1x1(block_dims[0], block_dims[1]) # channel to block_dims[1]
        self.layer1_outconv2 = nn.Sequential(
            conv3x3(block_dims[1], block_dims[1]),
            nn.BatchNorm2d(block_dims[1]),
            nn.LeakyReLU(),
            conv3x3(block_dims[1], block_dims[0])
        ) # channel to block_dims[0]
        
        # iterate through all the modules in nn.Module
        for m in self.modules():
            
            if isinstance(m, nn.Conv2d):
                # initialize the weights of the conv using kaiming (He) normal initialization
                # it is designed to keep the scale of gradients roughly the same in all layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)): # check if module is a BatchNorm2d or GroupNorm layers
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
    def forward(self, x):

        # downsamples to extract features
        x0 = self.relu(self.bn1(self.conv1(x))) # 1/2
        x1 = self.layer1(x0)
        x2 = self.layer2(x1) # 1/4
        x3 = self.layer3(x2) # 1/8

        # FPN
        x3_out = self.layer3_outconv(x3)

        # x3_out is scaled by a factor of 2 so x3_out_2x has the same dimensions as x2
        x3_out_2x = F.interpolate(x3_out, scale_factor=2., mode='bilinear', align_corners=True)

        # skip connection, block_dims[1] to [2] then [2] to block_dims[1]
        x2_out = self.layer2_outconv(x2)
        x2_out = self.layer2_outconv2(x2_out+x3_out_2x)

        # x2 spatial dimension * 2 then blocks_dims[0] to [1] then add then [1] to [0]
        x2_out_2x = F.interpolate(x2_out, scale_factor=2., mode='bilinear', align_corners=True)
        x1_out = self.layer1_outconv(x1)
        x1_out = self.layer1_outconv2(x1_out+x2_out_2x)

        # features extracted by the network at two different scales 1/8 and 1/2 the original resolution
        # [coarse, fine]
        return [x3_out, x1_out]


        
    def _make_layer(self, block, dim, stride=1):

        layer1 = block(self.in_planes, dim, stride=stride)
        layer2 = block(dim, dim, stride=1)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)




        