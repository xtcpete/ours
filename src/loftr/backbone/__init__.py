from .resnet_fpn import ResNetFPN_8_2


# get coarse and fine feture map
def build_backbone(config):
    return ResNetFPN_8_2(config['resnetfpn'])
