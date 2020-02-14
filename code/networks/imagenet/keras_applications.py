#!/usr/bin/env python

"""
Author: Shashank Kotyan
Email: shashankkotyan@gmail.com
"""

from tensorflow.keras import applications

from networks.imagenet.imagenet_model import ImagenetModel


class ResNet(ImagenetModel):
    def __init__(self, args): 
        self.size = 224
        self.name = f"ResNet-{self.value}"
        ImagenetModel.__init__(self, args)
    
class ResNet50(ResNet):
    def __init__(self, args):
        self.value = 50
        self.model_class = applications.ResNet50
        ResNet.__init__(self, args)

class ResNet101(ResNet):
    def __init__(self, args):
        self.value = 101
        self.model_class = applications.ResNet101
        ResNet.__init__(self, args)

class ResNet152(ResNet):
    def __init__(self, args):
        self.value = 50
        self.model_class = applications.ResNet152
        ResNet.__init__(self, args)

class ResNetV2(ImagenetModel):
    def __init__(self, args): 
        self.size = 224
        self.name = f"ResNetV2-{self.value}"
        ImagenetModel.__init__(self, args)

class ResNetV250(ResNetV2):
    def __init__(self, args):
        self.value = 50
        self.model_class = applications.ResNet50V2
        ResNet.__init__(self, args)

class ResNetV2101(ResNetV2):
    def __init__(self, args):
        self.value = 101
        self.model_class = applications.ResNet101V2
        ResNet.__init__(self, args)

class ResNetV2152(ResNetV2):
    def __init__(self, args):
        self.value = 50
        self.model_class = applications.ResNet152V2
        ResNet.__init__(self, args)


class InceptionV3(ImagenetModel):
    def __init__(self, args):
        self.name = 'InceptionV3'
        self.size = 299
        self.model_class = applications.InceptionV3
        ImagenetModel.__init__(self, args)


class InceptionResNetV2(ImagenetModel):
    def __init__(self, args):
        self.name = 'InceptionResnetV2'
        self.size = 299
        self.model_class = applications.InceptionResNetV2
        ImagenetModel.__init__(self, args)


class Xception(ImagenetModel):
    def __init__(self, args):
        self.name = 'Xception'
        self.size = 299
        self.model_class = applications.Xception
        ImagenetModel.__init__(self, args)


class DenseNet(ImagenetModel):
    def __init__(self, args):
        self.name = f"DenseNet-{self.value}"
        self.size = 224
        ImagenetModel.__init__(self, args)

class DenseNet121(DenseNet):
    def __init__(self, args):
        self.value = 121
        self.model_class = applications.DenseNet121
        DenseNet.__init__(self, args) 

class DenseNet169(DenseNet):
    def __init__(self, args):
        self.value = 169
        self.model_class = applications.DenseNet169
        DenseNet.__init__(self, args)    

class DenseNet201(DenseNet):
    def __init__(self, args):
        self.value = 201
        self.model_class = applications.DenseNet201
        DenseNet.__init__(self, args)   


class VGG16(ImagenetModel):
    def __init__(self, args):
        self.size = 224
        self.name = 'VGG-16'
        self.model_class = applications.VGG16
        ImagenetModel.__init__(self, args)

class VGG_19(ImagenetModel):
    def __init__(self, args):
        self.name = 'VGG-19'
        self.size = 224
        self.model_class = applications.VGG19
        ImagenetModel.__init__(self, args)


class MobileNet(ImagenetModel):
    def __init__(self, args):
        self.name = 'MobileNet'
        self.size = 224
        self.model_class = applications.MobileNet
        ImagenetModel.__init__(self, args)

class MobileNetV2(ImagenetModel):
    def __init__(self, args):
        self.size = 224
        self.name = 'MobileNetV2'
        self.model_class = applications.MobileNetV2
        ImagenetModel.__init__(self, args) 


class NasNet(ImagenetModel):
    def __init__(self, args):
        self.name = f"Nasnet-{self.value}"
        ImagenetModel.__init__(self, args)

class NASNetMobile(NasNet):
    def __init__(self, args):
        self.value = 'Mobile'
        self.size  = 224
        self.model_class = applications.NASNetMobile
        NasNet.__init__(self, args)

class NASNetLarge(NasNet):
    def __init__(self, args):
        self.value = 'Large'
        self.size  = 331
        self.model_class = applications.NASNetLarge
        NasNet.__init__(self, args)