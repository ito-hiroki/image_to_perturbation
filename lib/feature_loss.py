import torch
import torchvision
import torchvision.models as models

def vgg16_l2(input,ref,n):
    with torch.no_grad():
        input=input.cuda()
        ref=ref.cuda()
        vgg16 = models.vgg16(pretrained=True)
        features=vgg16.features[0:n]
        features=features.cuda()
        features.eval()
    return torch.mean((features(input)-features(ref))**2).item()

def vgg16_l1(input,ref):
    input=input.cuda()
    l1loss=torch.nn.L1Loss()
    ref=ref.cuda()
    vgg16 = models.vgg16(pretrained=True)
    features=vgg16.features
    features=features.cuda()
    features.eval()
    return l1loss(features(input),features(ref))

def densenet_l1(input,ref):
    input=input.cuda()
    l1loss=torch.nn.L1Loss()
    ref=ref.cuda()
    densenet = models.densenet161(pretrained=True)
    features=densenet.features
    features=features.cuda()
    features.eval()
    return l1loss(features(input),features(ref))

def densenet_l2(input,ref):
    with torch.no_grad():
        input=input.cuda()
        ref=ref.cuda()
        densenet = models.densenet161(pretrained=True)
        features=densenet.features
        features=features.cuda()
        features.eval()
    return torch.mean((features(input)-features(ref))**2).item()
