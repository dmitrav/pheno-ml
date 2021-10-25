
import torch
from torch import nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.transforms import RandomResizedCrop
from pl_bolts.models.self_supervised import SwAV

from src.datasets import MultiCropDataset


def get_supervised_resnet():

    resnet50 = models.resnet50(pretrained=True)
    modules = list(resnet50.children())[:-1]
    resnet50 = nn.Sequential(*modules)
    for p in resnet50.parameters():
        p.requires_grad = False

    return resnet50


def get_self_supervised_vit():
    vitb8 = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
    return vitb8


def get_self_supervised_resnet(method='swav'):

    if method == 'swav':
        weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/swav_imagenet/swav_imagenet.pth.tar'
        model = SwAV.load_from_checkpoint(weight_path, strict=True)
        model.freeze()

    elif method == 'dino':
        model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    else:
        raise ValueError('Unknown method: {}'.format(method))

    return model


if __name__ == "__main__":

    path_to_data = "/Users/andreidm/ETH/projects/pheno-ml/data/cropped/training/"
    crop_size = 224
    batch_size = 32
    train_size = 100

    cropping_strategy = ([crop_size], [1], [1], [1])
    dataset = MultiCropDataset(path_to_data, *cropping_strategy, no_aug=True, size_dataset=train_size, n_channels=3)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    model = get_supervised_resnet()
    # model = get_self_supervised_resnet()

    for batch in data_loader:
        n_crops = len(batch)
        for crops, _ in batch:

            codes = model(crops)
            print()



