
import numpy, random, torch, pandas, os
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.io import read_image

from src import constants


class RandomApply(torch.nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)


class MultiCropDataset(datasets.ImageFolder):
    def __init__(
            self,
            data_path,
            size_crops,
            nmb_crops,
            min_scale_crops,
            max_scale_crops,
            size_dataset=-1,
            return_index=False,
            no_aug=True,
            n_channels=1
    ):
        super(MultiCropDataset, self).__init__(data_path)
        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)
        if size_dataset >= 0:
            self.samples = self.samples[:size_dataset]
        self.return_index = return_index
        self.no_aug = no_aug
        self.n_channels = n_channels

        no_trans = []
        trans = []
        for i in range(len(size_crops)):
            no_trans.extend([
                             transforms.Compose([
                                 transforms.RandomResizedCrop(size_crops[i], scale=(min_scale_crops[i], max_scale_crops[i])),
                                 transforms.Grayscale(num_output_channels=self.n_channels),
                                 transforms.ToTensor()
                             ])
                         ] * nmb_crops[i]
            )

            trans.extend([
                             transforms.Compose([
                                 transforms.RandomResizedCrop(size_crops[i], scale=(min_scale_crops[i], max_scale_crops[i])),
                                 transforms.Grayscale(num_output_channels=self.n_channels),
                                 transforms.ToTensor(),
                                 transforms.RandomHorizontalFlip(p=0.5),
                                 RandomApply(transforms.GaussianBlur((3, 3), (.1, 2.0)), p=0.2)
                             ])
                         ] * nmb_crops[i]
            )

        self.trans = trans
        self.no_trans = no_trans

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = self.loader(path)
        if self.no_aug:
            # apply no augmentations
            multi_crops = list(map(lambda trans: trans(image), self.no_trans))
        else:
            multi_crops = list(map(lambda trans: trans(image), self.trans))

        multi_crops = [(crop, label) for crop in multi_crops]
        if self.return_index:
            return index, multi_crops
        return multi_crops


class MultiLabelDataset(Dataset):

    def __init__(self, label_dir_map, N=None, shuffle=False, transform=None, target_transform=None):

        self.label_dir_map = label_dir_map
        self.transform = transform
        self.target_transform = target_transform

        imgs = []
        labels = []
        for label, directory in label_dir_map.items():

            all_imgs = os.listdir(directory)
            all_labels = [label for x in all_imgs]

            if N is not None:
                # keep only n random (for balancing the data)
                n_random_indices = numpy.array(random.sample(range(len(all_imgs)), N))
                all_imgs = list(numpy.array(all_imgs)[n_random_indices])
                all_labels = list(numpy.array(all_labels)[n_random_indices])

            imgs.extend(all_imgs)
            labels.extend(all_labels)

        self.img_labels = pandas.DataFrame({
            'img': imgs,
            'label': labels
        })

        if shuffle:
            self.img_labels = self.img_labels.sample(frac=1)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        label = self.img_labels.iloc[idx, 1]
        directory = self.label_dir_map[label]
        img_path = os.path.join(directory, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image / 255.)
        if self.target_transform:
            label = self.target_transform(label)
        sample = (image, label)
        return sample


class CustomImageDataset(Dataset):

    def __init__(self, img_dir, label, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        if label >= 0:
            self.img_labels = pandas.DataFrame({
                'img': [f for f in os.listdir(img_dir)],
                'label': [label for f in os.listdir(img_dir)]
            })
        else:
            self.img_labels = pandas.DataFrame({
                'img': [f for f in os.listdir(img_dir)],
                'label': [self._infer_label(f) for f in os.listdir(img_dir)]
            })

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image / 255.)
        if self.target_transform:
            label = self.target_transform(label)
        sample = (image, label)
        return sample

    def _infer_label(self, filename):
        """ This method infers drugs labels from the filenames. """

        mapper = dict(zip(constants.drugs, [x for x in range(len(constants.drugs))]))
        self.n_classes = len(constants.drugs)

        for drug in constants.drugs:
            if drug in filename:
                return mapper[drug]


class JointImageDataset(Dataset):
    def __init__(self, datasets, transform=None, target_transform=None, n_channels=1):

        for subset in datasets:
            if subset.dataset.img_labels.columns[0] == 'path':
                # some weird bug made me do this
                continue
            else:
                subset.dataset.img_labels.insert(0, 'path', subset.dataset.img_dir)

        self.img_labels = pandas.concat([subset.dataset.img_labels for subset in datasets])
        self.transform = transform
        self.target_transform = target_transform
        self.n_channels = n_channels

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_labels.iloc[idx, 0], self.img_labels.iloc[idx, 1])

        if self.n_channels == 3:
            # read 3 channels
            image = numpy.array(Image.open(img_path).convert('RGB'))
            image = numpy.moveaxis(image, -1, 0)  # set channels as the first dim
            image = torch.Tensor(image)
        elif self.n_channels == 1:
            image = read_image(img_path)
        else:
            raise ValueError()

        label = self.img_labels.iloc[idx, 2]
        if self.transform:
            image = self.transform(image / 255.)
        if self.target_transform:
            label = self.target_transform(label)
        sample = (image, label)
        return sample
