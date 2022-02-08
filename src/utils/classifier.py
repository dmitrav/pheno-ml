
import os, torch, time, numpy
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torchvision.io import read_image
from torchvision.transforms import Resize
from torchmetrics import Accuracy
from tqdm import tqdm

from src.datasets import CustomImageDataset


class DeepClassifier(torch.nn.Module):

    def __init__(self, n_classes=2):
        super().__init__()

        # self.model = torch.nn.Sequential(
        #     torch.nn.Conv2d(1, 64, (3, 3), stride=(1, 1), padding=(1, 1)),  # 64, 128, 128
        #     torch.nn.ReLU(True),
        #     torch.nn.MaxPool2d(2),  # 64, 64, 64
        #
        #     torch.nn.Conv2d(64, 16, (3, 3), stride=(1, 1), padding=(1, 1)),  # 16, 64, 64
        #     torch.nn.ReLU(True),
        #     torch.nn.MaxPool2d(2),  # 16, 32, 32
        #
        #     torch.nn.Conv2d(16, 8, (3, 3), stride=(1, 1), padding=(1, 1)),  # 8, 32, 32
        #     torch.nn.ReLU(True),
        #     torch.nn.MaxPool2d(2),  # 8, 16, 16
        #
        #     # torch.nn.Conv2d(32, 8, (3, 3), stride=(1, 1), padding=(1, 1)),  # 8, 16, 16
        #
        #     torch.nn.Flatten(),
        #     torch.nn.Linear(2048, 128),
        #     torch.nn.LeakyReLU(True),
        #     torch.nn.Linear(128, n_classes),
        #     torch.nn.Softmax(dim=1)
        # )

        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 128, (3, 3), stride=(1, 1), padding=(1, 1)),  # 128, 128, 128
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2),  # 128, 64, 64

            torch.nn.Conv2d(128, 64, (3, 3), stride=(1, 1), padding=(1, 1)),  # 64, 64, 64
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2),  # 64, 32, 32

            torch.nn.Conv2d(64, 32, (3, 3), stride=(1, 1), padding=(1, 1)),  # 32, 32, 32
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2),  # 32, 16, 16

            torch.nn.Conv2d(32, 8, (3, 3), stride=(1, 1), padding=(1, 1)),  # 8, 16, 16
            torch.nn.LeakyReLU(True),

            torch.nn.Flatten(),
            torch.nn.Linear(2048, 256),
            torch.nn.LeakyReLU(True),
            torch.nn.Linear(256, n_classes),
            torch.nn.Softmax(dim=1)
        )

        print(self)
        print('number of parameters: {}\n'.format(self.count_parameters()))

    def forward(self, features):
        return self.model(features)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_deep_classifier_weakly(epochs, trained_cl=None, batch_size=256, device=torch.device('cuda')):

    path_to_drugs = 'D:\ETH\projects\pheno-ml\data\classification_drugs\\'
    path_to_controls = 'D:\ETH\projects\pheno-ml\data\classification_controls\\'
    save_path = 'D:\ETH\projects\pheno-ml\\res\\'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    N_train = 90000  # ~90%
    image_size = 128
    transform = lambda img: Resize(size=image_size)(img) / 255.

    training_drugs = CustomImageDataset(path_to_drugs, 0, transform=transform)
    training_drugs, validation_drugs = torch.utils.data.random_split(training_drugs, [N_train, training_drugs.__len__() - N_train])

    training_controls = CustomImageDataset(path_to_controls, 1, transform=transform)
    training_controls, validation_controls = torch.utils.data.random_split(training_controls, [N_train, training_controls.__len__() - N_train])

    loader_train_drugs = DataLoader(training_drugs, batch_size=batch_size, shuffle=True)
    loader_train_controls = DataLoader(training_controls, batch_size=batch_size, shuffle=True)
    loader_val_drugs = DataLoader(validation_drugs, batch_size=batch_size, shuffle=True)
    loader_val_controls = DataLoader(validation_controls, batch_size=batch_size, shuffle=True)

    if trained_cl is not None:
        model = trained_cl
    else:
        model = DeepClassifier().to(device)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)

    criterion = torch.nn.CrossEntropyLoss()

    last_epoch_acc = run_weakly_supervised_classifier_training(loader_train_drugs, loader_train_controls,
                                                               loader_val_drugs, loader_val_controls,
                                                               model, optimizer, criterion, device, epochs=epochs,
                                                               save_path=save_path)


def run_weakly_supervised_classifier_training(loader_train_drugs, loader_train_controls, loader_val_drugs, loader_val_controls,
                                              model, optimizer, criterion, device, lr_scheduler=None, epochs=10, save_path=None):
    acc = 0
    for epoch in range(epochs):

        start = time.time()
        loss = 0
        acc = 0
        for batch_features, batch_labels in loader_train_drugs:
            # load it to the active device
            batch_features = batch_features.float().to(device)
            # reset the gradients back to zero
            optimizer.zero_grad()
            with torch.enable_grad():
                train_loss = 0

                # process drugs data
                outputs = model(batch_features)
                train_loss += criterion(outputs, batch_labels.to(device))
                true_negatives = (outputs.argmax(-1) == 0).cpu().float().detach().numpy()

                # process controls data
                batch_features, batch_labels = next(iter(loader_train_controls))
                outputs = model(batch_features.float().to(device))
                train_loss += criterion(outputs, batch_labels.to(device))
                true_positives = (outputs.argmax(-1) == 1).cpu().float().detach().numpy()

                # compute accumulated gradients
                train_loss.backward()
                # perform parameter update based on current gradients
                optimizer.step()
                # add the mini-batch training loss to epoch loss
                loss += train_loss.item()
                acc += (true_positives.sum() + true_negatives.sum()) / (len(true_positives) + len(true_negatives))

        # compute epoch training loss
        loss = loss / len(loader_train_drugs)
        # compute epoch training accuracy
        acc = acc / len(loader_train_drugs)

        val_acc = 0
        for batch_features, batch_labels in loader_val_drugs:
            # process drugs data
            batch_features = batch_features.float().to(device)
            outputs = model(batch_features)
            true_negatives = (outputs.argmax(-1) == 0).cpu().float().detach().numpy()
            # process controls data
            batch_features, batch_labels = next(iter(loader_val_controls))
            outputs = model(batch_features.float().to(device))
            true_positives = (outputs.argmax(-1) == 1).cpu().float().detach().numpy()

            val_acc += (true_positives.sum() + true_negatives.sum()) / (len(true_positives) + len(true_negatives))

        # compute epoch training accuracy
        val_acc = val_acc / len(loader_val_drugs)

        if save_path is not None:
            torch.save(model.state_dict(), save_path + 'dcl_at_{}.torch'.format(epoch+1))

        # update lr
        if lr_scheduler is not None:
            lr_scheduler.step()

        # display the epoch training loss
        print("epoch {}/{}: {} min, loss = {:.4f}, acc = {:.4f}, val_acc = {:.4f}".format(epoch + 1, epochs, int((time.time() - start) / 60), loss, acc, val_acc))

    return acc


if __name__ == "__main__":

    device = torch.device('cuda')

    path_to_cl_model = "D:\ETH\projects\pheno-ml\\res\\third\dcl_at_21.torch"
    cl = DeepClassifier().to(device)
    cl.load_state_dict(torch.load(path_to_cl_model, map_location=device))
    cl.eval()

    train_deep_classifier_weakly(20, trained_cl=cl, batch_size=64)
