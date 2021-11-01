import os, torch, numpy, time, traceback, pandas, seaborn
from matplotlib import pyplot
from torch import nn
from src.datasets import MultiCropDataset
from torch.utils.data import DataLoader
from byol_pytorch import BYOL

from torch import nn


class Autoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 128, (3,3), stride=(1,1), padding=(1,1)),  # 128 x 128 x 128
            nn.ReLU(True),
            nn.MaxPool2d(2),  # 128 x 64 x 64

            nn.Conv2d(128, 64, (3,3), stride=(1,1), padding=(1,1)),  # 64 x 64 x 64
            nn.ReLU(True),
            nn.MaxPool2d(2),  # 64 x 32 x 32

            nn.Conv2d(64, 32, (3,3), stride=(1,1), padding=(1,1)),  # 32 x 32 x 32
            nn.ReLU(True),

            nn.Conv2d(32, 4, (1,1), stride=(1,1), padding=(1,1)),  # 4 x 32 x 32
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(4, 32, (1,1), stride=(1,1), padding=(1,1)),  # 32 x 32 x 32
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),  # 32 x 64 x 64

            nn.ConvTranspose2d(32, 64, (3,3), stride=(1,1), padding=(1,1)),  # 64 x 64 x 64
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),  # 64 x 128 x 128

            nn.ConvTranspose2d(64, 128, (3,3), stride=(1,1), padding=(1,1)),  # 128 x 128 x 128
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 1, (3,3), stride=(1,1), padding=(1,1)),
            nn.Sigmoid()
        )
        print(self)
        print('number of parameters: {}\n'.format(self.count_parameters()))

    def forward(self, features):
        encoded = self.encoder(features)
        decoded = self.decoder(encoded)
        return decoded

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Autoencoder_v2(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 128, (3,3), stride=(1,1), padding=(1,1)),  # 128 x 128 x 128
            nn.ReLU(True),
            nn.MaxPool2d(2),  # 128 x 64 x 64

            nn.Conv2d(128, 64, (3,3), stride=(1,1), padding=(1,1)),  # 64 x 64 x 64
            nn.ReLU(True),
            nn.MaxPool2d(2),  # 64 x 32 x 32

            nn.Conv2d(64, 32, (3,3), stride=(1,1), padding=(1,1)),  # 32 x 32 x 32
            nn.ReLU(True),

            nn.Conv2d(32, 4, (3, 3), stride=(1,1), padding=(1,1)),  # 4 x 32 x 32
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(4, 32, (3, 3), stride=(1,1), padding=(1,1)),  # 32 x 32 x 32
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),  # 32 x 64 x 64

            nn.ConvTranspose2d(32, 64, (3,3), stride=(1,1), padding=(1,1)),  # 64 x 64 x 64
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),  # 64 x 128 x 128

            nn.ConvTranspose2d(64, 128, (3,3), stride=(1,1), padding=(1,1)),  # 128 x 128 x 128
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 1, (3,3), stride=(1,1), padding=(1,1)),
            nn.Sigmoid()
        )
        print(self)
        print('number of parameters: {}\n'.format(self.count_parameters()))

    def forward(self, features):
        encoded = self.encoder(features)
        decoded = self.decoder(encoded)
        return decoded

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Backbone(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 128, (3, 3), stride=(1, 1), padding=(1, 1)),  # 128 x 128 x 128
            nn.ReLU(True),
            nn.MaxPool2d(2),  # 128 x 64 x 64

            nn.Conv2d(128, 64, (3, 3), stride=(1, 1), padding=(1, 1)),  # 64 x 64 x 64
            nn.ReLU(True),
            nn.MaxPool2d(2),  # 64 x 32 x 32

            nn.Conv2d(64, 32, (3, 3), stride=(1, 1), padding=(1, 1)),  # 32 x 32 x 32
            nn.ReLU(True),

            nn.Conv2d(32, 8, (1, 1), stride=(1, 1), padding=(1, 1)),  # 8 x 32 x 32
            nn.Flatten()
        )

        print(self)
        print('number of parameters: {}\n'.format(self.count_parameters()))

    def forward(self, features):
        return self.model(features)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def define_cropping_strategies(crop_size):

    strategies = []

    # # ONE CROP
    # size_crops = [crop_size]
    # nmb_crops = [1]
    # min_scale_crops = [1]
    # max_scale_crops = [1]
    # sid = "_".join([str(x) for x in [*nmb_crops, *min_scale_crops, *max_scale_crops]])
    # strategy = (sid, size_crops, nmb_crops, min_scale_crops, max_scale_crops)
    # strategies.append(strategy)
    #
    # # TWO CROPS
    # size_crops = [crop_size]
    # nmb_crops = [2]
    # min_scale_crops = [0.5]
    # max_scale_crops = [0.75]
    # sid = "_".join([str(x) for x in [*nmb_crops, *min_scale_crops, *max_scale_crops]])
    # strategy = (sid, size_crops, nmb_crops, min_scale_crops, max_scale_crops)
    # strategies.append(strategy)
    #
    # size_crops = [crop_size, crop_size]
    # nmb_crops = [1, 1]
    # min_scale_crops = [1, 0.5]
    # max_scale_crops = [1, 0.75]
    # sid = "_".join([str(x) for x in [*nmb_crops, *min_scale_crops, *max_scale_crops]])
    # strategy = (sid, size_crops, nmb_crops, min_scale_crops, max_scale_crops)
    # strategies.append(strategy)
    #
    # # THREE CROPS
    # size_crops = [crop_size]
    # nmb_crops = [3]
    # min_scale_crops = [0.5]
    # max_scale_crops = [0.75]
    # sid = "_".join([str(x) for x in [*nmb_crops, *min_scale_crops, *max_scale_crops]])
    # strategy = (sid, size_crops, nmb_crops, min_scale_crops, max_scale_crops)
    # strategies.append(strategy)
    #
    # size_crops = [crop_size, crop_size]
    # nmb_crops = [1, 2]
    # min_scale_crops = [1, 0.5]
    # max_scale_crops = [1, 0.75]
    # sid = "_".join([str(x) for x in [*nmb_crops, *min_scale_crops, *max_scale_crops]])
    # strategy = (sid, size_crops, nmb_crops, min_scale_crops, max_scale_crops)
    # strategies.append(strategy)
    #
    # size_crops = [crop_size, crop_size, crop_size]
    # nmb_crops = [1, 1, 1]
    # min_scale_crops = [1, 0.5, 0.5]
    # max_scale_crops = [1, 0.75, 0.5]
    # sid = "_".join([str(x) for x in [*nmb_crops, *min_scale_crops, *max_scale_crops]])
    # strategy = (sid, size_crops, nmb_crops, min_scale_crops, max_scale_crops)
    # strategies.append(strategy)
    #
    # # FOUR CROPS
    # size_crops = [crop_size]
    # nmb_crops = [4]
    # min_scale_crops = [0.5]
    # max_scale_crops = [0.75]
    # sid = "_".join([str(x) for x in [*nmb_crops, *min_scale_crops, *max_scale_crops]])
    # strategy = (sid, size_crops, nmb_crops, min_scale_crops, max_scale_crops)
    # strategies.append(strategy)
    #
    # size_crops = [crop_size, crop_size]
    # nmb_crops = [1, 3]
    # min_scale_crops = [1, 0.5]
    # max_scale_crops = [1, 0.75]
    # sid = "_".join([str(x) for x in [*nmb_crops, *min_scale_crops, *max_scale_crops]])
    # strategy = (sid, size_crops, nmb_crops, min_scale_crops, max_scale_crops)
    # strategies.append(strategy)
    #
    # size_crops = [crop_size, crop_size, crop_size]
    # nmb_crops = [1, 1, 2]
    # min_scale_crops = [1, 0.5, 0.5]
    # max_scale_crops = [1, 0.75, 0.5]
    # sid = "_".join([str(x) for x in [*nmb_crops, *min_scale_crops, *max_scale_crops]])
    # strategy = (sid, size_crops, nmb_crops, min_scale_crops, max_scale_crops)
    # strategies.append(strategy)
    #
    # # FIVE CROPS
    # size_crops = [crop_size, crop_size]
    # nmb_crops = [1, 4]
    # min_scale_crops = [1, 0.5]
    # max_scale_crops = [1, 0.75]
    # sid = "_".join([str(x) for x in [*nmb_crops, *min_scale_crops, *max_scale_crops]])
    # strategy = (sid, size_crops, nmb_crops, min_scale_crops, max_scale_crops)
    # strategies.append(strategy)
    #
    # size_crops = [crop_size, crop_size, crop_size]
    # nmb_crops = [1, 2, 2]
    # min_scale_crops = [1, 0.5, 0.5]
    # max_scale_crops = [1, 0.75, 0.5]
    # sid = "_".join([str(x) for x in [*nmb_crops, *min_scale_crops, *max_scale_crops]])
    # strategy = (sid, size_crops, nmb_crops, min_scale_crops, max_scale_crops)
    # strategies.append(strategy)

    size_crops = [crop_size, crop_size, crop_size]
    nmb_crops = [1, 1, 3]
    min_scale_crops = [1, 0.5, 0.5]
    max_scale_crops = [1, 0.75, 0.5]
    sid = "_".join([str(x) for x in [*nmb_crops, *min_scale_crops, *max_scale_crops]])
    strategy = (sid, size_crops, nmb_crops, min_scale_crops, max_scale_crops)
    strategies.append(strategy)

    return strategies


def run_training_for_128x128_crops(epochs, data_loader, device=torch.device('cuda'), run_id=""):

    save_path = 'D:\ETH\projects\pheno-ml\\res\\byol\\{}'.format(run_id)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = Backbone().to(device)
    train(model, epochs, data_loader, device, save_path)


def train(model, epochs, data_loader, device, save_path):

    try:
        learner = BYOL(model,
                       image_size=128, hidden_layer=-1,
                       projection_size=128, projection_hidden_size=2048,
                       moving_average_decay=0.8, use_momentum=True).to(device)

        optimizer = torch.optim.Adam(learner.parameters(), lr=0.005)
        scheduler = None

        loss_history = []
        try:
            for epoch in range(epochs):
                start = time.time()
                epoch_loss = 0
                n_crops = 1
                for batch in data_loader:
                    n_crops = len(batch)
                    for crops, _ in batch:
                        crops = crops.float().to(device)
                        loss = learner(crops)
                        epoch_loss += loss.item()
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        learner.update_moving_average()  # update moving average of teacher encoder and teacher centers

                epoch_loss = epoch_loss / len(data_loader) / n_crops
                loss_history.append(epoch_loss)
                print("epoch {}: {} min, loss = {:.4f}".format(epoch + 1, int((time.time() - start) / 60), epoch_loss))

                # update lr
                if scheduler is not None:
                    scheduler.step()

                # save network
                torch.save(model.state_dict(), save_path + '\\byol_at_{}.torch'.format(epoch+1))

                if epoch >= 2:
                    if epoch_loss > loss_history[epoch - 1] > loss_history[epoch - 2] or epoch_loss > loss_history[0]:
                        # if loss grows, stop training
                        break
                    elif round(epoch_loss, 4) == round(loss_history[epoch - 1], 4):
                        # if loss doesn't fall, stop
                        break

            save_history(loss_history, save_path)

        except Exception as e:
            print('failed with {}\n'.format(e))
    except Exception as e:
        print('failed building byol with {}\n'.format(e))
        print(traceback.print_exc())


def save_history(loss_history, save_path):

    # save history
    history = pandas.DataFrame({'epoch': [x+1 for x in range(len(loss_history))], 'loss': loss_history})
    history.to_csv(save_path + '\\history.csv', index=False)

    # plot history
    seaborn.lineplot(data=history, x='epoch', y='loss')
    pyplot.grid()
    pyplot.savefig(save_path + '\\loss_min={}.png'.format(round(min(loss_history), 4)))
    pyplot.close()
    print('history saved\n')


def train_autoencoder(epochs, data_loader_train, trained_ae=None, device=torch.device('cuda'), run_id=""):

    save_path = 'D:\ETH\projects\pheno-ml\\res\\ae\\{}\\'.format(run_id)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if trained_ae is not None:
        model = trained_ae
    else:
        # model = Autoencoder().to(device)
        model = Autoencoder_v2().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    rec_loss, val_rec_loss = run_autoencoder_training(data_loader_train, model, optimizer, criterion, device,
                                                      epochs=epochs, save_path=save_path)

    # save history
    history = pandas.DataFrame({'epoch': [x + 1 for x in range(len(rec_loss))], 'loss': rec_loss})
    history.to_csv(save_path + '\\history.csv', index=False)
    # save reconstruction
    plot_reconstruction(data_loader_train, model, save_to=save_path, n_images=10)


def run_autoencoder_training(data_loader_train, model, optimizer, criterion, device,
                             lr_scheduler=None, epochs=10, save_path="..\\res\\"):

    loss_history = []
    val_loss_history = []
    for epoch in range(epochs):

        start = time.time()
        loss = 0
        n_crops = 1
        for batch in data_loader_train:

            n_crops = len(batch)
            for crops, _ in batch:
                # load it to the active device
                crops = crops.float().to(device)
                # reset the gradients back to zero
                optimizer.zero_grad()
                # compute reconstructions
                outputs = model(crops)
                # compute training reconstruction loss
                train_loss = criterion(outputs, crops)
                # compute accumulated gradients
                train_loss.backward()
                # perform parameter update based on current gradients
                optimizer.step()
                # add the mini-batch training loss to epoch loss
                loss += train_loss.item()

        # compute the epoch training loss
        loss = loss / len(data_loader_train) / n_crops
        loss_history.append(loss)

        # update lr
        if lr_scheduler is not None:
            lr_scheduler.step()

        # display the epoch training loss
        print("epoch {}/{}: {} min, loss = {:.4f}".format(epoch + 1, epochs, int((time.time() - start) / 60), loss))
        # save model
        torch.save(model.state_dict(), save_path + 'autoencoder_at_{}.torch'.format(epoch+1))

    return loss_history, val_loss_history


def plot_reconstruction(data_loader, trained_model, save_to='res/', n_images=10):

    for i in range(n_images):
        batch = next(iter(data_loader))
        crops, _ = batch[0]
        img = crops.squeeze()[0]
        img_tensor = torch.Tensor(numpy.expand_dims(crops[0], axis=0))
        rec = trained_model(img_tensor.cuda())

        pyplot.figure()
        pyplot.subplot(121)
        pyplot.imshow(img, cmap="gray")
        pyplot.title("original")
        pyplot.subplot(122)
        pyplot.imshow(rec.cpu().detach().numpy()[0][0], cmap="gray")
        pyplot.title("reconstruction")

        if save_to is not None:
            if not os.path.exists(save_to + 'recs/'):
                os.makedirs(save_to + 'recs/')
            pyplot.savefig(save_to + 'recs/{}.pdf'.format(i))
        else:
            pyplot.show()
    pyplot.close('all')


if __name__ == "__main__":

    path_to_data = "D:\ETH\projects\pheno-ml\data\\full\\"
    crop_size = 128
    epochs = 5
    batch_size = 64
    train_size = 250000
    device = torch.device('cuda')

    strategies = define_cropping_strategies(crop_size)

    for cropping_id, *cropping_strategy in strategies:

        train_multi_crop = MultiCropDataset(path_to_data, *cropping_strategy, no_aug=False, size_dataset=train_size)
        print('training data:', train_multi_crop.__len__())
        data_loader_train = DataLoader(train_multi_crop, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        train_autoencoder(epochs, data_loader_train, device=device, run_id=cropping_id + '_v2')

