import os

import torch, time, traceback, pandas, seaborn
from matplotlib import pyplot
from torch import nn
from src.datasets import MultiCropDataset
from torch.utils.data import DataLoader
from byol_pytorch import BYOL


class DeepClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(1, 64, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 32, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 16, (3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(True),

            nn.Conv2d(16, 8, (3, 3), stride=(1, 1), padding=(1, 1)),  # 8 x 16 x 16

            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.LeakyReLU(True),
            nn.Linear(256, 2),
            nn.Softmax(dim=1)
        )

        print(self)
        print('number of parameters: {}\n'.format(self.count_parameters()))

    def forward(self, features):
        return self.model(features)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def define_cropping_strategies(crop_size):

    strategies = []

    # ONE CROP
    size_crops = [crop_size]
    nmb_crops = [1]
    min_scale_crops = [0.5]
    max_scale_crops = [0.75]
    sid = "_".join([str(x) for x in [*nmb_crops, *min_scale_crops, *max_scale_crops]])
    strategy = (sid, size_crops, nmb_crops, min_scale_crops, max_scale_crops)
    strategies.append(strategy)

    # TWO CROPS
    size_crops = [crop_size]
    nmb_crops = [2]
    min_scale_crops = [0.5]
    max_scale_crops = [0.75]
    sid = "_".join([str(x) for x in [*nmb_crops, *min_scale_crops, *max_scale_crops]])
    strategy = (sid, size_crops, nmb_crops, min_scale_crops, max_scale_crops)
    strategies.append(strategy)

    size_crops = [crop_size, crop_size]
    nmb_crops = [1, 1]
    min_scale_crops = [1, 0.25]
    max_scale_crops = [1, 0.5]
    sid = "_".join([str(x) for x in [*nmb_crops, *min_scale_crops, *max_scale_crops]])
    strategy = (sid, size_crops, nmb_crops, min_scale_crops, max_scale_crops)
    strategies.append(strategy)

    size_crops = [crop_size, crop_size]
    nmb_crops = [1, 1]
    min_scale_crops = [1, 0.5]
    max_scale_crops = [1, 0.75]
    sid = "_".join([str(x) for x in [*nmb_crops, *min_scale_crops, *max_scale_crops]])
    strategy = (sid, size_crops, nmb_crops, min_scale_crops, max_scale_crops)
    strategies.append(strategy)

    # THREE CROPS
    size_crops = [crop_size]
    nmb_crops = [3]
    min_scale_crops = [0.5]
    max_scale_crops = [0.75]
    sid = "_".join([str(x) for x in [*nmb_crops, *min_scale_crops, *max_scale_crops]])
    strategy = (sid, size_crops, nmb_crops, min_scale_crops, max_scale_crops)
    strategies.append(strategy)

    size_crops = [crop_size, crop_size]
    nmb_crops = [2, 1]
    min_scale_crops = [0.5, 0.25]
    max_scale_crops = [0.75, 0.5]
    sid = "_".join([str(x) for x in [*nmb_crops, *min_scale_crops, *max_scale_crops]])
    strategy = (sid, size_crops, nmb_crops, min_scale_crops, max_scale_crops)
    strategies.append(strategy)

    size_crops = [crop_size, crop_size]
    nmb_crops = [1, 2]
    min_scale_crops = [1, 0.25]
    max_scale_crops = [1, 0.5]
    sid = "_".join([str(x) for x in [*nmb_crops, *min_scale_crops, *max_scale_crops]])
    strategy = (sid, size_crops, nmb_crops, min_scale_crops, max_scale_crops)
    strategies.append(strategy)

    size_crops = [crop_size, crop_size, crop_size]
    nmb_crops = [1, 1, 1]
    min_scale_crops = [0.5, 0.25, 0.25]
    max_scale_crops = [0.75, 0.5, 0.25]
    sid = "_".join([str(x) for x in [*nmb_crops, *min_scale_crops, *max_scale_crops]])
    strategy = (sid, size_crops, nmb_crops, min_scale_crops, max_scale_crops)
    strategies.append(strategy)

    # FOUR CROPS
    size_crops = [crop_size]
    nmb_crops = [4]
    min_scale_crops = [0.25]
    max_scale_crops = [0.5]
    sid = "_".join([str(x) for x in [*nmb_crops, *min_scale_crops, *max_scale_crops]])
    strategy = (sid, size_crops, nmb_crops, min_scale_crops, max_scale_crops)
    strategies.append(strategy)

    size_crops = [crop_size, crop_size, crop_size]
    nmb_crops = [1, 1, 2]
    min_scale_crops = [1, 0.5, 0.25]
    max_scale_crops = [1, 0.75, 0.5]
    sid = "_".join([str(x) for x in [*nmb_crops, *min_scale_crops, *max_scale_crops]])
    strategy = (sid, size_crops, nmb_crops, min_scale_crops, max_scale_crops)
    strategies.append(strategy)

    size_crops = [crop_size, crop_size]
    nmb_crops = [2, 2]
    min_scale_crops = [0.5, 0.25]
    max_scale_crops = [0.75, 0.5]
    sid = "_".join([str(x) for x in [*nmb_crops, *min_scale_crops, *max_scale_crops]])
    strategy = (sid, size_crops, nmb_crops, min_scale_crops, max_scale_crops)
    strategies.append(strategy)

    size_crops = [crop_size, crop_size]
    nmb_crops = [1, 3]
    min_scale_crops = [0.5, 0.25]
    max_scale_crops = [0.75, 0.25]
    sid = "_".join([str(x) for x in [*nmb_crops, *min_scale_crops, *max_scale_crops]])
    strategy = (sid, size_crops, nmb_crops, min_scale_crops, max_scale_crops)
    strategies.append(strategy)

    size_crops = [crop_size, crop_size, crop_size, crop_size]
    nmb_crops = [1, 1, 1, 1]
    min_scale_crops = [1, 0.5, 0.25, 0.25]
    max_scale_crops = [1, 0.75, 0.5, 0.25]
    sid = "_".join([str(x) for x in [*nmb_crops, *min_scale_crops, *max_scale_crops]])
    strategy = (sid, size_crops, nmb_crops, min_scale_crops, max_scale_crops)
    strategies.append(strategy)

    # FIVE CROPS
    size_crops = [crop_size]
    nmb_crops = [5]
    min_scale_crops = [0.25]
    max_scale_crops = [0.5]
    sid = "_".join([str(x) for x in [*nmb_crops, *min_scale_crops, *max_scale_crops]])
    strategy = (sid, size_crops, nmb_crops, min_scale_crops, max_scale_crops)
    strategies.append(strategy)

    size_crops = [crop_size]
    nmb_crops = [5]
    min_scale_crops = [0.25]
    max_scale_crops = [0.25]
    sid = "_".join([str(x) for x in [*nmb_crops, *min_scale_crops, *max_scale_crops]])
    strategy = (sid, size_crops, nmb_crops, min_scale_crops, max_scale_crops)
    strategies.append(strategy)

    size_crops = [crop_size, crop_size, crop_size]
    nmb_crops = [1, 2, 2]
    min_scale_crops = [1, 0.5, 0.25]
    max_scale_crops = [1, 0.75, 0.5]
    sid = "_".join([str(x) for x in [*nmb_crops, *min_scale_crops, *max_scale_crops]])
    strategy = (sid, size_crops, nmb_crops, min_scale_crops, max_scale_crops)
    strategies.append(strategy)

    size_crops = [crop_size, crop_size, crop_size]
    nmb_crops = [1, 2, 2]
    min_scale_crops = [0.5, 0.25, 0.25]
    max_scale_crops = [0.75, 0.5, 0.25]
    sid = "_".join([str(x) for x in [*nmb_crops, *min_scale_crops, *max_scale_crops]])
    strategy = (sid, size_crops, nmb_crops, min_scale_crops, max_scale_crops)
    strategies.append(strategy)

    size_crops = [crop_size, crop_size]
    nmb_crops = [1, 4]
    min_scale_crops = [0.5, 0.25]
    max_scale_crops = [0.75, 0.25]
    sid = "_".join([str(x) for x in [*nmb_crops, *min_scale_crops, *max_scale_crops]])
    strategy = (sid, size_crops, nmb_crops, min_scale_crops, max_scale_crops)
    strategies.append(strategy)

    size_crops = [crop_size, crop_size]
    nmb_crops = [2, 3]
    min_scale_crops = [0.25, 0.25]
    max_scale_crops = [0.5, 0.25]
    sid = "_".join([str(x) for x in [*nmb_crops, *min_scale_crops, *max_scale_crops]])
    strategy = (sid, size_crops, nmb_crops, min_scale_crops, max_scale_crops)
    strategies.append(strategy)

    size_crops = [crop_size, crop_size, crop_size, crop_size]
    nmb_crops = [1, 1, 1, 3]
    min_scale_crops = [1, 0.5, 0.25, 0.25]
    max_scale_crops = [1, 0.75, 0.5, 0.25]
    sid = "_".join([str(x) for x in [*nmb_crops, *min_scale_crops, *max_scale_crops]])
    strategy = (sid, size_crops, nmb_crops, min_scale_crops, max_scale_crops)
    strategies.append(strategy)

    return strategies


def run_training_for_64x64_cuts(epochs, data_loader, device=torch.device('cuda'), run_id=""):

    save_path = 'D:\ETH\projects\pheno-ml\\res\\byol\\{}'.format(run_id)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model = DeepClassifier().to(device)
    train(model, epochs, data_loader, device, save_path)


def train(model, epochs, data_loader, device, save_path):

    try:
        learner = BYOL(model,
                       image_size=64, hidden_layer='model.9',
                       projection_size=64, projection_hidden_size=2048,
                       moving_average_decay=0.8, use_momentum=True).to(device)

        optimizer = torch.optim.Adam(learner.parameters(), lr=0.0001)
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


if __name__ == "__main__":

    path_to_data = "D:\ETH\projects\pheno-ml\data\\full\\"
    crop_size = 64
    epochs = 50
    batch_size = 512
    train_size = -1

    strategies = define_cropping_strategies(crop_size)

    for cropping_id, *cropping_strategy in strategies:

        train_multi_crop = MultiCropDataset(path_to_data, *cropping_strategy, no_aug=False, size_dataset=train_size)
        print('training data:', train_multi_crop.__len__())
        data_loader_train = DataLoader(train_multi_crop, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        run_training_for_64x64_cuts(epochs, data_loader_train, device=torch.device('cuda'), run_id=cropping_id)
