
import os, pandas, time, torch, numpy, uuid, seaborn, random, shutil, traceback
from PIL import Image
from matplotlib import pyplot
from byol_pytorch import BYOL
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize

from src.datasets import CustomImageDataset
from src.classifier import DeepClassifier


def get_byol_pars(im_size, randomize=True):

    if randomize:
        return dict(image_size=im_size,  # 256
                    hidden_layer='model.11',
                    projection_size=random.sample([64, 128, 256, 512, 1024, 2048, 4096], 1)[0],  # 256
                    projection_hidden_size=random.sample([512, 1024, 2048, 4096, 8192], 1)[0],  # 4096
                    augment_fn=None,
                    augment_fn2=None,
                    moving_average_decay=random.sample([0.8, 0.9, 0.99], 1)[0],
                    use_momentum=True)
    else:
        return dict(image_size=im_size,
                    hidden_layer='model.11',
                    projection_size=im_size,
                    # projection_hidden_size=2048,  # best
                    # projection_hidden_size=1024,  # second best
                    # projection_hidden_size=512,  # as good
                    projection_hidden_size=256,  #
                    augment_fn=None,
                    augment_fn2=None,
                    moving_average_decay=0.8,
                    use_momentum=True)


def generate_grid(grid_size, image_size, randomize=True):

    grid = {'id': [], 'byol': []}
    for _ in range(grid_size):
        grid['id'].append(str(uuid.uuid4())[:8])
        grid['byol'].append(get_byol_pars(image_size, randomize=randomize))

    return grid


def run_byol_training(model, epochs, batch_size, device, grid=None, save_path='D:\ETH\projects\morpho-learner\\res\\byol\\'):

    path_to_data = 'D:\ETH\projects\pheno-ml\data\cropped\\'
    image_size = 128

    if grid is None:
        grid_size = 1
        grid = generate_grid(grid_size, image_size, randomize=False)

    transform = lambda img: Resize(size=image_size)(img) / 255.
    training_data = CustomImageDataset(path_to_data, 0, transform=transform)
    data_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=)

    train(model, grid, epochs, data_loader, device, save_path)


def train(model, grid, epochs, data_loader, device, save_path):

    for i, id in enumerate(grid['id']):

        print(pandas.DataFrame(grid['byol'][i], index=['values'], columns=grid['byol'][i].keys()).T.to_string())
        print('training for set {} started'.format(id))
        if not os.path.exists(save_path + id):
            os.makedirs(save_path + id)

        try:
            learner = BYOL(model, **grid['byol'][i]).to(device)
            opt = torch.optim.Adam(learner.parameters(), lr=0.002)

            loss_history = []
            try:
                for epoch in range(epochs):
                    start = time.time()
                    epoch_loss = 0
                    for batch_features in data_loader:
                        images = batch_features[0].float().to(device)
                        loss = learner(images)
                        epoch_loss += loss.item()
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        learner.update_moving_average()  # update moving average of teacher encoder and teacher centers

                    epoch_loss = epoch_loss / len(data_loader)
                    loss_history.append(epoch_loss)
                    print("epoch {}: {} min, loss = {:.4f}".format(epoch + 1, int((time.time() - start) / 60),
                                                                   epoch_loss))
                    # save network
                    torch.save(model.state_dict(), save_path + id + '\\dcl+byol_at_{}.torch'.format(epoch))

                    if epoch >= 2:
                        if epoch_loss > loss_history[epoch - 1] > loss_history[epoch - 2] or epoch_loss > loss_history[0]:
                            # if loss grows, stop training
                            break
                        elif round(epoch_loss, 4) == round(loss_history[epoch - 1], 4):
                            # if loss doesn't fall, stop
                            break

                print('{}/{} completed'.format(i + 1, len(grid['id'])))
                save_history_and_parameters(loss_history, grid['byol'][i], save_path + id)

            except Exception as e:
                print('{}/{} failed with {}\n'.format(i + 1, len(grid['id']), e))
                save_history_and_parameters(loss_history, grid['byol'][i], save_path + id)
                # shutil.rmtree(save_path + id)
        except Exception as e:
            print('{}/{} failed building byol with {}\n'.format(i + 1, len(grid['id']), e))
            print(traceback.print_exc())
            # shutil.rmtree(save_path + id)


def save_history_and_parameters(loss_history, byol_pars, save_path):

    # save history
    history = pandas.DataFrame({'epoch': [x+1 for x in range(len(loss_history))], 'loss': loss_history})
    history.to_csv(save_path + '\\history.csv', index=False)

    # plot history
    seaborn.lineplot(data=history, x='epoch', y='loss')
    pyplot.grid()
    pyplot.savefig(save_path + '\\loss.png')
    pyplot.close()

    # save vit parameters
    pandas.DataFrame(byol_pars, index=['values'], columns=byol_pars.keys()).T \
        .to_csv(save_path + '\\byol_pars.csv', index=True)

    print('history and parameters saved\n')


if __name__ == "__main__":

    device = torch.device('cuda')

    path_to_cl_model = "D:\ETH\projects\pheno-ml\\res\\classifier\\fourth\dcl_at_8.torch"
    cl = DeepClassifier().to(device)
    cl.load_state_dict(torch.load(path_to_cl_model, map_location=device))
    cl.eval()

    run_byol_training(cl, 50, 64, device, save_path="D:\ETH\projects\pheno-ml\\res\\byol\\")


