import os, pandas, time, torch, numpy, itertools, seaborn
from matplotlib import pyplot
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision.io import read_image
from torchmetrics import Accuracy, Recall, Precision, Specificity
from PIL import Image
from tqdm import tqdm

from src.comparison import get_f_transform, get_image_encodings_of_cell_line, get_wells_of_drug_for_cell_line
from src.constants import cell_lines, drugs


class Classifier(nn.Module):

    def __init__(self):
        super().__init__()

        # plugs in after convolutional autoencoder -> needs flattening of the filters
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 2),
            nn.Softmax(dim=1)
        )

        print(self)
        print('number of parameters: {}\n'.format(self.count_parameters()))

    def forward(self, features):
        return self.model(features)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_classifier_with_pretrained_encoder(epochs, models, device=torch.device('cuda'), batch_size=256):

    path_to_images = '/Users/andreidm/ETH/projects/pheno-ml/data/cropped/training/single_class/'

    for model_name in models:

        save_path = '/Users/andreidm/ETH/projects/pheno-ml/res/comparison/classification/{}/'.format(model_name)

        transform = get_f_transform(model_name, device=device)

        all_drugs_codes = []
        all_controls_codes = []
        for cell_line in tqdm(cell_lines):

            all_drugs_wells = []
            for drug in drugs:
                max_conc_drug_wells, plate = get_wells_of_drug_for_cell_line(cell_line, drug)
                all_drugs_wells.extend([plate + '_' + well for well in max_conc_drug_wells])

            p1_dmso_wells, _ = get_wells_of_drug_for_cell_line(cell_line, 'DMSO', plate='P1')
            p2_dmso_wells, _ = get_wells_of_drug_for_cell_line(cell_line, 'DMSO', plate='P2')
            pbs_wells, plate = get_wells_of_drug_for_cell_line(cell_line, 'PBS')

            all_controls_wells = [
                *['P1_' + well for well in p1_dmso_wells],
                *['P2_' + well for well in p2_dmso_wells],
                *[plate + '_' + well for well in pbs_wells]
            ]

            drug_codes, _ = get_image_encodings_of_cell_line(path_to_images, cell_line, all_drugs_wells, transform=transform, last_n=15)
            control_codes, _ = get_image_encodings_of_cell_line(path_to_images, cell_line, all_controls_wells, transform=transform, last_n=23)

            all_drugs_codes.extend(drug_codes)
            all_controls_codes.extend(control_codes)

        print("train set: {} drugs, {} controls".format(len(all_drugs_codes), len(all_controls_codes)))

        # make datasets
        x_train = [*all_drugs_codes, *all_controls_codes]
        y_train = [*[1 for x in range(len(all_drugs_codes))], *[0 for x in range(len(all_controls_codes))]]
        train_dataset = TensorDataset(torch.Tensor(x_train), torch.LongTensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        model = Classifier().to(device)

        lrs = [0.1, 0.05, 0.01]
        ms = [0.9, 0.8, 0.7]
        wds = [1e-3, 1e-4, 1e-5]

        for lr in lrs:
            for m in ms:
                for wd in wds:

                    params = 'lr={},m={},wd={}'.format(lr, m, wd)
                    if not os.path.exists(save_path + '/' + params + '/'):
                        os.makedirs(save_path + '/' + params + '/')

                    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=m, weight_decay=wd)
                    criterion = nn.CrossEntropyLoss()

                    last_train_acc, last_val_acc = run_supervised_classifier_training(train_loader, model, optimizer, criterion, device,
                                                                                      epochs=epochs, save_to=save_path + '/' + params + '/')


def run_supervised_classifier_training(loader_train, model, optimizer, criterion, device,
                                       lr_scheduler=None, epochs=10, test_loader=None, save_to=""):

    f_acc = Accuracy().to(device)
    f_rec = Recall(num_classes=2, average='weighted').to(device)
    f_prec = Precision(num_classes=2, average='weighted').to(device)
    f_spec = Specificity(num_classes=2, average='macro').to(device)

    print("training started...")
    train_acc, val_acc = 0, -1
    train_acc_history = []
    train_rec_history = []
    train_prec_history = []
    train_spec_history = []
    train_f1_history = []
    for epoch in range(epochs):

        start = time.time()
        loss = 0
        train_acc = 0
        train_rec = 0
        train_prec = 0
        train_spec = 0
        for batch_features, batch_labels in loader_train:
            # load it to the active device
            batch_features = batch_features.float().to(device)
            batch_labels = batch_labels.to(device)
            # reset the gradients back to zero
            optimizer.zero_grad()
            with torch.enable_grad():
                train_loss = 0

                outputs = model(batch_features)
                train_loss += criterion(outputs, batch_labels)
                # compute accumulated gradients
                train_loss.backward()
                # perform parameter update based on current gradients
                optimizer.step()
                # add the mini-batch training loss to epoch loss
                loss += train_loss.item()

            train_acc += float(f_acc(outputs.argmax(-1), batch_labels))
            train_rec += float(f_rec(outputs.argmax(-1), batch_labels))
            train_prec += float(f_prec(outputs.argmax(-1), batch_labels))
            train_spec += float(f_spec(outputs.argmax(-1), batch_labels))

        # compute epoch training loss
        loss = loss / len(loader_train)
        # compute epoch metrics
        train_acc = train_acc / len(loader_train)
        train_rec = train_rec / len(loader_train)
        train_prec = train_prec / len(loader_train)
        train_spec = train_spec / len(loader_train)
        train_acc_history.append(train_acc)
        train_rec_history.append(train_rec)
        train_prec_history.append(train_prec)
        train_spec_history.append(train_spec)
        train_f1_history.append(2 * train_rec * train_prec / (train_prec + train_rec))

        torch.save(model.state_dict(), save_to + 'classifier_at_{}.torch'.format(epoch+1))

        # display the epoch training loss
        print("epoch {}/{}: {} sec, loss = {:.4f},\n"
              "acc = {:.4f}, rec = {:.4f}, prec = {:.4f}, spec = {:.4f}\n"
              .format(epoch + 1, epochs, int(time.time() - start), loss,
                      train_acc, train_rec, train_prec, train_spec))

    history = pandas.DataFrame({'epoch': [x + 1 for x in range(len(train_acc_history))],
                                'accuracy': train_acc_history,
                                'recall': train_rec_history,
                                'precision': train_rec_history,
                                'specificity': train_spec_history,
                                'f1': train_f1_history})

    history.to_csv(save_to + 'history.csv', index=False)

    return train_acc, val_acc


def collect_and_plot_classification_results(path_to_results='/Users/andreidm/ETH/projects/pheno-ml/res/comparison/classification/'):

    results = {
        'models': [], 'lrs': [], 'ms': [], 'wds': [],
        'epoch': [], 'accuracy': [], 'recall': [], 'precision': [], 'specificity': [], 'f1': []
    }

    for model in os.listdir(path_to_results):
        for param_set in os.listdir(path_to_results + model):
            data = pandas.read_csv(path_to_results + model + '/' + param_set + '/history.csv')
            best_f1_data = data.loc[data['f1'] == data['f1'].max(), :]

            results['lrs'].append(float(param_set.split(',')[0].split('=')[1]))
            results['ms'].append(float(param_set.split(',')[1].split('=')[1]))
            results['wds'].append(float(param_set.split(',')[2].split('=')[1]))
            results['epoch'].append(int(best_f1_data['epoch']))
            results['accuracy'].append(float(best_f1_data['accuracy']))
            results['recall'].append(float(best_f1_data['recall']))
            results['precision'].append(float(best_f1_data['precision']))
            results['specificity'].append(float(best_f1_data['specificity']))
            results['f1'].append(float(best_f1_data['f1']))
            if model == 'resnet50':
                results['models'].append('ResNet-50')
            else:
                results['models'].append('SwAV')

    results_df = pandas.DataFrame(results)

    i = 1
    seaborn.set()
    pyplot.figure(figsize=(12,3))
    for metric in ['accuracy', 'recall', 'precision', 'specificity', 'f1']:
        pyplot.subplot(1, 5, i)
        seaborn.boxplot(x='models', y=metric, data=results_df)
        pyplot.title(metric)
        i += 1
    pyplot.tight_layout()
    pyplot.show()


if __name__ == '__main__':
    pass