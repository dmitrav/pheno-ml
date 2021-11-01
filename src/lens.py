
import torch, os, time, pandas
from torch import nn, optim
from tqdm import tqdm
from torchvision.io import read_image
from sklearn.model_selection import train_test_split
from torchmetrics import Recall, Accuracy, Specificity, Precision
from torch.utils.data import DataLoader, TensorDataset

from src.constants import drugs, cell_lines, controls
from src.comparison import get_f_transform
from src.comparison import get_wells_of_drug_for_cell_line


class DrugClassifier(nn.Module):

    def __init__(self, in_dim=2048, out_dim=2):
        super().__init__()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, out_dim),
            nn.Softmax(dim=1)
        )

        print(self)
        print('number of parameters: {}\n'.format(self.count_parameters()))

    def forward(self, features):
        return self.model(features)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_codes_and_labels(path, cell_line, drugs_wells, drugs_labels, transform, device=torch.device('cuda'), last_n=10):
    """ Gets the last n time points of each drug. Returns codes and labels. """

    filenames = []
    for file in os.listdir(path):
        if file.split('_')[0] == cell_line:
            key = file.split('_')[2] + '_' + file.split('_')[3]
            if key in drugs_wells:
                filenames.append(
                    (file, drugs_labels[drugs_wells.index(key)])
                )

    filenames = sorted(filenames, key=lambda x: x[0])
    i = 0
    latest_timepoints_filenames = []
    while i < len(filenames)-1:
        if filenames[i][0].split('_')[3] != filenames[i+1][0].split('_')[3]:
            latest_timepoints_filenames.extend(filenames[i-last_n+1:i+1])
        i += 1
    latest_timepoints_filenames.extend(filenames[-last_n:])
    filenames = latest_timepoints_filenames

    encodings = []
    labels = []
    for file, label in filenames:
        img = read_image(path + file)
        img_encoded = transform(img.to(device))
        encodings.append(img_encoded)
        labels.append(label)

    return encodings, labels


def collect_data_and_split(path_to_data, save_path):

    transform = get_f_transform('trained_ae_v2', device=device)
    drugs_and_controls = [*controls, *drugs]

    all_codes = []
    all_labels = []
    for cell_line in tqdm(cell_lines):

        wells = []
        labels = []
        for i in range(len(drugs_and_controls)):
            max_conc_wells, plate = get_wells_of_drug_for_cell_line(cell_line, drugs_and_controls[i])
            if len(max_conc_wells) > 4:
                max_conc_wells = max_conc_wells[:4]
            wells.extend([plate + '_' + well for well in max_conc_wells])
            labels.extend([i for well in max_conc_wells])

        codes, labels = get_codes_and_labels(path_to_data, cell_line, wells, labels, device=device, transform=transform,
                                             last_n=30)

        all_codes.extend(codes)
        all_labels.extend(labels)

    X_train, X_test, y_train, y_test = train_test_split(all_codes, all_labels, test_size=0.1, random_state=42)
    print("train set: {}".format(len(y_train)))
    print("test set: {}".format(len(y_test)))

    pandas.DataFrame(X_train).to_csv(save_path + 'data/X_train.csv', index=False)
    pandas.DataFrame(X_test).to_csv(save_path + 'data/X_test.csv', index=False)
    pandas.DataFrame(y_train).to_csv(save_path + 'data/y_train.csv', index=False)
    pandas.DataFrame(y_test).to_csv(save_path + 'data/y_test.csv', index=False)


def train_drug_classifier_alone(path_to_data, epochs, uid='', device=torch.device('cuda')):

    # save_path = 'D:\ETH\projects\pheno-ml\\res\\drug_classifier\\{}\\'.format(uid)
    save_path = '/Users/andreidm/ETH/projects/pheno-ml/res/drug_classifier/{}/'.format(uid)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    X_train = pandas.read_csv(path_to_data + 'X_train.csv').values
    X_test = pandas.read_csv(path_to_data + 'X_test.csv').values
    y_train = pandas.read_csv(path_to_data + 'y_train.csv').values.reshape(-1)
    y_test = pandas.read_csv(path_to_data + 'y_test.csv').values.reshape(-1)

    # make datasets and loaders
    train_dataset = TensorDataset(torch.Tensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.LongTensor(y_test))

    for lr in [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]:
        for bs in [256, 512, 1024, 2048]:

            model = DrugClassifier(in_dim=4624, out_dim=33).to(device)

            train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=True)
            test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=True, drop_last=True)

            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            last_train_acc, last_val_acc = run_supervised_classifier_training(train_loader, test_loader, model, optimizer,
                                                                              criterion, device,
                                                                              epochs=epochs,
                                                                              save_to=save_path + 'lr={},bs={}/'.format(lr, bs))


def train_drug_classifier_with_lens(path_to_data, epochs,  device=torch.device('cuda')):
    pass


def run_supervised_classifier_training(loader_train, loader_test, model, optimizer, criterion, device,
                                       lr_scheduler=None, epochs=10, save_to=""):
    if not os.path.exists(save_to):
        os.makedirs(save_to)

    f_acc = Accuracy(top_k=3).to(device)
    f_rec = Recall(top_k=3).to(device)
    f_prec = Precision(top_k=3).to(device)
    f_spec = Specificity(top_k=3).to(device)

    print("training started...")
    train_acc, test_acc = 0, -1
    train_acc_history, train_rec_history, train_prec_history, train_spec_history, train_f1_history = [], [], [], [], []
    test_acc_history, test_rec_history, test_prec_history, test_spec_history, test_f1_history = [], [], [], [], []
    for epoch in range(epochs):

        start = time.time()
        loss = 0
        train_acc, train_rec, train_prec, train_spec = 0, 0, 0, 0
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

            train_acc += float(f_acc(outputs, batch_labels))
            train_rec += float(f_rec(outputs, batch_labels))
            train_prec += float(f_prec(outputs, batch_labels))
            train_spec += float(f_spec(outputs, batch_labels))

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

        # TEST METRICS
        test_acc, test_rec, test_prec, test_spec = 0, 0, 0, 0
        for batch_features, batch_labels in loader_test:
            # load it to the active device
            batch_features = batch_features.float().to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_features)

            test_acc += float(f_acc(outputs, batch_labels))
            test_rec += float(f_rec(outputs, batch_labels))
            test_prec += float(f_prec(outputs, batch_labels))
            test_spec += float(f_spec(outputs, batch_labels))

        # compute epoch metrics
        test_acc = test_acc / len(loader_test)
        test_rec = test_rec / len(loader_test)
        test_prec = test_prec / len(loader_test)
        test_spec = test_spec / len(loader_test)

        test_acc_history.append(test_acc)
        test_rec_history.append(test_rec)
        test_prec_history.append(test_prec)
        test_spec_history.append(test_spec)
        test_f1_history.append(2 * test_rec * test_prec / (test_prec + test_rec))

        torch.save(model.state_dict(), save_to + 'classifier_at_{}.torch'.format(epoch+1))

        # display the epoch training loss
        print("epoch {}/{}: {} sec, loss = {:.4f},\n"
              "acc = {:.4f}, rec = {:.4f}, prec = {:.4f}, spec = {:.4f}\n"
              "t_acc = {:.4f}, t_rec = {:.4f}, t_prec = {:.4f}, t_spec = {:.4f}\n"
              .format(epoch + 1, epochs, int(time.time() - start), loss,
                      train_acc, train_rec, train_prec, train_spec,
                      test_acc, test_rec, test_prec, test_spec))

    history = pandas.DataFrame({'epoch': [x + 1 for x in range(len(train_acc_history))],
                                'acc': train_acc_history, 'recall': train_rec_history, 'precision': train_rec_history,
                                'specificity': train_spec_history, 'f1': train_f1_history,
                                't_acc': test_acc_history, 't_recall': test_rec_history, 't_precision': test_rec_history,
                                't_specificity': test_spec_history, 't_f1': test_f1_history})

    history.to_csv(save_to + 'history.csv', index=False)

    return train_acc, test_acc


if __name__ == "__main__":

    path_to_data = '/Users/andreidm/ETH/projects/pheno-ml/res/drug_classifier/data/'

    device = torch.device('cpu')

    uid = 'without_lens'

    # classification of drugs vs controls
    train_drug_classifier_alone(path_to_data, 30, uid=uid, device=device)