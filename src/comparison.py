
import os, pandas, torch, numpy, random, umap, time, seaborn, itertools
from hdbscan import HDBSCAN
from torch.nn import Sequential
from torchvision.io import read_image
from torchvision.transforms import Resize, Grayscale, ToPILImage, ToTensor
from scipy.spatial.distance import pdist
from sklearn import metrics
from tqdm import tqdm
from matplotlib import pyplot
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Accuracy, Recall, Precision, Specificity
from tensorflow.keras.models import Model

from src.autoencoder import get_trained_autoencoder
from src.self_supervised import Autoencoder
from src.constants import cell_lines, drugs
from src import pretrained


class Classifier(nn.Module):

    def __init__(self, in_dim=2048):
        super().__init__()

        # plugs in after convolutional autoencoder -> needs flattening of the filters
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, 256),
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


def train_classifiers_with_pretrained_encoder_and_save_results(path_to_images, epochs, models, uid='', device=torch.device('cuda'), batch_size=256):

    save_path = 'D:\ETH\projects\pheno-ml\\res\\comparison\\classification\\'

    for model_name in models:

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

            drug_codes, _ = get_image_encodings_of_cell_line(path_to_images, cell_line, all_drugs_wells, device=device, transform=transform, last_n=15)
            control_codes, _ = get_image_encodings_of_cell_line(path_to_images, cell_line, all_controls_wells, device=device, transform=transform, last_n=23)

            all_drugs_codes.extend(drug_codes)
            all_controls_codes.extend(control_codes)

        print("train set: {} drugs, {} controls".format(len(all_drugs_codes), len(all_controls_codes)))

        # make datasets
        x_train = [*all_drugs_codes, *all_controls_codes]
        y_train = [*[1 for x in range(len(all_drugs_codes))], *[0 for x in range(len(all_controls_codes))]]
        train_dataset = TensorDataset(torch.Tensor(x_train), torch.LongTensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        if model_name == 'resnet50' or model_name == 'swav_resnet50' or model_name == 'dino_resnet50':
            model = Classifier().to(device)
        elif model_name == 'dino_vit':
            model = Classifier(in_dim=768).to(device)
        elif model_name == 'trained_ae':
            # old tensorflow model
            model = Classifier(in_dim=4096).to(device)
        else:
            # new pytorch models
            model = Classifier(in_dim=4624).to(device)

        lrs = [0.1, 0.05, 0.01]
        ms = [0.9, 0.8, 0.7]
        wds = [1e-3, 1e-4, 1e-5]

        for lr in lrs:
            for m in ms:
                for wd in wds:

                    params = 'lr={},m={},wd={}'.format(lr, m, wd)
                    if not os.path.exists(save_path + '{}/{}/'.format(model_name, params)):
                        os.makedirs(save_path + '{}/{}/'.format(model_name, params))

                    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=m, weight_decay=wd)
                    criterion = nn.CrossEntropyLoss()

                    last_train_acc, last_val_acc = run_supervised_classifier_training(train_loader, model, optimizer, criterion, device,
                                                                                      epochs=epochs, save_to=save_path + '{}/{}/'.format(model_name, params))

    # collect and save results
    results = {
        'method': [], 'lrs': [], 'ms': [], 'wds': [],
        'epoch': [], 'accuracy': [], 'recall': [], 'precision': [], 'specificity': [], 'f1': []
    }

    for model_name in models:
        for param_set in os.listdir(save_path + model_name):
            data = pandas.read_csv(save_path + '{}/{}/history.csv'.format(model_name, param_set))
            best_f1_data = data.loc[data['f1'] == data['f1'].max(), :]

            results['method'].append(model_name)
            results['lrs'].append(float(param_set.split(',')[0].split('=')[1]))
            results['ms'].append(float(param_set.split(',')[1].split('=')[1]))
            results['wds'].append(float(param_set.split(',')[2].split('=')[1]))
            results['epoch'].append(int(best_f1_data['epoch']))
            results['accuracy'].append(float(best_f1_data['accuracy']))
            results['recall'].append(float(best_f1_data['recall']))
            results['precision'].append(float(best_f1_data['precision']))
            results['specificity'].append(float(best_f1_data['specificity']))
            results['f1'].append(float(best_f1_data['f1']))

    results_df = pandas.DataFrame(results)
    results_df.to_csv(save_path + 'classification{}.csv'.format(uid), index=False)


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


def plot_classification_results(path_to_results='/Users/andreidm/ETH/projects/pheno-ml/res/comparison/classification/classification.csv'):

    results = pandas.read_csv(path_to_results)

    results.loc[results['method'] == 'resnet50', 'method'] = 'RN-50\n(pretrained)'
    results.loc[results['method'] == 'swav_resnet50', 'method'] = 'RN-50+SwAV\n(pretrained)'
    results.loc[results['method'] == 'dino_resnet50', 'method'] = 'RN-50+DINO\n(pretrained)'
    results.loc[results['method'] == 'dino_vit', 'method'] = 'ViT+DINO\n(pretrained)'
    results.loc[results['method'] == 'trained_ae_v1', 'method'] = 'ConvAE v1\n(trained)'
    results.loc[results['method'] == 'trained_ae_v2', 'method'] = 'ConvAE v2\n(trained)'

    i = 1
    seaborn.set()
    pyplot.figure(figsize=(12,3))
    pyplot.suptitle('Comparison of drug-control classification')
    for metric in ['accuracy', 'recall', 'specificity', 'f1']:
        pyplot.subplot(1, 4, i)
        seaborn.boxplot(x='method', y=metric, data=results)
        pyplot.title(metric)
        pyplot.xticks(rotation=45, fontsize=6)
        i += 1
    pyplot.tight_layout()
    pyplot.show()


def get_image_encodings_from_path(path, common_image_ids, transform, device=torch.device('cpu'), n=None, randomize=True):

    # get filenames to retrieve image ids
    if isinstance(common_image_ids, str):
        filenames = [f for f in os.listdir(path) if common_image_ids in f]

    elif isinstance(common_image_ids, list):
        # filenames must contain all ids
        filenames = [f for f in os.listdir(path) if sum([1 for x in common_image_ids if x in f]) == len(common_image_ids)]
    else:
        raise ValueError('Unknown common image ids: {}'.format(common_image_ids))

    if n is not None:
        if randomize:
            filenames = random.sample(filenames, n)
        else:
            filenames = sorted(filenames)[-n:]

    image_ids = {
        'filenames': filenames,
        'cell_lines': [f.split('_')[0] for f in filenames],
        'plates': [f.split('_')[2] for f in filenames],
        'wells': [f.split('_')[3] for f in filenames],
        'dates': ['_'.join(f.split('_')[-3:]) for f in filenames]
    }

    encodings = []
    # get encodings
    # print('processing images from {}'.format(path))
    for file in filenames:
        img = read_image(path + file)
        img_encoded = transform(img.to(device))
        encodings.append(img_encoded)

    return encodings, image_ids


def get_image_encodings_of_cell_line(path, cell_line, drugs_wells, transform, device=torch.device('cuda'), last_n=10):
    """ Gets the last n time points of each drug. For a big dataset this should work faster. """

    filenames = []
    for file in os.listdir(path):
        if file.split('_')[0] == cell_line:
            if file.split('_')[2] + '_' + file.split('_')[3] in drugs_wells:
                filenames.append(file)

    filenames = sorted(filenames)
    i = 0
    latest_timepoints_filenames = []
    while i < len(filenames)-1:
        if filenames[i].split('_')[3] != filenames[i+1].split('_')[3]:
            latest_timepoints_filenames.extend(filenames[i-last_n+1:i+1])
        i += 1
    latest_timepoints_filenames.extend(filenames[-last_n:])
    filenames = latest_timepoints_filenames

    image_ids = {
        'filenames': filenames,
        'cell_lines': [cell_line for f in filenames],
        'plates': [f.split('_')[2] for f in filenames],
        'wells': [f.split('_')[3] for f in filenames],
        'dates': ['_'.join(f.split('_')[-3:]) for f in filenames]
    }

    encodings = []
    for file in filenames:
        img = read_image(path + file)
        img_encoded = transform(img.to(device))
        encodings.append(img_encoded)

    return encodings, image_ids


def get_f_transform(method_name, device=torch.device('cpu')):

    if method_name == 'resnet50':
        # upload pretrained resnet50
        model = pretrained.get_supervised_resnet()
        transform = lambda x: model(
            torch.unsqueeze(  # add batch dimension
                ToTensor()(  # convert PIL to tensor
                    Grayscale(num_output_channels=3)(  # apply grayscale, keeping 3 channels
                        ToPILImage()(  # conver to PIL to apply grayscale
                            Resize(size=224)(  # and resnet is trained with 224
                                Resize(size=128)(x)  # images are 256, but all models are trained with 128
                            )
                        )
                    )
                ), 0)
        ).reshape(-1).detach().cpu().numpy()

    elif method_name == 'swav_resnet50':
        # upload  resnet50, pretrained with SwAV
        model = pretrained.get_self_supervised_resnet(method='swav')
        transform = lambda x: model(
            torch.unsqueeze(  # add batch dimension
                ToTensor()(  # convert PIL to tensor
                    Grayscale(num_output_channels=3)(  # apply grayscale, keeping 3 channels
                        ToPILImage()(  # conver to PIL to apply grayscale
                            Resize(size=224)(  # and resnet is trained with 224
                                Resize(size=128)(x)  # images are 256, but all models are trained with 128
                            )
                        )
                    )
                ), 0)
        ).reshape(-1).detach().cpu().numpy()

    elif method_name == 'dino_resnet50':
        # upload  resnet50, pretrained with DINO
        model = pretrained.get_self_supervised_resnet(method='dino')
        transform = lambda x: model(
            torch.unsqueeze(  # add batch dimension
                ToTensor()(  # convert PIL to tensor
                    Grayscale(num_output_channels=3)(  # apply grayscale, keeping 3 channels
                        ToPILImage()(  # conver to PIL to apply grayscale
                            Resize(size=224)(  # and resnet is trained with 224
                                Resize(size=128)(x)  # images are 256, but all models are trained with 128
                            )
                        )
                    )
                ), 0)
        ).reshape(-1).detach().cpu().numpy()

    elif method_name == 'dino_vit':
        # upload  resnet50, pretrained with DINO
        model = pretrained.get_self_supervised_vit()
        transform = lambda x: model(
            torch.unsqueeze(  # add batch dimension
                ToTensor()(  # convert PIL to tensor
                    Grayscale(num_output_channels=3)(  # apply grayscale, keeping 3 channels
                        ToPILImage()(  # conver to PIL to apply grayscale
                            Resize(size=224)(  # and resnet is trained with 224
                                Resize(size=128)(x)  # images are 256, but all models are trained with 128
                            )
                        )
                    )
                ), 0)
        ).reshape(-1).detach().cpu().numpy()

    elif method_name == 'trained_ae_v1':
        # upload trained autoencoder v1 (tensorflow)
        autoencoder = get_trained_autoencoder()
        model = Model(autoencoder.input, autoencoder.layers[-2].output)
        transform = lambda x: model(numpy.expand_dims(Resize(size=128)(x / 255.).numpy(), axis=-1))[0].numpy().reshape(-1)

    elif method_name == 'trained_ae_v2':
        # TODO: specify path to the 1_1_3_1_0.5_0.5_1_0.75_0.5
        # path_to_model = '/Users/andreidm/ETH/projects/pheno-ml/pretrained/convae/{}/'.format(method_name)
        path_to_model = 'D:\ETH\projects\pheno-ml\\pretrained\\convae\\{}\\'.format(method_name)
        model = Autoencoder().to(device)
        # load a trained deep classifier to use it in the transform
        model.load_state_dict(torch.load(path_to_model + 'autoencoder_at_5.torch', map_location=device))
        model.eval()

        # create a transform function with weakly supervised classifier
        transform = lambda x: model.encoder(torch.unsqueeze(Resize(size=128)(x / 255.), 0)).reshape(-1).detach().cpu().numpy()

    else:
        raise ValueError('Method not recognized')

    return transform


def get_wells_of_drug_for_cell_line(cell_line, drug, plate=''):

    # meta_path = '/Users/andreidm/ETH/projects/pheno-ml/data/metadata/'
    meta_path = 'D:\ETH\projects\pheno-ml\data\metadata\\'
    cell_plate_paths = [meta_path + file for file in os.listdir(meta_path) if cell_line in file and plate in file]

    wells = []
    for path in cell_plate_paths:
        data = pandas.read_csv(path)
        drug_data = data.loc[data['Drug'] == drug, ['Final_conc_uM', 'Well']]
        max_conc_wells = drug_data.loc[drug_data['Final_conc_uM'] == drug_data['Final_conc_uM'].max(), 'Well']
        wells.extend(list(max_conc_wells))
        if len(wells) > 0:
            plate = path.split('_')[-1].split('.')[0]
            return wells, plate


def calculate_distances_for_all_well_pairs(drug_codes):

    d_2 = []
    d_cos = []
    d_corr = []
    d_bray = []

    well_pairs = itertools.combinations([x for x in range(len(drug_codes))], 2)

    for i, j in well_pairs:
        for k in range(len(drug_codes[i])):
            # we're computing distances between biological replicates essentially
            # they are expected to be small at any time point
            d_2.append(pdist([drug_codes[i][k], drug_codes[j][k]], metric='euclidean'))
            d_cos.append(pdist([drug_codes[i][k], drug_codes[j][k]], metric='cosine'))
            d_corr.append(pdist([drug_codes[i][k], drug_codes[j][k]], metric='correlation'))
            d_bray.append(pdist([drug_codes[i][k], drug_codes[j][k]], metric='braycurtis'))

    res = {
        'euclidean': numpy.median(d_2),
        'cosine': numpy.median(d_cos),
        'correlation': numpy.median(d_corr),
        'braycurtis': numpy.median(d_bray)
    }

    return res


def get_common_control_wells():
    """ Get names of wells that correspond to controls (DMSO) for all cell lines. """

    wells_by_cell_line = {}
    unique_wells = set()
    for cell_line in cell_lines:
        wells, _ = get_wells_of_drug_for_cell_line(cell_line, 'DMSO')
        wells_by_cell_line[cell_line] = wells
        for well in wells:
            unique_wells.add(well)

    common_wells = []
    for well in list(unique_wells):
        present_everywhere = True
        for cell_line in cell_lines:
            if well not in wells_by_cell_line[cell_line]:
                present_everywhere = False
                break
        if present_everywhere:
            common_wells.append(well)

    return common_wells


def compare_similarity(path_to_data, methods, uid='', device=torch.device('cuda')):

    # save_to = '/Users/andreidm/ETH/projects/pheno-ml/res/comparison/similarity/'
    save_to = 'D:\ETH\projects\pheno-ml\\res\\comparison\\similarity\\'
    if not os.path.exists(save_to):
        os.makedirs(save_to)

    results = {'cell_line': [], 'drug': [], 'method': [],
               'euclidean': [], 'cosine': [], 'correlation': [], 'braycurtis': []}

    for method_name in methods:
        transform = get_f_transform(method_name, device=device)

        for cell_line in tqdm(cell_lines):
            for drug in drugs:

                drug_wells, _ = get_wells_of_drug_for_cell_line(cell_line, drug)

                drug_codes = []
                for well in drug_wells:
                    encodings, _ = get_image_encodings_from_path(path_to_data, [well, cell_line], transform, device=device, n=5, randomize=False)
                    drug_codes.append(encodings)

                # compare Methotrexate with Pemetrexed
                results['method'].append(method_name)
                results['cell_line'].append(cell_line)
                results['drug'].append(drug)
                distances = calculate_distances_for_all_well_pairs(drug_codes)
                results['euclidean'].append(distances['euclidean'])
                results['cosine'].append(distances['cosine'])
                results['correlation'].append(distances['correlation'])
                results['braycurtis'].append(distances['braycurtis'])

    results = pandas.DataFrame(results)
    results.to_csv(save_to + 'similarity{}.csv'.format(uid), index=False)


def collect_and_save_clustering_results_for_multiple_parameter_sets(path_to_images, methods, range_with_step, uid='', device=torch.device('cuda')):
    """ Cluster the dataset over multiple parameters, evaluate results and save results as a dataframe. """

    # save_to = '/Users/andreidm/ETH/projects/pheno-ml/res/comparison/clustering/'
    save_to = 'D:\ETH\projects\pheno-ml\\res\\comparison\\clustering\\'
    if not os.path.exists(save_to):
        os.makedirs(save_to)

    results = {'cell_line': [], 'method': [], 'min_cluster_size': [],
               'n_clusters': [], 'noise': [], 'silhouette': [], 'calinski_harabasz': [], 'davies_bouldin': []}

    for method_name in methods:

        transform = get_f_transform(method_name, device=device)

        for cell_line in tqdm(cell_lines):

            all_drugs_wells = []
            for drug in drugs:
                if drug not in ['PBS', 'DMSO']:
                    max_conc_drug_wells, plate = get_wells_of_drug_for_cell_line(cell_line, drug)
                    all_drugs_wells.extend([plate+'_'+well for well in max_conc_drug_wells])

            all_codes, _ = get_image_encodings_of_cell_line(path_to_images, cell_line, all_drugs_wells, transform=transform)
            encodings = numpy.array(all_codes)

            for min_cluster_size in tqdm(range(*range_with_step)):

                start = time.time()
                reducer = umap.UMAP(n_neighbors=min_cluster_size, metric='euclidean')
                embedding = reducer.fit_transform(encodings)
                # cluster encodings
                clusterer = HDBSCAN(metric='euclidean', min_samples=1, min_cluster_size=min_cluster_size, allow_single_cluster=False)
                clusterer.fit(embedding)
                clusters = clusterer.labels_
                print('umap + hdbscan clustering with n = {}, took {} s'.format(min_cluster_size, int(time.time() - start)))

                n_clusters = numpy.max(clusters) + 1
                noise = int(numpy.sum(clusters == -1) / len(clusters) * 100)

                try:
                    silhouette = metrics.silhouette_score(embedding, clusters)
                    calinski_harabasz = metrics.calinski_harabasz_score(embedding, clusters)
                    davies_bouldin = metrics.davies_bouldin_score(embedding, clusters)
                except ValueError:
                    # single cluster
                    silhouette, calinski_harabasz, davies_bouldin = -1, -1, -1

                results['cell_line'].append(cell_line)
                results['method'].append(method_name)
                results['min_cluster_size'].append(min_cluster_size)
                results['n_clusters'].append(n_clusters)
                results['noise'].append(noise)
                results['silhouette'].append(silhouette)
                results['calinski_harabasz'].append(calinski_harabasz)
                results['davies_bouldin'].append(davies_bouldin)

    results = pandas.DataFrame(results)
    results.to_csv(save_to + 'clustering_{}.csv'.format(uid), index=False)


def plot_similarity_results(path_to_results='/Users/andreidm/ETH/projects/pheno-ml/res/comparison/similarity/similarity.csv'):

    results = pandas.read_csv(path_to_results)

    results.loc[results['method'] == 'resnet50', 'method'] = 'RN-50\n(pretrained)'
    results.loc[results['method'] == 'swav_resnet50', 'method'] = 'RN-50+SwAV\n(pretrained)'
    results.loc[results['method'] == 'dino_resnet50', 'method'] = 'RN-50+DINO\n(pretrained)'
    results.loc[results['method'] == 'dino_vit', 'method'] = 'ViT+DINO\n(pretrained)'
    results.loc[results['method'] == 'trained_ae_v1', 'method'] = 'ConvAE v1\n(trained)'
    results.loc[results['method'] == 'trained_ae_v2', 'method'] = 'ConvAE v2\n(trained)'

    # normalize distances
    for method in list(results['method'].unique()):
        for metric in ['euclidean', 'cosine', 'correlation', 'braycurtis']:
            results.loc[results['method'] == method, metric] /= results.loc[results['method'] == method, metric].max()
            results.loc[results['method'] == method, metric] = 1 / results.loc[results['method'] == method, metric]

    seaborn.set()
    i = 1
    pyplot.figure(figsize=(10, 3))
    pyplot.suptitle('Comparison of similarity of drugs')
    for metric in ['euclidean', 'cosine', 'correlation', 'braycurtis']:
        pyplot.subplot(1, 4, i)
        seaborn.barplot(x='method', y=metric, data=results)
        pyplot.title(metric)
        pyplot.xticks(rotation=45, fontsize=6)
        i += 1
    pyplot.tight_layout()
    pyplot.show()


def plot_clustering_results(path_to_results='/Users/andreidm/ETH/projects/pheno-ml/res/comparison/clustering/clustering_by_cell_lines.csv'):

    results = pandas.read_csv(path_to_results)

    results.loc[results['method'] == 'resnet50', 'method'] = 'RN-50\n(pretrained)'
    results.loc[results['method'] == 'swav_resnet50', 'method'] = 'RN-50+SwAV\n(pretrained)'
    results.loc[results['method'] == 'dino_resnet50', 'method'] = 'RN-50+DINO\n(pretrained)'
    results.loc[results['method'] == 'dino_vit', 'method'] = 'ViT+DINO\n(pretrained)'
    results.loc[results['method'] == 'trained_ae_v1', 'method'] = 'ConvAE v1\n(trained)'
    results.loc[results['method'] == 'trained_ae_v2', 'method'] = 'ConvAE v2\n(trained)'

    # # filter out failed clustering attempts for better visualization
    # results = results.drop(results.loc[results['silhouette'] == -1, :].index)
    # # filter out too high calinski-harabasz values for better visualization
    # results = results.drop(results.loc[results['calinski_harabasz'] > 20000, :].index)

    results['not_noise'] = 100 - results['noise']
    results['davies_bouldin-1'] = 1 / results['davies_bouldin']

    seaborn.set()
    i = 1
    pyplot.figure(figsize=(12, 3))
    pyplot.suptitle('Comparison of clustering')
    for metric in ['not_noise', 'calinski_harabasz', 'silhouette', 'davies_bouldin-1']:
        pyplot.subplot(1, 4, i)
        seaborn.boxplot(x='method', y=metric, data=results)
        pyplot.title(metric)
        pyplot.xticks(rotation=45, fontsize=6)
        i += 1
    pyplot.tight_layout()
    pyplot.show()


if __name__ == "__main__":

    path_to_data = 'D:\ETH\projects\pheno-ml\\data\\full\\cropped\\'
    # path_to_data = '/Users/andreidm/ETH/projects/pheno-ml/data/cropped/training/single_class/'
    models = []

    device = torch.device('cuda')

    evaluate = False
    plot = True

    uid = ''

    if evaluate:
        # distance-based analysis
        compare_similarity(path_to_data, models, uid=uid, device=device)
        # classification of drugs vs controls
        train_classifiers_with_pretrained_encoder_and_save_results(path_to_data, 25, models, uid=uid, batch_size=1024, device=device)
        # clustering analysis within cell lines
        collect_and_save_clustering_results_for_multiple_parameter_sets(path_to_data, models, (10, 160, 10), uid='by_cell_lines{}'.format(uid), device=device)

    if plot:
        plot_similarity_results('/Users/andreidm/ETH/projects/pheno-ml/res/comparison/similarity/similarity.csv')
        plot_clustering_results('/Users/andreidm/ETH/projects/pheno-ml/res/comparison/clustering/clustering.csv')
        plot_classification_results('/Users/andreidm/ETH/projects/pheno-ml/res/comparison/classification/classification.csv')