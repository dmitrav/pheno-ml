
import os, pandas, torch, numpy, random, umap, time, seaborn
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

from src.self_supervised import DeepClassifier
from src.constants import cell_lines, drugs
from src import pretrained


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


def collect_and_plot_classification_results(models, path_to_results='/Users/andreidm/ETH/projects/pheno-ml/res/comparison/classification/'):

    results = {
        'models': [], 'lrs': [], 'ms': [], 'wds': [],
        'epoch': [], 'accuracy': [], 'recall': [], 'precision': [], 'specificity': [], 'f1': []
    }

    for model in models:
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
    results_df.to_csv(path_to_results + 'classification.csv')

    i = 1
    seaborn.set()
    pyplot.figure(figsize=(12,3))
    pyplot.suptitle('Comparison of drug-control classification')
    for metric in ['accuracy', 'recall', 'precision', 'specificity', 'f1']:
        pyplot.subplot(1, 5, i)
        seaborn.boxplot(x='models', y=metric, data=results_df)
        pyplot.title(metric)
        i += 1
    pyplot.tight_layout()
    pyplot.show()


def get_image_encodings_from_path(path, common_image_ids, transform, n=None, randomize=True):

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
        img_encoded = transform(img)
        encodings.append(img_encoded.detach().cpu().numpy())

    return encodings, image_ids


def get_image_encodings_of_cell_line(path, cell_line, drugs_wells, transform, last_n=10):
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
        img_encoded = transform(img)
        encodings.append(img_encoded.detach().cpu().numpy())

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
        ).reshape(-1)

    elif method_name == 'swav_resnet50':
        # upload  resnet50, pretrained with SwAV
        model = pretrained.get_self_supervised_resnet()
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
        ).reshape(-1)
    else:
        # upload my models
        # path_to_model = '/Users/andreidm/ETH/projects/pheno-ml/res/models/{}/'.format(method_name)
        path_to_model = 'D:\ETH\projects\pheno-ml\\res\\byol\\{}\\'.format(method_name)
        model = DeepClassifier().to(device)
        # load a trained deep classifier to use it in the transform
        model.load_state_dict(torch.load(path_to_model + 'best.torch', map_location=device))
        model.eval()
        # truncate to the layer with learned representations
        model = Sequential(*list(model.model.children())[:-4])
        # create a transform function with weakly supervised classifier
        transform = lambda x: model(
            Resize(size=128)(
                torch.Tensor(numpy.expand_dims((x / 255.), axis=0)).to(device)
            )
        ).reshape(-1)

    return transform


def get_wells_of_drug_for_cell_line(cell_line, drug, plate=''):

    meta_path = '/Users/andreidm/ETH/projects/pheno-ml/data/metadata/'
    # meta_path = 'D:\ETH\projects\pheno-ml\data\metadata\\'
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


def calculate_similarity_of_pair(codes_A, codes_B):

    d_2 = []
    d_cos = []
    d_corr = []
    d_bray = []

    for code_a in codes_A:
        for code_b in codes_B:

            d_2.append(pdist([code_a, code_b], metric='euclidean'))
            d_cos.append(pdist([code_a, code_b], metric='cosine'))
            d_corr.append(pdist([code_a, code_b], metric='correlation'))
            d_bray.append(pdist([code_a, code_b], metric='braycurtis'))

    res = {
        'euclidean': numpy.median(d_2),
        'cosine': numpy.median(d_cos),
        'correlation': numpy.median(d_corr),
        'braycurtis': numpy.median(d_bray)
    }

    return res


def compare_similarity(path_to_data, methods, uid='', device=torch.device('cuda')):

    save_to = '/Users/andreidm/ETH/projects/pheno-ml/res/comparison/similarity/'
    # save_to = 'D:\ETH\projects\pheno-ml\\res\\comparison\\similarity\\'
    if not os.path.exists(save_to):
        os.makedirs(save_to)

    results = {'group_by': [], 'method': [], 'comparison': [],
               'euclidean': [], 'cosine': [], 'correlation': [], 'braycurtis': []}

    for method_name in methods:

        transform = get_f_transform(method_name, device=device)

        for cell_line in tqdm(cell_lines):

            mtx_wells, plate = get_wells_of_drug_for_cell_line(cell_line, 'Methotrexate')
            mtx_controls_wells, _ = get_wells_of_drug_for_cell_line(cell_line, 'DMSO', plate=plate)
            mtx_controls_wells = random.sample(mtx_controls_wells, len(mtx_wells))

            ptx_wells, plate = get_wells_of_drug_for_cell_line(cell_line, 'Pemetrexed')
            ptx_controls_wells, _ = get_wells_of_drug_for_cell_line(cell_line, 'DMSO', plate=plate)
            ptx_controls_wells = random.sample(ptx_controls_wells, len(ptx_wells))

            mtx_codes = []
            for well in mtx_wells:
                encodings, _ = get_image_encodings_from_path(path_to_data, [well, cell_line], transform, n=10, randomize=False)
                mtx_codes.extend(encodings)

            mtx_controls_codes = []
            for well in mtx_controls_wells:
                encodings, _ = get_image_encodings_from_path(path_to_data, [well, cell_line], transform, n=10, randomize=False)
                mtx_controls_codes.extend(encodings)

            ptx_codes = []
            for well in ptx_wells:
                encodings, _ = get_image_encodings_from_path(path_to_data, [well, cell_line], transform, n=10, randomize=False)
                ptx_codes.extend(encodings)

            ptx_controls_codes = []
            for well in ptx_controls_wells:
                encodings, _ = get_image_encodings_from_path(path_to_data, [well, cell_line], transform, n=10, randomize=False)
                ptx_controls_codes.extend(encodings)

            # compare Methotrexate with Pemetrexed
            results['group_by'].append(cell_line)
            results['method'].append(method_name)
            results['comparison'].append('MTX-PTX')
            comparison = calculate_similarity_of_pair(mtx_codes, ptx_codes)
            results['euclidean'].append(comparison['euclidean'])
            results['cosine'].append(comparison['cosine'])
            results['correlation'].append(comparison['correlation'])
            results['braycurtis'].append(comparison['braycurtis'])

            # compare Methotrexate with DMSO (control)
            results['group_by'].append(cell_line)
            results['method'].append(method_name)
            results['comparison'].append('MTX-DMSO')
            comparison = calculate_similarity_of_pair(mtx_codes, mtx_controls_codes)
            results['euclidean'].append(comparison['euclidean'])
            results['cosine'].append(comparison['cosine'])
            results['correlation'].append(comparison['correlation'])
            results['braycurtis'].append(comparison['braycurtis'])

            # compare Pemetrexed with DMSO (control)
            results['group_by'].append(cell_line)
            results['method'].append(method_name)
            results['comparison'].append('PTX-DMSO')
            comparison = calculate_similarity_of_pair(ptx_codes, ptx_controls_codes)
            results['euclidean'].append(comparison['euclidean'])
            results['cosine'].append(comparison['cosine'])
            results['correlation'].append(comparison['correlation'])
            results['braycurtis'].append(comparison['braycurtis'])

    results = pandas.DataFrame(results)
    results.to_csv(save_to + 'similarity_{}.csv'.format(uid), index=False)


def collect_and_save_clustering_results_for_multiple_parameter_sets(path_to_images, methods, range_with_step, uid='', device=torch.device('cuda')):
    """ Cluster the dataset over multiple parameters, evaluate results and save results as a dataframe. """

    save_to = '/Users/andreidm/ETH/projects/pheno-ml/res/comparison/clustering/'
    # save_to = 'D:\ETH\projects\pheno-ml\\res\\comparison\\clustering\\'
    if not os.path.exists(save_to):
        os.makedirs(save_to)

    results = {'group_by': [], 'method': [], 'min_cluster_size': [],
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

                results['group_by'].append(cell_line)
                results['method'].append(method_name)

                results['min_cluster_size'].append(min_cluster_size)
                results['n_clusters'].append(n_clusters)
                results['noise'].append(noise)
                results['silhouette'].append(silhouette)
                results['calinski_harabasz'].append(calinski_harabasz)
                results['davies_bouldin'].append(davies_bouldin)

    results = pandas.DataFrame(results)
    results.to_csv(save_to + 'clustering_{}.csv'.format(uid), index=False)


def plot_similarity_results(path_to_results='/Users/andreidm/ETH/projects/pheno-ml/res/comparison/similarity/similarity_pretrained.csv'):

    results = pandas.read_csv(path_to_results)
    methods = list(results['method'].unique())

    seaborn.set()
    for method in methods:
        i = 1
        pyplot.figure(figsize=(10, 3))
        pyplot.suptitle('Comparison of drug similarity: {}'.format(method))
        for metric in ['euclidean', 'cosine', 'correlation', 'braycurtis']:
            pyplot.subplot(1, 4, i)
            seaborn.barplot(x='comparison', y=metric, data=results)
            pyplot.title(metric)
            pyplot.xticks(rotation=45)
            i += 1
        pyplot.tight_layout()
    pyplot.show()


def plot_clustering_results(path_to_results='/Users/andreidm/ETH/projects/pheno-ml/res/comparison/clustering/clustering_by_cell_lines_pretrained.csv'):

    results = pandas.read_csv(path_to_results)

    results.loc[results['method'] == 'resnet50', 'method'] = 'ResNet-50'
    results.loc[results['method'] == 'swav_resnet50', 'method'] = 'SwAV'

    results['not_noise'] = 100 - results['noise']
    results['davies_bouldin-1'] = 1 / results['davies_bouldin']

    seaborn.set()
    i = 1
    pyplot.figure(figsize=(10, 3))
    pyplot.suptitle('Comparison of clustering')
    for metric in ['not_noise', 'calinski_harabasz', 'silhouette', 'davies_bouldin-1']:
        pyplot.subplot(1, 4, i)
        seaborn.boxplot(x='method', y=metric, data=results)
        pyplot.title(metric)
        i += 1
    pyplot.tight_layout()
    pyplot.show()


if __name__ == "__main__":

    # path_to_data = 'D:\ETH\projects\pheno-ml\\data\\full\\cropped\\'
    path_to_data = '/Users/andreidm/ETH/projects/pheno-ml/data/cropped/training/single_class/'
    # models = os.listdir('D:\ETH\projects\pheno-ml\\res\\byol\\')
    models = ['resnet50', 'swav_resnet50']

    device = torch.device('cpu')

    evalute = False
    plot = True

    if evalute:
        # distance-based analysis of known drugs
        compare_similarity(path_to_data, models, uid='pretrained', device=device)
        # clustering analysis within cell lines
        collect_and_save_clustering_results_for_multiple_parameter_sets(path_to_data, models, (10, 160, 10), uid='by_cell_lines_pretrained', device=device)
        # classification of drugs vs controls
        train_classifier_with_pretrained_encoder(25, models, batch_size=1024, device=device)

    if plot:
        plot_similarity_results()
        plot_clustering_results()
        collect_and_plot_classification_results(models)