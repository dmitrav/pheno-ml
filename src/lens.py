
import torch, os, time, pandas, numpy, seaborn
from torch import nn, optim
from tqdm import tqdm
from torchvision.io import read_image
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from torchmetrics import Recall, Accuracy, Specificity, Precision
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import mannwhitneyu, ks_2samp, kruskal

from src.constants import drugs, cell_lines, controls
from src.comparison import get_f_transform
from src.comparison import get_wells_of_drug_for_cell_line
from src.self_supervised import Autoencoder


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


def collect_data_and_split(path_to_data, method='trained_ae_full', device=torch.device('cuda'), save_path=None):

    transform = get_f_transform(method, device=device)
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

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        pandas.DataFrame(X_train).to_csv(save_path + 'X_train.csv', index=False)
        pandas.DataFrame(X_test).to_csv(save_path + 'X_test.csv', index=False)
        pandas.DataFrame(y_train).to_csv(save_path + 'y_train.csv', index=False)
        pandas.DataFrame(y_test).to_csv(save_path + 'y_test.csv', index=False)

    return X_train, X_test, y_train, y_test


def train_drug_classifier_alone(path_to_data, epochs, uid='', device=torch.device('cuda')):

    save_path = 'D:\ETH\projects\pheno-ml\\res\\drug_classifier\\{}\\'.format(uid)
    # save_path = '/Users/andreidm/ETH/projects/pheno-ml/res/drug_classifier/{}/'.format(uid)
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

            model = DrugClassifier(in_dim=4096, out_dim=33).to(device)

            train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=True)
            test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=True, drop_last=True)

            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            run_supervised_classifier_training(train_loader, test_loader, model, optimizer, criterion, device,
                                               epochs=epochs, save_to=save_path + 'lr={},bs={}\\'.format(lr, bs))


def train_lens_with_drug_classifier(path_to_data, epochs, adv_coefs, initialize_lens=False, uid='', device=torch.device('cuda')):

    save_path = 'D:\ETH\projects\pheno-ml\\res\\drug_classifier\\{}\\'.format(uid)
    # save_path = '/Users/andreidm/ETH/projects/pheno-ml/res/drug_classifier/{}/'.format(uid)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cpu = torch.device('cpu')
    cuda = torch.device('cuda')

    # collect images
    X_train, X_test, y_train, y_test = collect_data_and_split(path_to_data, method='no', device=cpu)

    # make datasets and loaders
    train_dataset = TensorDataset(torch.Tensor(X_train), torch.LongTensor(y_train))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.LongTensor(y_test))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=True)

    dc = DrugClassifier(in_dim=4096, out_dim=33).to(cpu)
    dc_lr = 0.001

    # define feature extractor for lens
    # weights_path = '/Users/andreidm/ETH/projects/pheno-ml/pretrained/convae/trained_ae_v2/autoencoder_at_5.torch'
    weights_path = 'D:\ETH\projects\pheno-ml\\pretrained\\convae\\trained_ae_full\\autoencoder_at_5.torch'
    pretrained_model = Autoencoder().to(cpu)
    pretrained_model.load_state_dict(torch.load(weights_path, map_location=cpu))
    pretrained_model.eval()

    if initialize_lens:
        lens = Autoencoder().to(cuda)
        # initialize with the trained model
        lens.load_state_dict(torch.load(weights_path, map_location=cuda))
        lens.eval()
        lens_lr = dc_lr / 2
    else:
        lens = Autoencoder().to(cuda)
        lens_lr = dc_lr

    dc_optimizer = optim.Adam(dc.parameters(), lr=dc_lr)
    dc_criterion = nn.CrossEntropyLoss()

    lens_optimizer = optim.Adam(lens.parameters(), lr=lens_lr)
    lens_criterion = nn.BCELoss()

    for coef in adv_coefs:

        run_adversarial_lens_training(train_loader, test_loader, dc, lens, pretrained_model,
                                      dc_optimizer, lens_optimizer, dc_criterion, lens_criterion, coef,
                                      device, epochs=epochs, save_to=save_path + 'coef={}\\'.format(coef))


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


def run_adversarial_lens_training(loader_train, loader_test, classifier, lens, feature_extractor,
                                  c_optimizer, l_optimizer, c_criterion, l_criterion, advers_coef,
                                  device, lr_scheduler=None, epochs=10, save_to=""):

    cpu = torch.device('cpu')
    cuda = torch.device('cuda')

    if not os.path.exists(save_to):
        os.makedirs(save_to)

    f_acc = Accuracy(top_k=3).to(cpu)

    rec_loss_epoch = 0
    acc_epoch = 0
    rec_loss_history = []
    loss_history = []
    acc_history = []
    for epoch in range(epochs):

        start = time.time()
        lens_loss_epoch = 0
        classifier_loss_epoch = 0
        rec_loss_epoch = 0
        acc_epoch = 0
        for images, labels in loader_train:

            # TRAIN CLASSIFIER

            # reset gradients to zero
            c_optimizer.zero_grad()
            # get features of drugs
            images = images.float().to(cuda)
            labels = labels.to(cpu)

            # apply lens
            reconstructions = lens(images)
            # retrieve codes of reconstructions with pretrained model inside transform function
            encodings = feature_extractor.encoder(reconstructions.to(cpu))
            encodings = encodings.reshape((encodings.shape[0], -1))
            # run through classifier
            outputs = classifier(encodings)
            # calculate loss
            c_loss = c_criterion(outputs, labels)
            c_loss.backward()
            c_optimizer.step()

            acc_epoch += float(f_acc(outputs, labels))

            # TRAIN LENS

            # reset the gradients to zero
            l_optimizer.zero_grad()
            with torch.enable_grad():

                l_loss = 0.
                # compute reconstructions
                outputs = lens(images)
                # compute training reconstruction loss
                l_loss += l_criterion(outputs, images)
                rec_loss_epoch += l_loss.item()
                # introduce adversary: subtract classifier loss
                l_loss -= c_loss.item() * advers_coef
                # compute accumulated gradients
                l_loss.backward()
                # perform parameter update based on current gradients
                l_optimizer.step()

            # add the mini-batch training loss to epoch loss
            lens_loss_epoch += l_loss.item()
            classifier_loss_epoch += c_loss.item()

        # compute the epoch training loss
        lens_loss_epoch = lens_loss_epoch / len(loader_train)
        classifier_loss_epoch = classifier_loss_epoch / len(loader_train)
        rec_loss_epoch = rec_loss_epoch / len(loader_train)
        acc_epoch = acc_epoch / len(loader_train)
        rec_loss_history.append(rec_loss_epoch)
        acc_history.append(acc_epoch)
        loss_history.append(lens_loss_epoch)

        val_acc = 0
        for images, labels in loader_test:
            # process drugs
            images = images.float().to(device)
            labels = labels.to(cpu)
            reconstructions = lens(images)
            encodings = feature_extractor.encoder(reconstructions.to(cpu))
            encodings = encodings.reshape((encodings.shape[0], -1))
            outputs = classifier(encodings)
            val_acc += float(f_acc(outputs, labels))

        # compute epoch validation accuracy
        val_acc = val_acc / len(loader_test)

        # display the epoch training loss
        print("epoch {}/{}: {} min, lens_loss = {:.4f},\n"
              "classifier_loss = {:.4f}, reconstruction_loss = {:.4f},\n"
              "accuracy = {:.4f}, t_accuracy = {:.4f}\n"
              .format(epoch + 1, epochs, int((time.time() - start) / 60), lens_loss_epoch, classifier_loss_epoch,
                      rec_loss_epoch, acc_epoch, val_acc))

        torch.save(lens.state_dict(), save_to + 'lens_at_{}.torch'.format(epoch + 1))
        torch.save(classifier.state_dict(), save_to + 'classifier_at_{}.torch'.format(epoch + 1))

    plot_reconstructions_and_lens_effects(loader_test, lens, feature_extractor, save_to=save_to, n_images=30)

    history = pandas.DataFrame({'epoch': [x + 1 for x in range(len(loss_history))], 'lens_loss': loss_history,
                                'lens_rec_loss': rec_loss_history, 'acc': acc_history})
    history.to_csv(save_to + 'history.csv', index=False)


def plot_reconstructions_and_lens_effects(data_loader, lens, feature_extractor, save_to='res/', n_images=10):

    cpu = torch.device('cpu')
    cuda = torch.device('cuda')

    for i in range(n_images):
        images, _ = next(iter(data_loader))
        initial_image = images.squeeze()[0].numpy()
        reconstructed = feature_extractor(torch.unsqueeze(images[0], 0).to(cpu))
        reconstructed = reconstructed.cpu().detach().numpy()[0][0]
        lensed = lens(torch.unsqueeze(images[0], 0).to(cuda))
        lensed = lensed.cpu().detach().numpy()[0][0]

        difference = numpy.abs(reconstructed - lensed)

        pyplot.figure()
        pyplot.subplot(141)
        pyplot.imshow(initial_image, cmap="gray")
        pyplot.title("original")
        pyplot.subplot(142)
        pyplot.imshow(reconstructed, cmap="gray")
        pyplot.title("reconstructed")
        pyplot.subplot(143)
        pyplot.imshow(lensed, cmap="gray")
        pyplot.title("lensed")
        pyplot.subplot(144)
        pyplot.imshow(difference, cmap="gray")
        pyplot.title("difference")

        if save_to:
            if not os.path.exists(save_to + 'recs/'):
                os.makedirs(save_to + 'recs/')
            pyplot.savefig(save_to + 'recs/{}.pdf'.format(i))
        else:
            pyplot.show()
    pyplot.close('all')


def collect_and_save_lens_classification_results(save=False):

    # collect and save results
    results = {
        'method': [], 'lrs': [], 'bs': [],
        'epoch': [], 'accuracy': [], 'recall': [], 'precision': [], 'specificity': [], 'f1': []
    }

    save_path = 'D:\ETH\projects\pheno-ml\\res\drug_classifier\\'
    methods = ['without_lens', 'with_reg_lens', 'with_adv_lens']

    for method_name in methods:
        for param_set in os.listdir(save_path + method_name):
            if param_set != '.DS_Store':
                data = pandas.read_csv(save_path + '{}\\{}\\history.csv'.format(method_name, param_set))
                best_f1_data = data.loc[data['f1'] == data['f1'].max(), :]

                results['method'].append(method_name)
                results['lrs'].append(float(param_set.split(',')[0].split('=')[1]))
                results['bs'].append(float(param_set.split(',')[1].split('=')[1]))

                results['epoch'].append(int(best_f1_data['epoch']))
                results['accuracy'].append(float(best_f1_data['acc']))
                results['recall'].append(float(best_f1_data['recall']))
                results['precision'].append(float(best_f1_data['precision']))
                results['specificity'].append(float(best_f1_data['specificity']))
                results['f1'].append(float(best_f1_data['f1']))

    results_df = pandas.DataFrame(results)
    if save:
        results_df.to_csv(save_path + 'classification_lens.csv', index=False)
    return results_df


def plot_lens_classification_results(results):

    metric = 'accuracy'

    results = results[results['lrs'] <= 0.01]

    _, p12 = mannwhitneyu(results.loc[results['method'] == 'without_lens', metric],
                          results.loc[results['method'] == 'with_adv_lens', metric],
                          alternative='less')

    _, p13 = mannwhitneyu(results.loc[results['method'] == 'without_lens', metric],
                          results.loc[results['method'] == 'with_reg_lens', metric],
                          alternative='less')

    print('Mann-Whitney U:')
    print('no lens {} < adversarial lens {}:'.format(metric, metric), p12 * 2)
    print('no lens {} < regularized lens {}:'.format(metric, metric), p13 * 2)

    results.loc[results['method'] == 'without_lens', 'method'] = 'No lens'
    results.loc[results['method'] == 'with_adv_lens', 'method'] = 'Lens\n(adversarial)'
    results.loc[results['method'] == 'with_reg_lens', 'method'] = 'Lens\n(regularized)'

    seaborn.set()
    pyplot.figure(figsize=(4,3))
    seaborn.boxplot(x='method', y=metric, data=results)
    pyplot.xlabel("")
    # pyplot.xticks(rotation=45)
    pyplot.tight_layout()
    pyplot.show()


def plot_image_and_map(image, reconstructed_image, lensed_image,
                       no_lens_label, no_lens_prob, lens_label, lens_prob,
                       save_to=None):

    reconstructed = reconstructed_image.cpu().detach().numpy()[0][0]
    lensed = lensed_image.cpu().detach().numpy()[0][0]
    # plot the image and the morphology map
    difference = numpy.abs(reconstructed - lensed)
    enhanced_diff = numpy.power(difference, 1.5)  # magic number

    pyplot.figure(figsize=(5, 3))
    pyplot.subplot(121)
    pyplot.imshow(numpy.squeeze(image), cmap="gray")
    pyplot.title("original image")
    pyplot.subplot(122)
    pyplot.imshow(enhanced_diff, cmap="gray")
    pyplot.title("feature importance map")
    # and provide probabilities
    error_type = 'FP' if no_lens_label != lens_label else 'TP'
    pyplot.suptitle('without lens: {}, p={} ({})\nwith lens: {}, p={} (TP)'.format(
        no_lens_label, no_lens_prob, error_type, lens_label, lens_prob)
    )
    pyplot.tight_layout()

    if save_to:
        if not os.path.exists(save_to):
            os.makedirs(save_to)
        pyplot.savefig(save_to + '{},p={},{},p={}.pdf'.format(lens_label, lens_prob, no_lens_label, no_lens_prob))
    else:
        pyplot.show()

    pyplot.close('all')


def plot_altered_morphology_maps(path_to_data, n=30):

    save_path = 'D:\ETH\projects\pheno-ml\\res\\drug_classifier\\lens_maps\\'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cuda = torch.device('cuda')

    # define feature extractor
    fe_path = 'D:\ETH\projects\pheno-ml\\pretrained\\convae\\trained_ae_full\\autoencoder_at_5.torch'
    feature_extractor = Autoencoder().to(cuda)
    feature_extractor.load_state_dict(torch.load(fe_path, map_location=cuda))
    feature_extractor.eval()

    # define default classifier
    classifier_no_lens_path = 'D:\ETH\projects\pheno-ml\\res\drug_classifier\without_lens\lr=0.001,bs=1024\\classifier_at_30.torch'
    classifier = DrugClassifier(in_dim=4096, out_dim=33).to(cuda)
    classifier.load_state_dict(torch.load(classifier_no_lens_path, map_location=cuda))
    classifier.eval()

    # define lens
    lens_path = 'D:\ETH\projects\pheno-ml\\res\drug_classifier\lens_init\coef=-60\\lens_at_5.torch'
    lens = Autoencoder().to(cuda)
    lens.load_state_dict(torch.load(lens_path, map_location=cuda))
    lens.eval()

    # define classifier for lens
    classifier_for_lens_path = lens_path.replace('lens_at', 'classifier_at')
    classifier_lens = DrugClassifier(in_dim=4096, out_dim=33).to(cuda)
    classifier_lens.load_state_dict(torch.load(classifier_for_lens_path, map_location=cuda))
    classifier_lens.eval()

    all_labels = [*controls, *drugs]

    # collect images
    images, _, labels, _ = collect_data_and_split(path_to_data, method='no', device=cuda)

    control_fp_count = 0
    drug_fp_count = 0
    increased_confidence_count = 0

    for image, label in zip(images, labels):

        image_tensor = torch.unsqueeze(torch.Tensor(image), 0).to(cuda)

        reconstructed_image = feature_extractor(image_tensor)
        image_code = feature_extractor.encoder(image_tensor)
        default_predicted_label = classifier(image_code)

        lensed_image = lens(image_tensor)
        lensed_image_code = feature_extractor.encoder(lensed_image)
        lensed_predicted_label = classifier_lens(lensed_image_code)

        true_label = all_labels[label]
        no_lens_label = all_labels[int(default_predicted_label.argmax())]
        lens_label = all_labels[int(lensed_predicted_label.argmax())]
        no_lens_prob = round(float(default_predicted_label.max()), 4)
        lens_prob = round(float(lensed_predicted_label.max()), 4)

        if no_lens_label != true_label and lens_label == true_label:
            # if lens helped to predict correct label
            if no_lens_label in ['DMSO', 'PBS'] and true_label not in ['DMSO', 'PBS'] and \
                    no_lens_prob < 0.5 < lens_prob and control_fp_count < n:
                # lens allowed to differentiate a drug from controls
                plot_image_and_map(image, reconstructed_image, lensed_image,
                                   no_lens_label, no_lens_prob,
                                   lens_label, lens_prob,
                                   save_to=save_path + 'control_fps\\')
                control_fp_count += 1
            elif no_lens_prob < 0.5 < lens_prob and drug_fp_count < n:
                # lens allowed to avoid drug misclassification
                plot_image_and_map(image, reconstructed_image, lensed_image,
                                   no_lens_label, no_lens_prob,
                                   lens_label, lens_prob,
                                   save_to=save_path + 'drug_fps\\')
                drug_fp_count += 1
            else:
                pass
        elif no_lens_label == lens_label == true_label:
            if no_lens_prob + 0.5 < lens_prob and increased_confidence_count < n:
                # lens increased confidence in drug classification
                plot_image_and_map(image, reconstructed_image, lensed_image,
                                   no_lens_label, no_lens_prob,
                                   lens_label, lens_prob,
                                   save_to=save_path + 'higher_confidence\\')
                increased_confidence_count += 1
        else:
            pass


if __name__ == "__main__":

    # path_to_data = '/Users/andreidm/ETH/projects/pheno-ml/data/cropped/training/single_class/'
    path_to_data = 'D:\ETH\projects\pheno-ml\data\\full\\cropped\\'
    # save_data_path = '/Users/andreidm/ETH/projects/pheno-ml/res/drug_classifier/data/'
    save_data_path = 'D:\ETH\projects\pheno-ml\\res\\drug_classifier\\data\\'

    device = torch.device('cuda')

    # # obtain codes and save as DFs
    # collect_data_and_split(path_to_data, method='trained_ae_full', device=device, save_path=save_data_path)

    # # classification of 33 drugs
    # train_drug_classifier_alone(save_data_path, 30, uid='without_lens', device=device)

    # # training of the lens with classification adversary
    # train_lens_with_drug_classifier(path_to_data, 5, [-1, -20, -40, -60], initialize_lens=True, uid='lens_init')
    # # training of the lens with classification adversary
    # train_lens_with_drug_classifier(path_to_data, 5, [-1, -20, -40, -60, 1], initialize_lens=False, uid='lens_no_init')

    # # save lensed data and train a classifier on it
    # # adversarial setup
    # # save_path = '/Users/andreidm/ETH/projects/pheno-ml/res/drug_classifier/data_adv_lens/'
    # save_path = 'D:\ETH\projects\pheno-ml\\res\drug_classifier\\data_adv_lens\\'
    # # collect_data_and_split(path_to_data, method='adv_lens', save_path=save_path)
    # train_drug_classifier_alone(save_path, 30, uid='with_adv_lens', device=device)

    # # save lensed data and train a classifier on it
    # # regularized setup
    # # save_path = '/Users/andreidm/ETH/projects/pheno-ml/res/drug_classifier/data_reg_lens/'
    # save_path = 'D:\ETH\projects\pheno-ml\\res\drug_classifier\\data_reg_lens\\'
    # collect_data_and_split(path_to_data, method='reg_lens', save_path=save_path)
    # train_drug_classifier_alone(save_path, 30, uid='with_reg_lens', device=device)

    # # collect classification results and compare: no lens vs regularizing lens
    # results = collect_and_save_lens_classification_results()
    # plot_lens_classification_results(results)

    # plot altered morphology examples
    plot_altered_morphology_maps(path_to_data, n=100)