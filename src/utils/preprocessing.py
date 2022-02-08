
import os, numpy, shutil
from multiprocessing import Process
from PIL import Image
from PIL import ImageFilter
from tqdm import tqdm


def shrink_images(input_path, output_path):

    target_size = (256, 256)

    for cell_line_folder in tqdm(os.listdir(input_path)):

        if cell_line_folder.startswith('.'):
            continue
        else:
            # make same directory in output folder
            if not os.path.exists(output_path + cell_line_folder):
                os.makedirs(output_path + cell_line_folder)

            for image_name in tqdm(os.listdir(input_path + cell_line_folder)):

                if image_name.startswith("."):
                    continue
                else:
                    image = Image.open(input_path + cell_line_folder + '/' + image_name)

                    image = image.resize(target_size)
                    image = image.convert("L")  # convert to grey scale
                    image = image.filter(ImageFilter.SHARPEN)  # sharpen 2 times
                    image = image.filter(ImageFilter.SHARPEN)

                    image.save(output_path + cell_line_folder + '/' + image_name.replace('.tif', '.jpg'), "JPEG")


def find_and_shrink_remaining_images(input_path, output_path):
    """ Full data set could not be processed at once because of network interruptions.
        This method searches for the pictures that were not yet processed, gets them processed
        and saves the result to the folder of this batch and cell line. """

    target_size = (256, 256)

    for cell_line_folder in tqdm(os.listdir(input_path)):

        if cell_line_folder.startswith('.'):
            continue
        else:
            # make same directory in output folder
            if not os.path.exists(output_path + cell_line_folder):
                os.makedirs(output_path + cell_line_folder)

            for image_name in tqdm(os.listdir(input_path + cell_line_folder)):

                if image_name.startswith(".") or not image_name.endswith('.tif'):
                    continue
                else:

                    if os.path.exists(output_path + cell_line_folder + '/' + image_name.replace('.tif', '.jpg')):
                        # if this image already exists in output folder in jpeg format, then it was processed already
                        continue
                    else:

                        image = Image.open(input_path + cell_line_folder + '/' + image_name)

                        image = image.resize(target_size)
                        image = image.convert("L")  # convert to grey scale
                        image = image.filter(ImageFilter.SHARPEN)  # sharpen 2 times
                        image = image.filter(ImageFilter.SHARPEN)

                        # image.save(output_path + cell_line_folder + '/' + image_name.replace('.tif', '.jpg'), "JPEG")


def find_and_crop_remaining_images(input_path, output_path):
    """ Full data set could not be processed at once because of network interruptions.
        This method searches for the pictures that were not yet processed, gets them processed
        and saves the result to the folder of this batch and cell line.

        NB: this method CROPS images in the center (as opposed to previous method which was resizing the entire image). """

    target_dim = 256  # output is a square of this size

    for cell_line_folder in tqdm(os.listdir(input_path)):

        if cell_line_folder.startswith('.'):
            continue

        # debug: only process a single folder in a batch
        elif cell_line_folder != 'SW620_CL1_P2':
            continue

        else:
            # make same directory in output folder
            if not os.path.exists(output_path + cell_line_folder):
                os.makedirs(output_path + cell_line_folder)

            for image_name in tqdm(os.listdir(input_path + cell_line_folder)):

                if image_name.startswith(".") or not image_name.endswith('.tif'):
                    continue
                else:

                    if os.path.exists(output_path + cell_line_folder + '/' + image_name.replace('.tif', '.jpg')):
                        # if this image already exists in output folder in jpeg format, then it was processed already
                        continue
                    else:

                        image = Image.open(input_path + cell_line_folder + '/' + image_name)

                        a = numpy.array(image)  # get an array to crop the image
                        cropped_array = a[(a.shape[0] // 2 - target_dim // 2):(a.shape[0] // 2 + target_dim // 2),
                                            (a.shape[1] // 2 - target_dim // 2):(a.shape[1] // 2 + target_dim // 2)]

                        cropped_image = Image.fromarray(cropped_array)  # get back the image object
                        sharped_image = cropped_image.filter(ImageFilter.SHARPEN)  # one filter is enough

                        sharped_image.save(output_path + cell_line_folder + '/' + image_name.replace('.tif', '.jpg'), "JPEG")


def copy_control_and_drug_images():
    """ This is to formulate a balanced binary classification problem. """

    # copy from
    dmso_ids_path = 'D:\ETH\projects\morpho-learner\data\cut_controls\\'
    drugs_ids_path = 'D:\ETH\projects\morpho-learner\data\cut\\'
    full_data_path = 'D:\ETH\projects\pheno-ml\data\cropped\\'

    # copy to
    controls_path = 'D:\ETH\projects\pheno-ml\data\classification_controls\\'
    drugs_path = 'D:\ETH\projects\pheno-ml\data\classification_drugs\\'

    # get ids of images
    dmso_ids = list(set([x.split('DMSO')[0][:-1] for x in os.listdir(dmso_ids_path)]))
    pbs_ids = list(set([x.split('PBS')[0][:-1] for x in os.listdir(drugs_ids_path) if 'PBS' in x]))
    drugs_ids = list(set(['_'.join(x.split('_')[:4]) for x in os.listdir(drugs_ids_path) if x.split('_')[4] != 'PBS']))
    control_ids = [*dmso_ids, *pbs_ids]

    all_files = sorted(os.listdir(full_data_path))
    nc, nd = 0, 0
    i = 0
    while i < len(all_files):

        index = '_'.join(all_files[i].split('_')[:4])
        j = i + 1
        while '_'.join(all_files[j].split('_')[:4]) == index:
            j += 1
            if j == len(all_files):
                break

        # calculate how many images will be in each category
        if index in control_ids:
            nc += j-i
        elif index in drugs_ids:
            nd += 40
        else:
            pass

        if index in control_ids:
            # copy j-i DMSO or PBS images
            for k in range(i, j):
                shutil.copy(full_data_path + all_files[k], controls_path + all_files[k])
        elif index in drugs_ids:
            # copy last 40 drug images to have a balanced drug vs control dataset
            for k in range(j-40, j):
                shutil.copy(full_data_path + all_files[k], drugs_path + all_files[k])
        else:
            # drugs of low concentrations are of no interest
            # to the classification problem, as they show no effect
            pass

        i += j-i

    print('copying is complete')
    print('controls:', nc)
    print('drugs:', nd)


if __name__ == "__main__":

    copy_control_and_drug_images()

    if False:
        # batches 1, 3, 4, 6, 7 have been processed successfully
        for n in [2, 5]:

            batch = "batch_{}/".format(str(n))

            input_path = "/Volumes/biol_imsb_sauer_1/users/Mauro/Cell_culture_data/190310_LargeScreen/imageData/" + batch
            output_path = "/Volumes/biol_imsb_sauer_1/users/Andrei/cell_line_images_2/" + batch

            # p = Process(target=shrink_images, args=(input_path, output_path))
            # find_and_shrink_remaining_images(input_path, output_path)

            p = Process(target=find_and_shrink_remaining_images, args=(input_path, output_path))
            p.start()

    if False:

        # for n in [1, 2, 3, 4, 5, 6, 7]:
        for n in [3]:
            batch = "batch_{}/".format(str(n))

            input_path = "/Volumes/biol_imsb_sauer_1/users/Mauro/Cell_culture_data/190310_LargeScreen/imageData/" + batch
            # output_path = "/Volumes/biol_imsb_sauer_1/users/Andrei/cell_line_images/" + batch
            output_path = "/Users/andreidm/ETH/projects/pheno-ml/data/cropped/" + batch

            p = Process(target=find_and_crop_remaining_images, args=(input_path, output_path))
            p.start()

