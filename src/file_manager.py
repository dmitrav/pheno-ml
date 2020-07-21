
import os, shutil, numpy
from tqdm import tqdm


def move_files_to_another_folder(old_folder, new_folder):

    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    for file_name in tqdm(os.listdir(old_folder)):
        if file_name.endswith(".jpg"):
            shutil.move(old_folder + file_name, new_folder + file_name)


def copy_files_to_a_folder(old_folder, new_folder):

    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    for file_name in tqdm(os.listdir(old_folder)):
        if file_name.endswith(".jpg"):
            shutil.copy(old_folder + file_name, new_folder + file_name)


def move_random_files_to_another_folder(folder, new_folder, percent):

    if not os.path.exists(new_folder):
        os.makedirs(new_folder)

    number_of_files_to_move = int(len(os.listdir(folder)) * percent)
    files_to_move = numpy.random.choice(os.listdir(folder), size=number_of_files_to_move, replace=False)

    for file_name in tqdm(files_to_move):
        shutil.move(folder + file_name, new_folder + file_name)


def folders_have_equal_number_of_files(folder_1, folder_2):

    num_1 = len([x for x in os.listdir(folder_1) if not x.startswith(".")])
    num_2 = len([x for x in os.listdir(folder_2) if not x.startswith(".")])

    print(num_1, num_2, num_1 == num_2)

    return num_1 == num_2


if __name__ == "__main__":

    if False:
        path = "/Volumes/biol_imsb_sauer_1/users/Andrei/cell_line_images/batch_1/ACHN_CL3_P2/validation/class_1/"
        new_path = path.replace("class_1/","")

        move_files_to_another_folder(path, new_path)

    if False:

        old_data_path = "/Users/andreidm/ETH/projects/pheno-ml/data/batch_{}/"
        new_data_path = "/Users/andreidm/ETH/projects/pheno-ml/data/training/single_class/"

        for n in [3, 6, 7]:

            print("batch {} is being processed".format(n))
            batch_path = old_data_path.format(n)

            for cell_line_folder in os.listdir(batch_path):

                if cell_line_folder.startswith("."):
                    continue
                else:

                    print("copying from {}...".format(cell_line_folder))

                    old_path_full = batch_path + cell_line_folder + "/"
                    # new_path_full = new_data_path + "batch_{}/".format(n) + cell_line_folder + "/"
                    #
                    # if os.path.exists(new_path_full):
                    #     # if this folder exists, assume files are there already
                    #     continue
                    # else:

                    copy_files_to_a_folder(old_path_full, new_data_path)

    if False:

        path = "/Volumes/biol_imsb_sauer_1/users/Andrei/cell_line_images/batch_1/ACHN_CL3_P2/train/class_1/"
        new_path = path.replace('train', 'validation')

        move_random_files_to_another_folder(path, new_path, 0.25)

    if True:

        ending = "batch_2/IGROV1_CL2_P2/"

        original_folder = "/Volumes/biol_imsb_sauer_1/users/Mauro/Cell_culture_data/190310_LargeScreen/imageData/" + ending
        my_folder = "/Volumes/biol_imsb_sauer_1/users/Andrei/cell_line_images_2/" + ending

        folders_have_equal_number_of_files(original_folder, my_folder)
