
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


if __name__ == "__main__":

    if False:
        path = "/Volumes/biol_imsb_sauer_1/users/Andrei/cell_line_images/batch_1/ACHN_CL3_P2/validation/class_1/"
        new_path = path.replace("class_1/","")

        move_files_to_another_folder(path, new_path)

    if False:
        path = "/Volumes/biol_imsb_sauer_1/users/Andrei/cell_line_images/batch_1/ACHN_CL3_P2/single_class/"
        new_path = "/Users/andreidm/ETH/projects/pheno-ml/data/ACHN_CL3_P2/single_class/"

        copy_files_to_a_folder(path, new_path)

    if False:
        path = "/Volumes/biol_imsb_sauer_1/users/Andrei/cell_line_images/batch_1/ACHN_CL3_P2/train/class_1/"
        new_path = path.replace('train', 'validation')

        move_random_files_to_another_folder(path, new_path, 0.25)