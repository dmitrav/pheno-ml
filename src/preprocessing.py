
import os
from PIL import Image
from PIL import ImageFilter
from tqdm import tqdm


def shrink_images(input_path, output_path):

    target_size = (224, 224)

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


if __name__ == "__main__":

    batch = "batch_1/"

    input_path = "/Volumes/biol_imsb_sauer_1/users/Mauro/Cell_culture_data/190310_LargeScreen/imageData/" + batch
    output_path = "/Volumes/biol_imsb_sauer_1/users/Andrei/cell_line_images/" + batch

    shrink_images(input_path, output_path)