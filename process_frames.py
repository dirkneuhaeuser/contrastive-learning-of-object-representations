import os
from natsort import natsorted
import h5py
from PIL import Image
import numpy as np

path_to_train = 'path_to_train_frames'
path_to_test = 'path_to_test_frames'
h5_file_name = 'path_where_h5_shall_be_created/ds.h5'

def save_dict_h5py_image_sequence(path_to_train, path_to_test, h5_fname):
    """
    h5 containing groups/datasets which works with key/values
    :param array_dict:
    :param fname: path to the .h5 file which we want to create
    """

    # Ensure directory exists
    directory = os.path.dirname(h5_fname)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with h5py.File(h5_fname, 'w') as hf:
        train_imgs = os.listdir(path_to_train)
        train_imgs = natsorted(train_imgs)
        grp = hf.create_group('train')  # each group is an episode
        for i, p in enumerate(train_imgs):
            img_path = os.path.join(path_to_train, p)
            image = Image.open(img_path)
            image = np.asarray(image)
            grp.create_dataset(str(i), data=image, compression="gzip")

        test_imgs = os.listdir(path_to_test)
        test_imgs = natsorted(test_imgs)
        grp = hf.create_group('test')  # each group is an episode
        for i, p in enumerate(test_imgs):
            img_path = os.path.join(path_to_test, p)
            image = Image.open(img_path)
            image = np.asarray(image)
            grp.create_dataset(str(i), data=image, compression="gzip")

    print(f'images have been saved to {h5_fname}')

save_dict_h5py_image_sequence(path_to_train, path_to_test, h5_file_name)

