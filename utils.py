import torch
from itertools import repeat
from torch.utils.data import Dataset
import numpy as np
import h5py
import random


def save_checkpoints(checkpoint, filename_checkpoints):
    print("=> Saving checkpoints")
    torch.save(checkpoint, filename_checkpoints)


def load_checkpoints(checkpoint, model, optimizer=None):
    print("=> Loading checkpoints")
    model.load_state_dict((checkpoint['state_dict']))
    if optimizer is not None:
        optimizer.load_state_dict((checkpoint['optimizer_state_dict']))


def load_weights(checkpoint, model):
    print("=> Loading weights")
    model.load_state_dict((checkpoint['state_dict']))


def repeater(data_loader):
    """
    To draw unlimitedly from the dataloader
    """
    for loader in repeat(data_loader):
        for _data in loader:
            yield _data


class CustomDataloader30fps(Dataset):
    """
    In this set up, we are loading (batch-size) random frames. Each along with its previous and the previous of that. As
    we used 30 fps, we opted to leave out three frames each time. (cur_image at index idx and the previous is at idx-4)
    Note: In the original-paper `Object-centric learning of video models`, the batch-size referred to sequences.
    For example, the bouncing-ball dataset contained 1000 sequences of 12 frames. The authors then included (batch-size)
    sequences taking all 12 frames. As we consider large video instead of short sequences in this work, we wil change
    that and draw single random images with its predecessors.
    """
    def __init__(self, h5_path, group, transform=None, short=False):
        self.np_images_in_array = load_list_h5py_image_sequence(h5_path, group)
        self.transform = transform
        self.short = short

    def __len__(self):
        if self.short:
            # as input videos were 30min long, dividing by 15 -> 2min
            return len(self.np_images_in_array)//15 - 8
        return len(self.np_images_in_array) - 8

    def __getitem__(self, idx):
        idx += 8
        cur_image = self.np_images_in_array[idx]
        prev_image = self.np_images_in_array[idx-4]
        prev_prev_image = self.np_images_in_array[idx-8]

        # applying data augmentation to all three frames in the same way
        seed = np.random.randint(2147483647) # make a seed with numpy generator

        if self.transform:
            # make all three frames have the same data augmentation
            # https://pytorch.org/vision/0.8/transforms.html
            # https://github.com/pytorch/vision/issues/9

            # visualise the transform with save_image_np before transform and save_image_torch after transform
            random.seed(seed)
            torch.manual_seed(seed)
            cur_image = self.transform(cur_image)
            random.seed(seed)
            torch.manual_seed(seed)
            prev_image = self.transform(prev_image)
            random.seed(seed)
            torch.manual_seed(seed)
            prev_prev_image = self.transform(prev_prev_image)

        return cur_image, prev_image, prev_prev_image


def load_list_dict_h5py(fname):
    """Restore list of dictionaries containing numpy arrays from h5py file."""
    array_dict = list()
    with h5py.File(fname, 'r') as hf:
        for i, grp in enumerate(hf.keys()):
            array_dict.append(dict())
            for key in hf[grp].keys():
                array_dict[i][key] = hf[grp][key][:]
    return array_dict


def spatial_broadcast(slots, resolution):
    """
    :param slots: tensor of shape [batch_size, slots, features]
    :param resolution: tuple of integers (height, width)f
    :return: tensor of shape [batch_size*num_slots, height, width, features], the first dimension just reshaped
    and for height and width the array is copied multiple times
    """
    slots = slots.view(-1, slots.size(2))  # [(batch_size-2)*slots, features]
    slots = torch.unsqueeze(slots, 1)
    slots = torch.unsqueeze(slots, 1)  # [(batch_size-2)*slots,1, 1 features]
    slots = slots.expand(slots.size(0), resolution[0], resolution[1], slots.size(3))
    return slots


def unstack_and_split(x, batch_size, num_channels=3):
    """
    Unstack batch dimension and split into channels and alpha mask.
    :param x: Tensor of shape  # [(batch_size-2)*num_slots, 4, height, width]
    :param batch_size: Integer indicating (batch_size-2)
    :param num_channels: the channes of the orginal image, most likely 3 for a RGB image
    :return: 2 Tensors:
        channels of Shape [(batch_size-2), num_slots, 3, height, width] and
        masks of shape  [(batch_size-2), num_slots, 1, height, width]

    """
    unstacked = x.view(batch_size, -1, x.size(1), x.size(2), x.size(3))  # [(batch_size-2), num_slots, 4, height, width]
    channels, masks = torch.split(unstacked, [num_channels, 1], dim=2)  # [(batch_size-2), num_slots, 3 and 1, height, width]
    return channels, masks


def build_grid(resolution, device):
    """
    Building the grid for linear embedding
    :param device: As we create a new tensor, we need to put it on the device as well
    :param resolution: tuple of integers (height, width)
    :return: Tensor of shape [1, height, height, 4]
    """
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    grid = np.concatenate([grid, 1.0 - grid], axis=-1)
    grid = torch.from_numpy(grid)
    grid = grid.to(device)
    return grid


def load_list_h5py_image_sequence(fname, group):
    """Restore list of dictionaries containing numpy arrays from h5py file."""
    with h5py.File(fname, 'r') as hf:
        num_elements = len(hf[group].keys())
        array_dict = [[] for _ in range(num_elements)]
        for key in hf[group].keys():
            array_dict[int(key)] = hf[group][key][:]
    return array_dict