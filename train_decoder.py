import torch
import torch.optim as optim
import torchvision.transforms
from torch.utils.data import DataLoader
from models import Composed, Decoder, reconstruction_loss
from utils import repeater, save_checkpoints, load_checkpoints, CustomDataloader30fps, load_weights
import argparse
import sys
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int,
                    default=64,
                    help='Batch size.')
parser.add_argument('--learning-rate', type=float,
                    default=1e-4,
                    help='Learning rate.')
parser.add_argument('--cnn', type=str,
                    default="normal",
                    help='The size of the CNN encoder and decoder. Choose between small, normal and large')
parser.add_argument('--hidden-dim', type=int,
                    default=512,
                    help='Number of hidden units in DeepSet MLPs rho and phi,'
                         ' as well as hidden dimension in MLP in slot-attention module.')
parser.add_argument('--feature-dim', type=int,
                    default=512,
                    help='Filter-size of the CNN')
parser.add_argument('--num-slots', type=int,
                    default=8,
                    help='Number of object slots.')
parser.add_argument('--data-path', type=str,
                    default='/home/data2/objectcentric_video/long_easy_fps30_resized_compr.h5',
                    help='Path to video (the frames) provided in a .h5 file.')
parser.add_argument('--name', type=str,
                    default='',
                    help='Add a unique name to this run. It will be added to the checkpoint file name')
parser.add_argument('--augmenting', type=bool,
                    default=False,
                    help='If set to `True`, an augmentation pipeline will increase the number of samples')
parser.add_argument('--start', type=int,
                    default=1,
                    help='Start run at this step. If start is higher than 1, the checkpoint'
                         ' with the same name will be loaded. If not available, it will throw an exception')
parser.add_argument('--end', type=int,
                    default=250_000,
                    help='The number of iterations, the model should train for')
parser.add_argument('--pretext-path', type=str,
                    default=None,
                    help='The path to the checkpoint of the learned contrastive pretext model')
parser.add_argument('--short', type=bool,
                    default=False,
                    help='Whether you like to use only 2 min from 30 min video')
args = parser.parse_args()

filename_checkpoints_decoder = f'cp_decoder_{args.num_slots}slots_{args.batch_size}batch_{args.feature_dim}fd_{args.hidden_dim}hd_{args.name}_2min_{args.short}.pth.tar'

if args.pretext_path == None:
    print('You need to specify a pth.tar file to train the decoder on')
    sys.exit(1)

name = args.name + "-" + filename_checkpoints_decoder
load_model = args.start != 1
logging = False

# Hyperparameters
input_channels = 3
cropped_resolution = (128, 128)  # epic kitchen has shape [256, 456]
weight_decay = 1e-6

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pretext_model = Composed(input_channels=input_channels, feature_dim=args.feature_dim, hidden_dim=args.hidden_dim,
                         resolution=cropped_resolution,
                         num_slots=args.num_slots, cnn=args.cnn).to(device)

decoder_model = Decoder(resolution=cropped_resolution, feature_dim=args.feature_dim, cnn=args.cnn).to(device)

for param in pretext_model.parameters():
    param.requires_grad = False

load_weights(torch.load(args.pretext_path), pretext_model)
print('loaded Base-models weight (learned from prediction task)')
print('Pretext-Model initialized')
print(pretext_model)
print('Downstream-decoder-Model initialized')
print(decoder_model)

# Data Augmentation
if args.augmenting:
    print("applying weak augmentation")
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(0.4),
        torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
        torchvision.transforms.RandomApply(torch.nn.ModuleList([
            torchvision.transforms.RandomCrop((64, 64)),
            torchvision.transforms.Resize((128, 128))
        ]), p=0.2)
    ])
else:
    print("no applying augmentation")
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

# Load Data
dataset = CustomDataloader30fps(h5_path=args.data_path, group='train', transform=transforms, short=args.short)
print(f'Data loaded, there are in total {len(dataset)} training images')
if logging:
    wandb.config.data_size = len(dataset)
train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

# Infinitely loop
train_loader = repeater(train_loader)

train_iterator = train_loader.__iter__()

# optimizer
optimizer = optim.Adam(decoder_model.parameters(), lr=args.learning_rate, weight_decay=weight_decay)

if load_model:
    load_checkpoints(torch.load(filename_checkpoints_decoder), decoder_model, optimizer)
    print(f'Checkpoint has been loaded: {filename_checkpoints_decoder}')

# Train network
print('Starting training-loop')
for step in range(args.start, args.end + 1):

    # saving checkpoints every 3 epochs
    if step % 1000 == 0 and step > 0:
        checkpoint = {
            'state_dict': decoder_model.state_dict(),  # the weights
            'optimizer_state_dict': optimizer.state_dict()  # contains buffers and parameters
        }
        save_checkpoints(checkpoint, filename_checkpoints_decoder)

    # gets out the mini_batches
    batch = train_iterator.__next__()  # [batch_size, channels, height, width]

    # data is tensor
    batch = [tensor.to(device) for tensor in batch]
    cur, prev, prev_prev = batch

    if cur.size(0) != args.batch_size:
        continue

    # no grad, just to get embeddings:
    emb_prediction = pretext_model.forward(x=cur)  # [batch_size, num_slots, feature_dim]
    # now with grad:
    pred_recon, pred_recon_slots, pred_masks = decoder_model(emb_prediction)

    loss = reconstruction_loss(pred_recon, cur)

    loss.backward()

    if step % 100 == 0:
        print(f"Step {step} with loss {loss}, current time: {datetime.now()}")

        # you might want to log: step, loss and also the images, e.g:
        # cur[0] the ground truth for first element in batch
        # pred_recon[0] the reconstructed image form all slots for first element in batch
        # and also pred_recon_slots[0, k] or pred_masks[0, k], the reconstructed slots and attention masks

    # updating the weights by using the gradients (from backward function). And set gradients back to zero
    optimizer.step()
    optimizer.zero_grad()
