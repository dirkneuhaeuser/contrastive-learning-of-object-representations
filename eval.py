import torch
import torchvision.transforms
from torch.utils.data import DataLoader  # handy to get mini batches
import wandb
from models import Composed, Decoder, reconstruction_loss
from utils import load_weights, CustomDataloader30fps
import sys
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int,
                    default=64,
                    help='Batch size.')
parser.add_argument('--hidden-dim', type=int,
                    default=128,
                    help='Number of hidden units in DeepSet MLPs rho and phi,'
                         ' as well as hidden dimension in MLP in slot-attention module.')
parser.add_argument('--cnn', type=str,
                    default="normal",
                    help='The size of the CNN encoder and decoder. Choose between small, normal and large')
parser.add_argument('--feature-dim', type=int,
                    default=64,
                    help='Filter-size of the CNN')
parser.add_argument('--num-slots', type=int,
                    default=5,
                    help='Number of object slots.')
parser.add_argument('--data-path', type=str,
                    default='/',
                    help='Path to the video (frames) provided in a .h5 file.')
parser.add_argument('--name', type=str,
                    default='',
                    help='Add a remark to this run')
parser.add_argument('--pretext-path', type=str,
                    default=None,
                    help='The path to the checkpoint of the learned contrastive pretext model')
parser.add_argument('--decoder-path', type=str,
                    default=None,
                    help='The path to the checkpoint of the learned downstraem decoder')
parser.add_argument('--short', type=bool,
                    default=False,
                    help='Whether you like to use only 2 min from 30 min video')
args = parser.parse_args()

if args.pretext_path == None or args.decoder_path is None:
    print('You need to specify a pth.tar-file for both, the trained pretext model and the trained downstream model. ')
    sys.exit(1)

name = 'Eval ' +args.name+"-" + args.decoder_path
logging = False

# Hyperparameters
input_channels = 3
cropped_resolution = (128, 128)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pretext_model = Composed(input_channels=input_channels, feature_dim=args.feature_dim, hidden_dim=args.hidden_dim,
                         resolution=cropped_resolution,
                         num_slots=args.num_slots, cnn=args.cnn).to(device)

decoder_model = Decoder(resolution=cropped_resolution, feature_dim=args.feature_dim, cnn=args.cnn).to(device)



load_weights(torch.load(args.pretext_path), pretext_model)
load_weights(torch.load(args.decoder_path), decoder_model)
print('loaded Base-models weight (learned from prediction task)')



if logging:
    os.environ['WANDB_API_KEY'] = 'ef94a92678c9a088899b27fb3eb2ca4b7c19642c'
    wandb.init(project='HO(eval, sorted + MSE)', name=name)
    wandb.watch(decoder_model)

print('Pretext-Model initialized')
print(pretext_model)
print('Downstream-decoder-Model initialized')
print(decoder_model)

# Load Data
transforms = torchvision.transforms.Compose([
    # torchvision.transforms.CenterCrop(cropped_resolution),
    torchvision.transforms.ToTensor()
])
dataset = CustomDataloader30fps(h5_path=args.data_path, group='test', transform=transforms, short=args.short)
print(f'Data loaded, there are in total {len(dataset)} testing images')
if logging:
    wandb.config.data_size = len(dataset)
test_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=4)  # turn of shuffle for eval
test_iterator = test_loader.__iter__()

print('Starting eval-loop')
pretext_model.eval()
decoder_model.eval()

with torch.no_grad():
    denominator = len(test_loader)
    losses = []
    for step in range(len(test_loader)):

        # gets out the mini_batches
        batch = test_iterator.__next__()  # [batch_size, channels, height, width]

        # data is tensor
        batch = [tensor.to(device) for tensor in batch]
        cur, prev, prev_prev = batch

        # no grad, just to get embeddings:
        emb = pretext_model.forward(x=cur)  # [batch_size, num_slots, feature_dim]

        # here with grad:
        pred_recon, pred_recon_slots, pred_masks = decoder_model(emb)

        loss_pred = reconstruction_loss(pred_recon, cur)
        losses.append(loss_pred)

        print(f"Step {step} with pred-loss {loss_pred}")

        # you might want to log: step, loss and also the images, e.g:
        # cur[0] the ground truth for first element in batch
        # pred_recon[0] the reconstructed image form all slots for first element in batch
        # and also pred_recon_slots[0, k] or pred_masks[0, k], the reconstructed slots and attention masks

    mean = sum(losses)/len(losses)
    deviation = (sum([(l - mean)**2 for l in losses]) / len(losses))**0.5
    wandb.log({'MSE-Mean': mean,
               'MSE-Deviation': deviation,
               })


