import torch
import torch.nn as nn
from utils import spatial_broadcast, unstack_and_split, build_grid
import torch.nn.functional as f

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LargeCNN(nn.Module):
    """
    The CNN encoder to extract features from the data. This is the large version.
    """

    def __init__(self, input_channels, feature_dim):
        """
        Args:
          feature_dim: Integer stating the filter-size of the CNN (number of kernels)
          input_channels: Integer stating the channels for the given images. Most likely 3 as RGB images
        """
        super(LargeCNN, self).__init__()

        # [batch_size, feature_dim, height/8, width/8]
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=(9, 9), stride=(1, 1),
                      padding=(4, 4)),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(7, 7), stride=(2, 2),
                      padding=(3, 3)),  # for saving memory: stride==2
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2),
                      padding=(1, 1)),  # for saving memory: stride==2
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2),
                      padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=feature_dim, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        :param x: Tensor of shape [batch_size, channels, height, width]
        :return: Tensor of shape [batch_size, features, height/8, width/8]
        """
        return self.enc(x)


class NormalCNN(nn.Module):
    """
    The CNN encoder to extract features from the data. This is the normal-sized version.
    """

    def __init__(self, input_channels, feature_dim):
        """
        Args:
          feature_dim: Integer stating the filter-size of the CNN (number of kernels)
          input_channels: Integer stating the channels for the given images. Most likely 3 as RGB images
        """
        super(NormalCNN, self).__init__()
        # [batch_size, feature_dim, height/8, width/8]
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=(7, 7), stride=(2, 2),
                      padding=(3, 3)),  # for saving memory: stride==2
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2),
                      padding=(1, 1)),  # for saving memory: stride==2
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2),
                      padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=feature_dim, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        :param x: Tensor of shape [batch_size, channels, height, width]
        :return: Tensor of shape [batch_size, features, height, width]
        """
        return self.enc(x)


class SmallCNN(nn.Module):
    """
    The CNN encoder to extract features from the data. This is the small-sized version.
    """

    def __init__(self, input_channels, feature_dim):
        """
        Args:
          feature_dim: Integer stating the filter-size of the CNN (number of kernels)
          input_channels: Integer stating the channels for the given images. Most likely 3 as RGB images
        """
        super(SmallCNN, self).__init__()
        # [batch_size, feature_dim, height/8, width/8]
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=(7, 7), stride=(2, 2),
                      padding=(3, 3)),  # for saving memory: stride==2
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(2, 2),
                      padding=(1, 1)),  # for saving memory: stride==2
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2),
                      padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=feature_dim, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            nn.ReLU(inplace=True))

    def forward(self, x):
        """
        :param x: Tensor of shape [batch_size, channels, height, width]
        :return: Tensor of shape [batch_size, features, height/8, width/8]
        """
        return self.enc(x)


class SoftPositionEmbed(nn.Module):
    """
    Adds soft positional embedding layer with learnable projection. It helps the model to understand where objects
    are. In `Object-Centric Learning with Slot Attention`, the authors say, that `the performance in predicting
    the object position clearly decreases if we remove the position embedding`.
    """

    def __init__(self, feature_dim, resolution):
        """
        Args:
          feature_dim: feature embedding from cnn
          resolution: Tuple of integers specifying height and width of the images. Eg (256, 456) for a 256 x 456 image.
        """
        super(SoftPositionEmbed, self).__init__()

        self.ln = nn.Linear(in_features=4, out_features=feature_dim)  # bias is always set to true
        self.grid = build_grid(resolution, device)  # 1024 = 32 x 32

    def forward(self, inputs):
        """
        :param inputs: Tensor of shape [batch_size, height*, width*, feature_dim]
        :return:
            Tensor of shape [batch_size, height, width, feature_dim]
            (positional enriched tensor and changed permutation)
        """
        # permute the features to the right
        x = self.grid  # [1, height, width, 4]
        x = self.ln(x)  # [1, height, width, feature_dim]
        return inputs + x  # [batch_size, height, width, feature_dim] + [1, height, width, feature_dim] =  [batch_size, height, width, feature_dim]


class Encoder(nn.Module):
    """Encodes the image and enriches with positional embeddings"""

    def __init__(self, feature_dim, resolution, input_channels, cnn):  # channels in rgb is 3, here is black/white,so 1
        """
        Args:
          feature_dim: Integer stating the filtersize of the CNN (number of kernels)
          resolution: Tuple of integers specifying height and width of the images. Eg (256, 456) for a 256 x 456 image.
          input_channels: Integer stating the channels for the given images. Most likely 3 as RGB images
        """
        super(Encoder, self).__init__()

        # the CNNs down-sample the images from 128 ->16, which is 1/8
        resolution = (resolution[0] // 8, resolution[1] // 8)
        self.resolution = resolution

        if cnn == "small":
            self.cnn = SmallCNN(input_channels=input_channels, feature_dim=feature_dim)
        elif cnn == "normal":
            self.cnn = NormalCNN(input_channels=input_channels, feature_dim=feature_dim)
        elif cnn == "large":
            self.cnn = LargeCNN(input_channels=input_channels, feature_dim=feature_dim)
        else:
            raise TypeError()

        self.position_embedding = SoftPositionEmbed(feature_dim,
                                                    (resolution[0], resolution[1]))  # 64 from CNN, (32, 32) resolution
        self.ln = nn.LayerNorm(normalized_shape=feature_dim)  # 64 is size of last dimension
        self.mlp = nn.Sequential(
            nn.Linear(in_features=feature_dim, out_features=feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=feature_dim, out_features=feature_dim))

    def forward(self, x):
        """
        :param x: Tensor of shape [batch_size, channels, height, width]
        :return: Tensor of shape  [batch_size, resolution*, feature_dim],
        where * means down-sampled width/height
        """
        x = self.cnn(x)  # if not resnet: [batch_size, feature_dim, height*, width*]
        x = x.permute(0, 2, 3, 1)  # [batch_size, height*, width*, feature_dim]
        x = self.position_embedding(x)  # [batch_size, height*, width*, feature_dim]
        x = x.view(x.size(0), -1, x.size(3))  # [batch_size, resolution*, feature_dim]
        x = self.ln(x)
        x = self.mlp(x)  # [batch_size, resolution*, feature_dim]
        return x


class SlotAttentionAdjusted(nn.Module):
    """
    Module analogously to https://github.com/lucidrains/slot-attention/blob/master/slot_attention/slot_attention.py
    In accordance with object-centric video module however, the slots in this module however are learnable parameters.
    This leads to having a fixed amount of objects to detect, in contrast to the original implementation,
    Also for each frame we just update the slot with 1 iteration
    """

    def __init__(self, num_slots, feature_dim, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = feature_dim ** -0.5

        slots_mu = torch.randn(1, num_slots, feature_dim)
        slots_sigma = torch.randn(1, num_slots, feature_dim)

        # in contrast to the original paper, we want the slots to be learned not the mu/sigma specialisation
        self.slots = torch.nn.Parameter(torch.normal(slots_mu, slots_sigma))

        self.to_q = nn.Linear(feature_dim, feature_dim)
        self.to_k = nn.Linear(feature_dim, feature_dim)
        self.to_v = nn.Linear(feature_dim, feature_dim)

        self.gru = nn.GRUCell(feature_dim, feature_dim)

        hidden_dim = max(feature_dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim)
        )

        self.norm_input = nn.LayerNorm(feature_dim)
        self.norm_slots = nn.LayerNorm(feature_dim)
        self.norm_pre_ff = nn.LayerNorm(feature_dim)

    def forward(self, inputs):
        """
        :param inputs: Tensor of shape [batch_size, resolution*, feature_dim]
        :return: Tensor of shape [batch_size, num_slots, feature_dim]
        """
        batch_size, resolution, feature_dim = inputs.shape

        #  expand out learned slots to meet the batch_size and use the same for all batch_elements
        slots = self.slots.expand(batch_size, self.num_slots, feature_dim)  # [batch, num_slots, feature_dim]
        inputs = self.norm_input(inputs)  # [batch_size, resolution*, feature_dim]
        k, v = self.to_k(inputs), self.to_v(inputs)  # [batch_size, resolution*, feature_dim]

        for _ in range(self.iters):
            slots_prev = slots  # [batch, num_slots, feature_dim]

            slots = self.norm_slots(slots)
            q = self.to_q(slots)  # [batch, num_slots, feature_dim]

            # [batch, num_slots, resolution]: mm with feature_dim as common axis
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps  # softmax and competition over slots
            attn = attn / attn.sum(dim=-1, keepdim=True)  # [batch, num_slots, resolution]

            # [batch, num_slots, feature_dim]: mm with resolution as common axis
            updates = torch.einsum('bjd,bij->bid', v, attn)

            #  [batch * num_slots, feature_dim]  (here slots are shared parameters)
            slots = self.gru(updates.reshape(-1, feature_dim), slots_prev.reshape(-1, feature_dim))

            #  [batch * num_slots, feature_dim]  (undo sharing of parameters)
            slots = slots.reshape(batch_size, -1, feature_dim)

            slots = slots + self.mlp(self.norm_pre_ff(slots))  # optional residual MLP

        return slots


class TransitionModel(nn.Module):
    """
    The transition model learns to predict the representations at the next time-step as specified
    in 'Learning Object-Centric Video Models by Contrasting Sets'

    See: https://arxiv.org/abs/2011.10287

    It says:
    pkt+2 = skt+1 + f_transition([skt , skt+1, skt+1 - skt ]),
    where [·] denotes concatenation of vectors.
    The transition function f_transition consists of three steps: a linear down-projection, LayerNorm and
    a linear transformation.
    """
    def __init__(self, feature_dim):
        super().__init__()
        concatenated_dim = feature_dim * 3
        self.down_projection = nn.Linear(concatenated_dim, feature_dim)
        self.ln = nn.LayerNorm(feature_dim)
        self.lin = nn.Linear(feature_dim, feature_dim)

    def forward(self, prev, prev_prev):
        """
        :param prev: latent tensor with shape [batch_size, num_slots, feature_dim]
        :param prev_prev: latent tensor with shape [batch_size, num_slots, feature_dim]
        :return: tensor of shape [batch_size, num_slots, feature_dim], which is the prediction for the next step
        """

        diff = prev - prev_prev  # diff tensor containing (t_1 - t_0, t_2 - t_1, ..., t_63 - t_62)
        composed = torch.cat([prev_prev, prev, diff], dim=-1)  # [batch_size, slots, features * 3]

        batch_size = composed.size(0)
        composed = composed.view(-1, composed.size(2))  # [(batch_size)*slots, features * 3]
        delta = self.down_projection(composed)  # [(batch_size)*slots, features]
        delta = self.ln(delta)
        delta = self.lin(delta)

        # revert parameter sharing:
        delta = delta.view(batch_size, -1, delta.size(1))  # [(batch_size), slots, features]
        return prev + delta


class DeepSet(nn.Module):
    """
    As Object-slots don't have a specific order, a normal MLP to predict next state can't be applied, as it operates
    on fixed dimensional vectors. In 'Learning Object-Centric Video Models by Contrasting Sets'
    (https://arxiv.org/abs/2011.10287), the authors use a DeepSet Network
    (https://papers.nips.cc/paper/2017/file/f22e4747da1aa27e363d86d40ff442fe-Paper.pdf). This DeepSet Network learns
    a representation of the input which is invariant of the order (as sets don't have an order).
    An invariant function output is the same, no matter the order of the input.

    In Compliance with 'Learning Object-Centric Video Models by Contrasting Sets', the DeepSet Network will be:
    MLP(LayerNorm(Sum over k (MLP(slot_k))))

    Implementation analogously to the original one:
    https://github.com/manzilzaheer/DeepSets/blob/master/PopStats/model.py
    phi: learned feature transform
    rho: processes the summed representations
    """

    def __init__(self, feature_dim, hidden_dim):
        """
        :param feature_dim: latent feature dimension
        :param hidden_dim: hidden dim of the MLPs phi and rho
        """
        super(DeepSet, self).__init__()

        self.phi = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
        )

        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        :param x: Tensor of shape [batch_size, slots, features]
        :return: Tensor of shape [batch_size, features],
        A representation of all slots invariant of the slot-order (slots get summed out)
        """
        x = self.phi(x)  # [batch_size, slots, hidden_dim]
        x = torch.sum(x, dim=1)  # [batch_size, hidden_dim]
        x = self.ln(x)
        x = self.rho(x)  # [batch_size, features]
        return x


def contrastive_loss(encoded_prediction, encoded_truth):
    """
    Taking each image and its prediction for each element in the batch,
    this method calculates the InfoNCE loss, as used by the authors of 'Learning Object-Centric
    Video Models by Contrasting Sets' (https://arxiv.org/abs/2011.10287). The InfoNCE loss originally suggested in
    'Representation Learning with Contrastive Predictive Coding' (https://arxiv.org/abs/1807.03748) contrasts the
    ground trugh image (positive example) and its prediction (anchor) against all other images and prediction in the
    batch (negative examples)
    :param encoded_truth: tensor of shape [(batch_size-2), features]
    :param encoded_prediction: tensor of shape [(batch_size-2), features]
    :return: float stating the loss
    """
    # convert to double, because otherwise values will go too low,
    # with exponential of this value gets rounded to 0, log will -inf
    # https://discuss.pytorch.org/t/nan-loss-coming-after-some-time/11568

    encoded_prediction = encoded_prediction.double()
    encoded_truth = encoded_truth.double()

    encoded_prediction = f.normalize(encoded_prediction, p=2)
    encoded_truth = f.normalize(encoded_truth, p=2)

    temperature = 0.5
    mm = encoded_prediction @ torch.t(encoded_truth)  # (batch_size x batch_size) z_p times z_s

    nominator = torch.diag(mm)
    nominator = nominator / temperature
    denominator = mm / temperature

    nominator = torch.exp(nominator)  # [(batch_size-2),(batch_size-2)]
    denominator = torch.sum(torch.exp(denominator), dim=1)

    mm2 = encoded_prediction @ torch.t(encoded_prediction)
    denominator2 = mm2.fill_diagonal_(0)
    denominator2 = denominator2 / temperature
    denominator2 = torch.sum(torch.exp(denominator2), dim=1)

    return -torch.mean(torch.log(nominator / (denominator + denominator2)))


class SmallDecoderCNN(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(in_channels=feature_dim, out_channels=256, kernel_size=(3, 3),
                               stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(3, 3),
                               stride=(2, 2), padding=(1, 1), output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(3, 3),
                               stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 3),
                               stride=(2, 2), padding=(1, 1), output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=4, kernel_size=(7, 7),
                               stride=(2, 2), padding=(3, 3), output_padding=1)
        )

    def forward(self, x):
        """
        :param x: Tensor of shape [batch_size*num_slots, height_init, width_init, features]
        :return: Tensor of shape [batch_size*num_slots, 4, height, width]
        """
        x = x.permute(0, 3, 1, 2)  # [batch_size*num_slots, features, height, width]
        x = self.dec(x)
        return x


class NormalDecoderCNN(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=(3, 3),
                               stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=(3, 3),
                               stride=(1, 1), padding=(1, 1)),
            nn.ConvTranspose2d(in_channels=feature_dim, out_channels=256, kernel_size=(3, 3),
                               stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(3, 3),
                               stride=(2, 2), padding=(1, 1), output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(3, 3),
                               stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 3),
                               stride=(2, 2), padding=(1, 1), output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=4, kernel_size=(7, 7),
                               stride=(2, 2), padding=(3, 3), output_padding=1),
        )


    def forward(self, x):
        """
        Maps latent space towards decoded images containing also an alpha channel
        :param x: Tensor of shape [batch_size*num_slots, height_init, width_init, features]
        :return: Tensor of shape [batch_size*num_slots, 4, height, width]
        """
        x = x.permute(0, 3, 1, 2)  # [batch_size*num_slots, features, height, width]
        x = self.dec(x)
        return x


class LargeDecoderCNN(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=(3, 3),
                               stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=feature_dim, out_channels=feature_dim, kernel_size=(3, 3),
                               stride=(1, 1), padding=(1, 1)),
            nn.ConvTranspose2d(in_channels=feature_dim, out_channels=256, kernel_size=(3, 3),
                               stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(3, 3),
                               stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(3, 3),
                               stride=(2, 2), padding=(1, 1), output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=(3, 3),
                               stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(3, 3),
                               stride=(2, 2), padding=(1, 1), output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(7, 7),
                               stride=(2, 2), padding=(3, 3), output_padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=4, kernel_size=(9, 9),
                               stride=(1, 1), padding=(4, 4))
        )

    def forward(self, x):
        """
        Maps latent space towards decoded images containing also an alpha channel
        :param x: Tensor of shape [batch_size*num_slots, height_init, width_init, features]
        :return: Tensor of shape [batch_size*num_slots, 4, height, width]
        """
        x = x.permute(0, 3, 1, 2)  # [batch_size*num_slots, features, height, width]
        x = self.dec(x)
        return x


class Decoder(nn.Module):
    def __init__(self, resolution, feature_dim, cnn):
        super().__init__()
        self.resolution = resolution
        self.decoder_initial_size = (resolution[0] // 8, resolution[1] // 8)
        self.position_embedding = SoftPositionEmbed(feature_dim,
                                                    (self.decoder_initial_size[0], self.decoder_initial_size[1]))

        if cnn == "small":
            self.decoder_cnn = SmallDecoderCNN(feature_dim=feature_dim)
        elif cnn == "normal":
            self.decoder_cnn = NormalDecoderCNN(feature_dim=feature_dim)
        elif cnn == "large":
            self.decoder_cnn = LargeDecoderCNN(feature_dim=feature_dim)
        else:
            raise TypeError()
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        """
        :param x: Tensor of shape [batch_size, slots, features]
        :return:
            recon: The overall reconstructed image with shape [batch_size, 3, height, width]
            recon_slots: The slot-base for the reconstructed image with shape [batch_size, num_slots, 3, height, width]
            masks: The mask-base for the reconstructed image with shape [batch_size, num_slots, 1, height, width]
            Note: Summing over recon_slots and weighting it with masks lead to recon. In the process, the `num_slots`
                dimension is summed out.
        """

        batch_size = x.size(0)
        # [batch_size*num_slots, height_init, width_init, features] (share parameters!)
        x = spatial_broadcast(x, self.decoder_initial_size)

        x = self.position_embedding(x)  # [batch_size*num_slots, height_init, width_init, features]
        x = self.decoder_cnn(x)  # [batch_size*num_slots, 4, height, width]
        recons, masks = unstack_and_split(x, batch_size)  # [batch_size, num_slots, 3 or 1, height, width]
        masks = self.sm(masks)
        recon_slots = recons * masks  # [batch_size, num_slots, 3, height, width]
        recon = torch.sum(recon_slots, dim=1)  # [batch_size, 3, height, width]
        return recon, recon_slots, masks


class Composed(nn.Module):
    def __init__(self, input_channels, feature_dim, hidden_dim, resolution, num_slots, cnn):
        super(Composed, self).__init__()
        self.encoder = Encoder(feature_dim=feature_dim, resolution=resolution, input_channels=input_channels, cnn=cnn)
        self.attn = SlotAttentionAdjusted(feature_dim=feature_dim, num_slots=num_slots, hidden_dim=hidden_dim)
        self.transition = TransitionModel(feature_dim=feature_dim)
        self.deep_set = DeepSet(feature_dim=feature_dim, hidden_dim=hidden_dim)

    def forward(self, x):
        """
        :param x: Tensor of shape [batch_size, channels, height, width]
        :return: Tensor of shape [batch_size, num_slots, feature_dim] (latent representations)
        """
        return self.attn(self.encoder(x))

    def predict(self, current, prev, prev_prev):
        """
        :param current: Tensor of shape [batch_size, channels, height, width]
        :param prev: Tensor of shape [batch_size, channels, height, width]
        :param prev_prev: Tensor of shape [batch_size, channels, height, width]
        :return: latent anchor (prediction) and positive example(current, which is the ground truth):
        enriched by the slots (no deepset applied). Both have shape [batch_size, slots, feature_dim]
        """

        # After Encoder: [batch_size, resolution*, feature_dim]
        current = self.encoder(current)
        prev = self.encoder(prev)
        prev_prev = self.encoder(prev_prev)

        # After Slot Attention: [batch_size, num_slots, feature_dim]
        current = self.attn(current)
        prev = self.attn(prev)
        prev_prev = self.attn(prev_prev)

        prediction = self.transition(prev=prev, prev_prev=prev_prev)  # [batch_size, num_slots, features]

        return current, prediction

    def contrasting(self, current, prev, prev_prev):
        """
        :param current: Tensor of shape [batch_size, channels, height, width]
        :param prev: Tensor of shape [batch_size, channels, height, width]
        :param prev_prev: Tensor of shape [batch_size, channels, height, width]
        :return: loss
        """
        # After Encoder: [batch_size, resolution*, feature_dim]
        current = self.encoder(current)
        prev = self.encoder(prev)
        prev_prev = self.encoder(prev_prev)

        # After Slot Attention: [batch_size, num_slots, feature_dim]
        current = self.attn(current)
        prev = self.attn(prev)
        prev_prev = self.attn(prev_prev)

        prediction = self.transition(prev=prev, prev_prev=prev_prev)  # [batch_size, num_slots, features]

        # After Deep-Set: [batch_size, feature_dim]
        current = self.deep_set(current)
        prediction = self.deep_set(prediction)

        loss = contrastive_loss(encoded_prediction=prediction, encoded_truth=current)
        return loss


def reconstruction_loss(prediction, target):
    return torch.mean((prediction - target) ** 2)
