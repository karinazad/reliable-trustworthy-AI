import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time

from torchvision import datasets, transforms
from tensorboardX import SummaryWriter
from model import Net, ConvNet

use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")
batch_size = 64

import matplotlib.pyplot as plt

# hard-code random seeds for deterministic outcomes
np.random.seed(42)
torch.manual_seed(42)

# loading the dataset
# note that this time we do not perfrom the normalization operation, see next cell
test_dataset = datasets.MNIST('mnist_data/', train=False, download=True, transform=transforms.Compose(
    [transforms.ToTensor()]
))


class Normalize(nn.Module):
    def forward(self, x):
        return (x - 0.1307) / 0.3081


# we load the body of the pre-trained neural net...
model = torch.load('model.net', map_location='cpu')

# ... and add the data normalization as a first "layer" to the network
# this allows us to search for adverserial examples to the real image, rather than
# to the normalized image
model = nn.Sequential(Normalize(), model)

# and here we also create a version of the model that outputs the class probabilities
model_to_prob = nn.Sequential(model, nn.Softmax())

# we put the neural net into evaluation mode (this disables features like dropout)
model.eval()
model_to_prob.eval()


# define a show function for later
def show(original, adv, model_to_prob):
    p0 = model_to_prob(original).detach().numpy()
    p1 = model_to_prob(adv).detach().numpy()
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(original.detach().numpy().reshape(28, 28), cmap='gray')
    axarr[0].set_title("Original, class: " + str(p0.argmax()))
    axarr[1].imshow(adv.detach().numpy().reshape(28, 28), cmap='gray')
    axarr[1].set_title("Original, class: " + str(p1.argmax()))
    print("Class\t\tOrig\tAdv")
    for i in range(10):
        print("Class {}:\t{:.2f}\t{:.2f}".format(i, float(p0[:, i]), float(p1[:, i])))


def _fgsm(model, x, target, eps, targeted=True, clip_min=None, clip_max=None):
    # create a copy of the input, remove all previous associations to the compute graph...
    input_ = x.clone().detach_()
    # ... and make sure we are differentiating toward that variable
    input_.requires_grad_()

    # run the model and obtain the loss
    logits = model(input_)
    target = torch.LongTensor([target])
    model.zero_grad()
    loss = nn.CrossEntropyLoss()(logits, target)
    loss.backward()

    # perfrom either targeted or untargeted attack
    if targeted:
        out = input_ - eps * input_.grad.sign()
    else:
        out = input_ + eps * input_.grad.sign()

    # if desired clip the ouput back to the image domain
    if (clip_min is not None) or (clip_max is not None):
        out.clamp_(min=clip_min, max=clip_max)
    return out

# x: input image
# target: target class
# eps: size of l-infinity ball
def fgsm_targeted(model, x, target, eps):
    return _fgsm(model, x, target, eps, targeted=True)

# x: input image
# target: target class
# eps: size of l-infinity ball
def fgsm_untargeted(model, x, label, eps):
    return _fgsm_(model, x, label, eps, targeted=False,)


def pgd(model, x, target, k, eps, eps_step, clip_min=None, clip_max=None):
    x_min = x - eps
    x_max = x + eps

    # Randomize the starting point x.
    x = x + eps * (2 * torch.rand_like(x) - 1)
    if (clip_min is not None) or (clip_max is not None):
        x.clamp_(min=clip_min, max=clip_max)

    for i in range(k):
        # FGSM step
        # We don't clamp here (arguments clip_min=None, clip_max=None)
        # as we want to apply the attack as defined
        x = fgsm_(model, x, target, eps_step, targeted=False)
        # Projection Step
        x = torch.max(x_min, x)
        x = torch.min(x_max, x)
    # if desired clip the ouput back to the image domain
    if (clip_min is not None) or (clip_max is not None):
        x.clamp_(min=clip_min, max=clip_max)
    return x



# try out our attacks
original = torch.unsqueeze(test_dataset[0][0], dim=0)
adv = pgd(model, original, 7, 10, 0.08, 0.05)
show(original, adv, model_to_prob)
plt.show()
