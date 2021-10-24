import argparse
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from pgd import *


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 200)
        self.fc2 = nn.Linear(200, 10)

    def forward(self, x):
        x = x.view((-1, 28 * 28))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Normalize(nn.Module):
    def forward(self, x):
        return (x - 0.1307) / 0.3081


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--seed', type=int, default='42', help='seed')
parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
parser.add_argument('--defense', type=str, choices=['none', 'PGD', 'TRADES'], default='TRADES', help='defense')
parser.add_argument('--num_epochs', type=int, default=10, help='epochs')
parser.add_argument('--eps', type=float, default=0.1, help='pgd epsilon')
parser.add_argument('-k', type=int, default=7, help='pgd steps')
parser.add_argument('--trades_fact', type=float, default=1.0, help='TRADES lambda')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Setting the random number generator
torch.manual_seed(args.seed)

# Datasets
train_dataset = datasets.MNIST('mnist_data/', train=True, download=True,
                               transform=transforms.Compose([transforms.ToTensor()]))
test_dataset = datasets.MNIST('mnist_data/', train=False, download=True,
                              transform=transforms.Compose([transforms.ToTensor()]))

# Data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Add the data normalization as a first "layer" to the network
# this allows us to search for adverserial examples to the real image, rather than
# to the normalized image
model = nn.Sequential(Normalize(), Net())
model = model.to(device)

opt = optim.Adam(params=model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
scheduler = optim.lr_scheduler.StepLR(opt, 15)
ce_loss = torch.nn.CrossEntropyLoss()
kl_loss = torch.nn.KLDivLoss(reduction='batchmean')


def cross_entropy(a, b):
    logsoftmax = nn.LogSoftmax(dim=1)
    softmax = nn.Softmax(dim=1)
    return torch.mean(torch.sum(- softmax(b) * logsoftmax(a), 1))


EPS = 0.1
K = 7
EPS_STEP = 2.5 * (EPS / K)
LAMBDA = 1

print("Defense:", args.defense)

for epoch in range(1, args.num_epochs + 1):
    # Training
    for batch_idx, (x_batch, y_batch) in enumerate(tqdm(train_loader)):

        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        if args.defense == 'PGD':
            model.eval()
            x_max = pgd(model=model, x=x_batch, label=y_batch, k=K, eps=EPS, eps_step=EPS_STEP)

            model.train()
            logits_xmax = model(x_max)
            loss = ce_loss(logits_xmax, y_batch)


        elif args.defense == 'TRADES':

            model.train()
            logits_x = model(x_batch)

            model.eval()
            x_max = pgd(model=model,
                        x=x_batch,
                        label=y_batch,
                        k=K, eps=EPS,
                        eps_step=EPS_STEP)


            logits_xmax = model(x_max)

            loss = ce_loss(logits_x, y_batch) + LAMBDA * cross_entropy(logits_x, logits_xmax)

        elif args.defense == 'none':
            model.train()
            out_nat = model(x_batch)
            loss = ce_loss(out_nat, y_batch)

        opt.zero_grad()
        loss.backward()
        opt.step()

    # Testing
    model.eval()
    tot_test, tot_acc, tot_adv_acc = 0.0, 0.0, 0.0
    for batch_idx, (x_batch, y_batch) in enumerate(tqdm(test_loader)):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        out = model(x_batch)
        pred = torch.max(out, dim=1)[1]
        acc = pred.eq(y_batch).sum().item()

        # Problem 1.1 calculate accuracy under PGD attack
        x_max = pgd_batch(model=model, x_batch=x_batch, target=y_batch, k=K, eps=EPS, eps_step=EPS_STEP)
        out_adv = model(x_max)
        pred_adv = torch.max(out_adv, dim=1)[1]
        acc_adv = pred_adv.eq(y_batch).sum().item()

        tot_acc += acc
        tot_adv_acc += acc_adv
        tot_test += x_batch.size()[0]
    scheduler.step()
    print('Epoch %d: Accuracy %.5lf, Adv Accuracy %.5lf' % (epoch, tot_acc / tot_test, tot_adv_acc / tot_test))


# optionally save the model
# os.makedirs("models", exist_ok=True)
# torch.save(model.state_dict(), f"models/Net_{args.num_epochs}_{args.defense}")
