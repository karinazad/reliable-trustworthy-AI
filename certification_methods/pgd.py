import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")
batch_size = 64

def fgsm_(model, x, target, eps, targeted=True, clip_min=None, clip_max=None, loss_fn = None):
    """Internal process for all FGSM and PGD attacks."""
    input_ = x.clone().detach_()
    input_.requires_grad_()

    logits = model(input_)
    target = target.type(torch.long)
    model.zero_grad()

    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, target)
    else:
        loss_fn()
        logits_x = model()

    loss.backward()

    if targeted:
        out = input_ - eps * input_.grad.sign()
    else:
        out = input_ + eps * input_.grad.sign()

    if (clip_min is not None) or (clip_max is not None):
        out.clamp_(min=clip_min, max=clip_max)
    return out

def pgd_(model, x, target, k, eps, eps_step, targeted=True, clip_min=None, clip_max=None, loss_fn=None):
    x_min = x - eps
    x_max = x + eps

    # Randomize the starting point x.
    x = x.detach()

    with torch.no_grad():
        x = x + eps * (2 * torch.rand_like(x) - 1)
        if (clip_min is not None) or (clip_max is not None):
            x.clamp_(min=clip_min, max=clip_max)

    for i in range(k):
        x.detach_().requires_grad_()

        with torch.enable_grad():
            x = fgsm_(model, x, target, eps_step, targeted, loss_fn=loss_fn)

            x = torch.max(x_min, x)
            x = torch.min(x_max, x)

    # if desired clip the ouput back to the image domain
    if (clip_min is not None) or (clip_max is not None):
        x.clamp_(min=clip_min, max=clip_max)
    return x.detach()


def pgd(model, x, label, k, eps, eps_step, **kwargs):
    return pgd_(model, x, label, k, eps, eps_step, targeted=False, **kwargs)


def pgd_batch(model, x_batch, target, k, eps, eps_step, kl_loss=False):
    if kl_loss:
        # loss function for the case that target is a distribution rather than a label (used for TRADES)
        loss_fn = torch.nn.KLDivLoss(reduction='sum')
    else:
        # standard PGD
        loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    with torch.no_grad():  # disable gradients here
        # initialize with a random point inside the considered perturbation region
        x_adv = x_batch.detach() + eps * (2 * torch.rand_like(x_batch) - 1)
        x_adv.clamp_(min=0.0, max=1.0)  # project back to the image domain

        for step in range(k):
            # make sure we don't have a previous compute graph and enable gradient computation
            x_adv.detach_().requires_grad_()

            with torch.enable_grad():  # re-enable gradients
                # run the model and obtain the loss
                out = F.log_softmax(model(x_adv), dim=1) if kl_loss else model(x_adv)
                model.zero_grad()
                # compute gradient
                loss_fn(out, target).backward()

            # compute step
            step = eps_step * x_adv.grad.sign()
            # project to eps ball
            x_adv = x_batch + (x_adv + step - x_batch).clamp(min=-eps, max=eps)
            # clamp back to image domain; in contrast to the previous exercise we clamp at each step (so this is part of the projection)
            # both implementations are valid; this dents to work slightly better
            x_adv.clamp_(min=0.0, max=1.0)
    return x_adv.detach()
