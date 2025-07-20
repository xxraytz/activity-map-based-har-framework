import torch
import torch.nn.functional as F


def kld_loss(pred_logits, target_probs):
    """
    KL Divergence between distributions:
    pred = softmax(logits); compute KL(pred || target)
    """
    pred_log = F.log_softmax(pred_logits, dim=-1)
    # use reduction='batchmean' for average over batch
    return F.kl_div(pred_log, target_probs, reduction='batchmean')