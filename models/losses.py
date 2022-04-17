import torch
import torch.nn.functional as F
import segmentation_models_pytorch.losses as smp_losses
from torch import nn

import numpy as np
from torch.autograd import Variable
from typing import Optional
def cross_entropy(input, target, weight=None, reduction='mean',ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    #print("pred: ", input.shape, "target: ", target.shape)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

    weight_ = torch.Tensor([1, 1]).cuda()
    return F.cross_entropy(input=input, target=target, weight=weight_,
                           ignore_index=ignore_index, reduction=reduction)


# def focal_loss(pred, true):
#     B, C, H, W = pred.shape
#     true = true.squeeze()
#     msk0 = torch.zeros([B,H,W]).cuda()
#     msk1 = torch.zeros([B,H,W]).cuda()
#     msk0[true == 0] = 1
#     msk1[true == 1] = 1

#     loss0 = focal_loss2D(pred[:,0,:,:], msk0)
#     loss1 = focal_loss2D(pred[:,1,:,:], msk1)

#     return loss

def one_hot(index, classes):
    size = index.size() + (classes,)
    view = index.size() + (1,)

    mask = torch.Tensor(*size).fill_(0)
    index = index.view(*view)
    ones = 1.

    if isinstance(index, Variable):
        ones = Variable(torch.Tensor(index.size()).fill_(1)).cuda()
        mask = Variable(mask, volatile=index.volatile).cuda()

    return mask.scatter_(1, index, ones)



def one_hot(
    labels: torch.Tensor,
    num_classes: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    r"""Convert an integer label x-D tensor to a one-hot (x+1)-D tensor.

    Args:
        labels: tensor with labels of shape :math:`(N, *)`, where N is batch size.
          Each value is an integer representing correct classification.
        num_classes: number of classes in labels.
        device: the desired device of returned tensor.
        dtype: the desired data type of returned tensor.

    Returns:
        the labels in one hot tensor of shape :math:`(N, C, *)`,

    Examples:
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> one_hot(labels, num_classes=3)
        tensor([[[[1.0000e+00, 1.0000e-06],
                  [1.0000e-06, 1.0000e+00]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e+00],
                  [1.0000e-06, 1.0000e-06]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e-06],
                  [1.0000e+00, 1.0000e-06]]]])

    """
    if not isinstance(labels, torch.Tensor):
        raise TypeError(f"Input labels type is not a torch.Tensor. Got {type(labels)}")

    if not labels.dtype == torch.int64:
        labels = labels.to(torch.int64)
        if not labels.dtype == torch.int64:
            raise ValueError(f"labels must be of the same dtype torch.int64. Got: {labels.dtype}")

    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one." " Got: {}".format(num_classes))

    shape = labels.shape
    one_hot = torch.zeros((shape[0], num_classes) + shape[1:], device=device, dtype=dtype)

    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps
    
def focal_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha=0.5,
    gamma: float = 2.0,
    reduction: str = 'mean',
    eps: Optional[float] = None,
) -> torch.Tensor:
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        input: logits tensor with shape :math:`(N, C, *)` where C = number of classes.
        target: labels tensor with shape :math:`(N, *)` where each value is :math:`0 ≤ targets[i] ≤ C−1`.
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Deprecated: scalar to enforce numerical stabiliy. This is no longer used.

    Return:
        the computed loss.

    Example:
        >>> N = 5  # num_classes
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = focal_loss(input, target, alpha=0.5, gamma=2.0, reduction='mean')
        >>> output.backward()
    """

    target = target.squeeze()

    if eps is not None and not torch.jit.is_scripting():
        warnings.warn(
            "`focal_loss` has been reworked for improved numerical stability "
            "and the `eps` argument is no longer necessary",
            DeprecationWarning,
            stacklevel=2,
        )

    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) >= 2:
        raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input.shape}")

    if input.size(0) != target.size(0):
        raise ValueError(f'Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).')

    n = input.size(0)
    out_size = (n,) + input.size()[2:]
    if target.size()[1:] != input.size()[2:]:
        raise ValueError(f'Expected target size {out_size}, got {target.size()}')

    if not input.device == target.device:
        raise ValueError(f"input and target must be in the same device. Got: {input.device} and {target.device}")

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1)
    log_input_soft: torch.Tensor = F.log_softmax(input, dim=1)

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)

    # compute the actual focal loss
    weight = torch.pow(-input_soft + 1.0, gamma)

    focal = -alpha * weight * log_input_soft
    loss_tmp = torch.einsum('bc...,bc...->b...', (target_one_hot, focal))

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss

# class FocalLoss(nn.Module):

#     def __init__(self, gamma=0, eps=1e-7):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.eps = eps

#     def forward(self, input, target):
#         target = target.squeeze(1)
#         y = one_hot(target, input.size(1)).transpose(3,1)
#         logit = F.softmax(input, dim=-1)
#         logit = logit.clamp(self.eps, 1. - self.eps)

#         print(y.shape, logit.shape)
#         loss = -1 * y * torch.log(logit) # cross entropy
#         loss = loss * (1 - logit) ** self.gamma # focal loss

#         return loss.sum()


# # https://github.com/clcarwin/focal_loss_pytorch.git
# class FocalLoss(nn.Module):
#     def __init__(self, gamma=0, alpha=None, size_average=True):
#         super(FocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.size_average = size_average

#     def forward(self, input, target):
#         if input.dim()>2:
#             input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
#             input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
#             input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
#         target = target.view(-1,1)

#         logpt = F.log_softmax(input)
#         logpt = logpt.gather(1,target)
#         logpt = logpt.view(-1)
#         pt = Variable(logpt.data.exp())

#         loss = -1 * (1-pt)**self.gamma * logpt
#         if self.size_average: return loss.mean()
#         else: return loss.sum()


def focal_loss_xBD(pred, true):
    B, C, H, W = pred.shape
    true = true.squeeze()
    msk0 = torch.zeros([B,H,W]).cuda()
    msk1 = torch.zeros([B,H,W]).cuda()
    msk2 = torch.zeros([B,H,W]).cuda()
    msk3 = torch.zeros([B,H,W]).cuda()
    msk4 = torch.zeros([B,H,W]).cuda()

    msk0[true == 0] = 1
    msk1[true == 1] = 1
    msk2[true == 2] = 1
    msk3[true == 3] = 1
    msk4[true == 4] = 1

    loss0 = focal_loss2D(pred[:,0,:,:], msk0)
    loss1 = focal_loss2D(pred[:,1,:,:], msk1)
    loss2 = focal_loss2D(pred[:,2,:,:], msk2)
    loss3 = focal_loss2D(pred[:,3,:,:], msk3)
    loss4 = focal_loss2D(pred[:,4,:,:], msk4)

    return loss0 * 0.01 + loss1 * 0.1 + loss2 * 2 + loss3 * 1.5 + loss4 * 2
    
    # too much oscillating
    # return loss0 * 0.1 + loss1 * 1 + loss2 * 5 + loss3 * 5 + loss4 * 10
    
    # return loss0 * 1 + loss1 * 100


def focal_loss2D(outputs, targets, gamma=2, ignore_index=255):
    outputs = torch.sigmoid(outputs).contiguous()
    targets = targets.contiguous()

    eps = 1e-8
    outputs = torch.clamp(outputs, 1e-8, 1. - 1e-8)
    targets = torch.clamp(targets, 1e-8, 1. - 1e-8)
    pt = (1 - targets) * (1 - outputs) + targets * outputs
    return (-(1. - pt) ** gamma * torch.log(pt)).mean()


def multi_cross_entropy(input, target, weight=None, reduction='mean',ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)

    weight_ = torch.Tensor([1, 5, 100, 90, 100]).cuda()
    # print("pred: ", input.shape, "target: ", target.shape)
 
    return F.cross_entropy(input=input, target=target, weight=weight_,
                           ignore_index=ignore_index, reduction=reduction)



def ce_dice(input, target, weight=None, ignore_index=255, reduction='mean'):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    #target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)
    
    weight_ = torch.Tensor([0.2, 0.8]).cuda()
    ce  = torch.nn.CrossEntropyLoss(weight=weight,
                           ignore_index=ignore_index, reduction=reduction)
    celoss = ce(input,target)    
    dice = smp_losses.DiceLoss(mode='binary')
    input = torch.argmax(input, dim=1).type(torch.float32)
    target = target.type(torch.float32)
    diceloss = dice(input, target)

    loss = 0.5*diceloss + 0.5*celoss

    return loss


def diceloss(input, target, weight=None):
    input = torch.argmax(input, dim=1).type(torch.float32)
    target = target.type(torch.float32)

    diceloss = smp_losses.DiceLoss(mode='binary')
    loss = diceloss(input, target)
    return loss