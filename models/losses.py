import torch
import torch.nn.functional as F
import segmentation_models_pytorch.losses as smp_losses

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

    weight_ = torch.Tensor([1, 100]).cuda()
    return F.cross_entropy(input=input, target=target, weight=weight_,
                           ignore_index=ignore_index, reduction=reduction)


def focal_loss(pred, true):
    B, C, H, W = pred.shape
    true = true.squeeze()

    pred = pred.argmax(axis=1)
    loss = focal_loss2D(pred, true)

    return loss


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

    loss = 0.7*diceloss + 0.3*celoss

    return loss
