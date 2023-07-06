import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.5):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target):
        pt = torch.sigmoid(input)
        loss = -self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        return loss.mean()


# criterion = FocalLoss()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        # 计算预测掩膜和真实掩膜的交集和并集
        intersection = torch.sum(prediction * target)
        union = torch.sum(prediction) + torch.sum(target)

        # 计算 Dice 系数
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # 计算 Dice Loss
        loss = 1 - dice

        return loss