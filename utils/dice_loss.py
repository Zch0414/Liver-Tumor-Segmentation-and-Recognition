import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, n_classes=1):
        """

        Parameters
        ----------
        n_classes: When calculate the class number larger than 1, you should consider the background as a single class.
        """
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def one_hot_encoder(self, input):

        if self.n_classes == 1:
            return input.unsqueeze(1).float()

        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = (input == i)
            tensor_list.append(temp_prob.unsqueeze(1))
        output = torch.cat(tensor_list, dim=1)
        return output.float()

    @classmethod
    def dice_coeff(cls, input, target):
        input = input.float()
        smooth = 1e-5
        inter = torch.sum(input * target)
        union = torch.sum(input * input) + torch.sum(target * target)
        dice = (2 * inter + smooth) / (union + smooth)
        return dice

    def forward(self, input, target, weight=None, softmax=True):
        """

        Parameters
        ----------
        input
        target
        weight: The weight should be a list which contain the coefficient for each class
        softmax

        Returns
        -------

        """
        if self.n_classes == 1:
            input = input.unsqueeze(1)
            softmax = False
        if softmax:
            _softmax = nn.Softmax(dim=1)
            input = _softmax(input)

        target = self.one_hot_encoder(target)

        if weight is None:
            weight = [1] * self.n_classes

        assert input.size() == target.size(), \
            "predict {0} and target {1} shape do not match".format(input.size(), target.size())

        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self.dice_coeff(input[:, i], target[:, i])
            class_wise_dice.append(dice.item())
            loss += (1 - dice) * weight[i]
        return loss / self.n_classes
