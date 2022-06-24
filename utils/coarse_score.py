import cv2
import numpy as np
import torch

def coarse(input, target, writer, epoch, global_step):
    assert input.shape == target.shape, "the inputs should have same shape"
    input = 255 * input.float().squeeze(0).cpu().numpy().transpose((1, 2, 0))
    target = 255 * target.float().squeeze(0).cpu().numpy().transpose((1, 2, 0))

    edge_input = cv2.Laplacian(input, cv2.CV_32F)
    edge_target = cv2.Laplacian(target, cv2.CV_32F)

    edge_input[edge_input > 0] = 1
    edge_input[edge_input < 0] = 0
    edge_target[edge_target > 0] = 1
    edge_target[edge_target < 0] = 0

    edge = np.concatenate((edge_target, edge_input), axis=0)
    edge = torch.from_numpy(edge).type(torch.FloatTensor)
    edge = edge.unsqueeze(0).unsqueeze(1)

    edge_input = edge_input - np.mean(edge_input)
    edge_target = edge_target - np.mean(edge_target)
    std_input = np.std(edge_input)
    std_target = np.std(edge_target)
    cov = (np.mean(edge_input * edge_target) + 1e-6) / (std_input * std_target + 1e-6)
    cov = np.absolute(cov)

    if writer is not None:
        writer.add_images('edges/test_epoch{0}/{1}_{2}'.format(epoch, global_step, cov), edge, epoch)

    return cov



