import os
import torch

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms


def compute_mean_std(path):
    dataset = datasets.ImageFolder(os.path.join(path, 'train'), transform=transforms.Compose([transforms.Resize(256),
                                                                                              transforms.CenterCrop(
                                                                                                  224),
                                                                                              transforms.ToTensor()]))

    loader = DataLoader(dataset,
                        batch_size=10,
                        num_workers=0,
                        shuffle=False)

    mean = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(loader.dataset)

    var = 0.0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
    std = torch.sqrt(var / (len(loader.dataset) * 224 * 224))

    return mean, std


def add_prefix(prefix, path):
    return os.path.join(prefix, path)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    print(compute_mean_std('./data'))
