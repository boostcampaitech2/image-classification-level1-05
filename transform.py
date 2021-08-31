import torch

from torchvision import transforms as T


class AddGaussianNoise():
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + f'mean: {self.mean:.2f}, std: {self.std:.2f}'


class BaseTransform:
    def __init__(self, resize, mean, std):
        self.transforms = [
            # T.CenterCrop((320, 256)),
            T.Resize(resize, T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]

    def __call__(self, image):
        return T.Compose(self.transforms)(image)


class CustomTransform(BaseTransform):
    def __init__(self, resize, mean, std):
        super().__init__(resize=resize, mean=mean, std=std)
        self.transforms = [
            *self.transforms,
            T.RandomHorizontalFlip(p=0.5)
        ]
