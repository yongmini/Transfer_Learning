from torchvision import datasets, transforms
import torch
from collections import defaultdict
class LimitedImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, max_images_per_class=50):
        super(LimitedImageFolder, self).__init__(root, transform=transform, target_transform=target_transform)
        
        # 각 클래스별로 최대 이미지 수 제한
        self.samples = self._limit_samples(self.samples, max_images_per_class)
        self.targets = [s[1] for s in self.samples]
    
    def _limit_samples(self, samples, max_images_per_class):
        class_count = defaultdict(int)
        limited_samples = []
        
        for sample in samples:
            path, class_idx = sample
            if class_count[class_idx] < max_images_per_class:
                limited_samples.append(sample)
                class_count[class_idx] += 1
        
        return limited_samples
           #  transforms.CenterCrop(224),
def load_data(data_folder, batch_size, train, num_workers=0, **kwargs):
    transform = {
        'train': transforms.Compose(
            [transforms.CenterCrop(512),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])]),
        'test': transforms.Compose(
            [transforms.CenterCrop(224),
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
    }

  #  data = LimitedImageFolder(root=data_folder, transform=transform['train' if train else 'test'], max_images_per_class=10)
    data = datasets.ImageFolder(root=data_folder, transform=transform['train' if train else 'test'])
    data_loader = get_data_loader(data, batch_size=batch_size, 
                                shuffle=True if train else False, 
                                num_workers=num_workers, **kwargs, drop_last=True if train else False)
    n_class = len(data.classes)
    print(f"Number of classes: {n_class}")
    return data_loader, n_class


def get_data_loader(dataset, batch_size, shuffle=True, drop_last=False, num_workers=0, infinite_data_loader=False, **kwargs):
    if not infinite_data_loader:
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers, **kwargs)
    else:
        return InfiniteDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers, **kwargs)

class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch

class InfiniteDataLoader:
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, num_workers=0, weights=None, **kwargs):
        if weights is not None:
            sampler = torch.utils.data.WeightedRandomSampler(weights,
                replacement=False,
                num_samples=batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(dataset,
                replacement=False)
            
        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=drop_last)

        self._infinite_iterator = iter(torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler)
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        return 0 # Always return 0
