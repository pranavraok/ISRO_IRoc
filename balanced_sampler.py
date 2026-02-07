import random
from torch.utils.data import Sampler
from collections import defaultdict

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, n_classes, n_samples):
        self.labels = labels
        self.labels_set = list(set(labels))
        self.label_to_indices = defaultdict(list)

        for idx, label in enumerate(labels):
            self.label_to_indices[label].append(idx)

        for label in self.labels_set:
            random.shuffle(self.label_to_indices[label])

        self.n_classes = n_classes
        self.n_samples = n_samples
        self.batch_size = self.n_classes * self.n_samples

    def __iter__(self):
        count = 0
        while count + self.batch_size <= len(self.labels):
            classes = random.sample(self.labels_set, self.n_classes)
            indices = []

            for cls in classes:
                cls_indices = self.label_to_indices[cls]
                indices.extend(cls_indices[:self.n_samples])

                self.label_to_indices[cls] = cls_indices[self.n_samples:] + cls_indices[:self.n_samples]

            yield indices
            count += self.batch_size

    def __len__(self):
        return len(self.labels) // self.batch_size

