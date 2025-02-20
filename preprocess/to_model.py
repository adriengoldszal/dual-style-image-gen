from typing import Dict
from torch.utils.data import Dataset

class MultiTaskWrapper(Dataset):
    def __init__(self, name2dataset, meta_args, split):

        # Raw data and size.
        name2data = dict()
        for name, dataset in name2dataset.items():
            name2data[name] = [dataset[idx] for idx in range(len(dataset))]

        # Add split and name.
        for name, data in name2data.items():
            for item in data:
                item['split'] = split
                item['name'] = name

        # Concatenate.
        self.dataset = []
        for name in sorted(name2data.keys()):
            self.dataset.extend(name2data[name])

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)


def get_multi_task_dataset_splits(meta_args, name2dataset_splits):

    name2dev_dataset = dict()
    for name, dataset_splits in name2dataset_splits.items():
        name2dev_dataset[name] = dataset_splits['dev']

    return {
        'dev': MultiTaskDataset(meta_args, name2dev_dataset, split='dev'),
    }


class MultiTaskDataset(Dataset):

    def __init__(self, meta_args, name2dataset: Dict[str, Dataset], split: str):
        self.meta_args = meta_args

        self.data = MultiTaskWrapper(name2dataset=name2dataset, meta_args=meta_args, split=split)

    def __getitem__(self, index):
        data = self.data[index]
        model_inputs = {
            k: data[k] for k in data['model_kwargs']
        }
        return model_inputs

    def __len__(self):
        return len(self.data)
