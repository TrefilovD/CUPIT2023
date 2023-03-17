import os
import json

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class Dataset(Dataset):
    def __init__(self, 
        root: str, # path to directory
        transfrom,
        purpose,
        random_state=42,
        val_size=0.25
    ) -> None:
        assert purpose in ['train', 'val', 'test']

        self.root = root
        self.purpose = purpose
        self.random_state = random_state
        self.val_size = val_size

        self.data = self.load_data()
        self.transform = transfrom


    def __len__(self):
        return len(self.data)

    def load_data(self):
        if self.purpose == 'train':
            with open(os.path.join(self.root, 'ranking_train.jsonl'), 'r', encoding='utf-8') as f:
                train_data = [json.loads(line) for line in f]
            train_dataset, _ = train_test_split(train_data, test_size=self.val_size, random_state=self.random_state)
            return train_dataset
        elif self.purpose == 'val':
            with open(os.path.join(self.root, 'ranking_train.jsonl'), 'r', encoding='utf-8') as f:
                train_data = [json.loads(line) for line in f]
            _, val_dataset = train_test_split(train_data, test_size=self.val_size, random_state=self.random_state)
            return val_dataset
        else:
            with open(os.path.join(self.root, 'ranking_test.jsonl'), 'r', encoding='utf-8') as f:
                test_data = [json.loads(line) for line in f]

            return test_data
        
    def __getitem__(self, ind):
        sample = self.data[ind]
        post = sample['text']
        comments = None
        if self.purpose != 'test':
            comments = [(self.transform(comm['text']), comm['score']) for comm in sample['comments']]
        post = self.transform(post)

        return post, comments
