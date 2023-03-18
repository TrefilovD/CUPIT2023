import torch

from torch.utils.data import DataLoader
from torch.utils.data import Sampler, RandomSampler, SequentialSampler

from torch.nn.utils.rnn import pad_sequence

import random


def get_dataloader(dataset):
    is_shuffle = dataset.purpose == 'train'
    drop_last = dataset.purpose == 'train'
    dataloader = DataLoader(dataset, 
                            batch_size=8, 
                            shuffle=is_shuffle, 
                            drop_last=drop_last, 
                            # collate_fn=collate_fn_v1
                            collate_fn=collate_fn
                            )

    return dataloader

def collate_fn(batch):
    ind = list(range(5))
    random.shuffle(ind)
    post = []
    comment = []
    score = []
    for p, c, s in batch:
        for i in ind:
            # a = c[i].unsqueeze(0)
            # b = p.unsqueeze(0)
            pp = pad_sequence([c[i], p]).permute(1, 0)
            a, b = pp[0], pp[1]
            # tt = torch.cat([a, a * b, torch.abs(a - b), b], dim=0)
            comment.append(a)
            # comment.append(torch.tensor(c[i], dtype=torch.int32))
        post.append(p)
        new_s = [s[i] for i in ind]
        # comment.append(new_c)
        score.append(torch.stack(new_s))

    seq_len = torch.tensor([len(c) for c in comment], dtype=torch.int32)
    comment = pad_sequence(comment).permute(1, 0).reshape(len(batch) * 5, -1)
    post = pad_sequence(comment).permute(1, 0)
    score = torch.stack(score)

    # post, comment = pad_sequence(post), pad_sequence(comment) # padded inputs
    # comment = pad_sequence(comment)
    # attention_mask1 = np.where(sentence1 != 0, 1, 0)
    # attention_mask2 = np.where(sentence2 != 0, 1, 0)
    return post, comment, seq_len, score

def collate_fn_v1(batch):
    post = []
    comment = []
    score = []
    for p, c, s in batch:
        ind = list(range(5))
        random.shuffle(ind)
        ind = ind[0]
        comment.append(c[ind])
        post.append(p)
        score.append(s[ind])

    post = pad_sequence(post).permute(1, 0)
    comment = pad_sequence(comment).permute(1, 0)
    score = torch.stack(score)
    # post = torch.stack(post)
    # comment = torch.stack(comment)

    post_lens = torch.tensor([len(i) for i in post], dtype=torch.int32)
    com_lens = torch.tensor([len(i) for i in comment], dtype=torch.int32)

    # post, comment = pad_sequence(post), pad_sequence(comment) # padded inputs
    # comment = pad_sequence(comment)
    # attention_mask1 = np.where(sentence1 != 0, 1, 0)
    # attention_mask2 = np.where(sentence2 != 0, 1, 0)
    return post, comment, post_lens, com_lens, score


# class TextSampler(Sampler):
    