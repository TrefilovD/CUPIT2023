import torch
import time

from tqdm import tqdm


def train(model, dataloader, optimizer, criterion, device='cuda'):
    epoch_loss = torch.tensor(0.).to(device)
    ind = 0
    pbar = tqdm(dataloader)
    # for (post_word_seq, com_word_seq, post_seq_len,com_seq_len, target) in tqdm(dataloader):
    for (post, comments, seq_len, target) in pbar:
        # out = model(post_word_seq, com_word_seq, post_seq_len, com_seq_len)
        inp = [post, comments, seq_len]
        out = model(inp).reshape(dataloader.batch_size, 5, 5)
        optimizer.zero_grad()
        loss = criterion(out, target)
        loss.backward()
        epoch_loss += loss.detach()

        ind += 1

        if ind % 100 == 0:
            pbar.set_description(f"batch {ind}/{len(dataloader)} loss {loss}")

    return epoch_loss.cpu().item()