import torch
import random

import torch.nn as nn

from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CoralLayer(torch.nn.Module):
    """ Implements CORAL layer described in
    Cao, Mirjalili, and Raschka (2020)
    *Rank Consistent Ordinal Regression for Neural Networks
       with Application to Age Estimation*
    Pattern Recognition Letters, https://doi.org/10.1016/j.patrec.2020.11.008
    Parameters
    -----------
    size_in : int
        Number of input features for the inputs to the forward method, which
        are expected to have shape=(num_examples, num_features).
    num_classes : int
        Number of classes in the dataset.
    preinit_bias : bool (default=True)
        If true, it will pre-initialize the biases to descending values in
        [0, 1] range instead of initializing it to all zeros. This pre-
        initialization scheme results in faster learning and better
        generalization performance in practice.
    """
    def __init__(self, size_in, num_classes, preinit_bias=True):
        super().__init__()
        self.size_in, self.size_out = size_in, 1

        self.coral_weights = torch.nn.Linear(self.size_in, 1, bias=False)
        if preinit_bias:
            self.coral_bias = torch.nn.Parameter(
                torch.arange(num_classes - 1, 0, -1).float() / (num_classes-1))
        else:
            self.coral_bias = torch.nn.Parameter(
                torch.zeros(num_classes-1).float())

    def forward(self, x):
        """
        Computes forward pass.
        Parameters
        -----------
        x : torch.tensor, shape=(num_examples, num_features)
            Input features.
        Returns
        -----------
        logits : torch.tensor, shape=(num_examples, num_classes-1)
        """
        return self.coral_weights(x) + self.coral_bias

class BaseModel(torch.nn.Module):
    def __init__(self, head) -> None:
        super().__init__()
        self.head = head

    def forward(self, x):
        x = self.head(x)
        return x


class WrapBert(object):
    def __init__(self, model, transform) -> None:
        self.model = model.eval()
        self.transform = transform

    @torch.no_grad()
    def __call__(self, text):
        _, indexed_tokens, segments_ids = self.transform(text)
        while len(indexed_tokens) >= 512:
            ind = random.randint(20, 500)
            indexed_tokens.pop(ind)
            segments_ids.pop(ind)

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        assert len(indexed_tokens) < 512, f"{len(indexed_tokens)}"
        assert tokens_tensor.size() == segments_tensors.size(), f"tokens {tokens_tensor.size()}, segments {segments_tensors.size()}"
        outputs = self.model(tokens_tensor, segments_tensors)
        hidden_states = outputs.hidden_states # outputs[2]
        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1,0,2)

        token_vecs_cat = []

        for token in token_embeddings:
            cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)            
            token_vecs_cat.append(cat_vec)

        token_vecs = hidden_states[-2][0]

        sentence_embedding = torch.mean(token_vecs, dim=0)

        return sentence_embedding
    

NEG_INF = -10000
TINY_FLOAT = 1e-6

def mask_softmax(matrix, mask=None):
    """Perform softmax on length dimension with masking.

    Parameters
    ----------
    matrix: torch.float, shape [batch_size, .., max_len]
    mask: torch.long, shape [batch_size, max_len]
        Mask tensor for sequence.

    Returns
    -------
    output: torch.float, shape [batch_size, .., max_len]
        Normalized output in length dimension.
    """

    if mask is None:
        result = F.softmax(matrix, dim=-1)
    else:
        mask_norm = ((1 - mask) * NEG_INF).to(matrix)
        for i in range(matrix.dim() - mask_norm.dim()):
            mask_norm = mask_norm.unsqueeze(1)
        result = F.softmax(matrix + mask_norm, dim=-1)

    return result


def mask_mean(seq, mask=None):
    """Compute mask average on length dimension.

    Parameters
    ----------
    seq : torch.float, size [batch, max_seq_len, n_channels],
        Sequence vector.
    mask : torch.long, size [batch, max_seq_len],
        Mask vector, with 0 for mask.

    Returns
    -------
    mask_mean : torch.float, size [batch, n_channels]
        Mask mean of sequence.
    """

    if mask is None:
        return torch.mean(seq, dim=1)

    mask_sum = torch.sum(  # [b,msl,nc]->[b,nc]
        seq * mask.unsqueeze(-1).float(), dim=1)
    seq_len = torch.sum(mask, dim=-1)  # [b]
    mask_mean = mask_sum / (seq_len.unsqueeze(-1).float() + TINY_FLOAT)

    return mask_mean


def mask_max(seq, mask=None):
    """Compute mask max on length dimension.

    Parameters
    ----------
    seq : torch.float, size [batch, max_seq_len, n_channels],
        Sequence vector.
    mask : torch.long, size [batch, max_seq_len],
        Mask vector, with 0 for mask.

    Returns
    -------
    mask_max : torch.float, size [batch, n_channels]
        Mask max of sequence.
    """

    if mask is None:
        return torch.mean(seq, dim=1)

    torch
    mask_max, _ = torch.max(  # [b,msl,nc]->[b,nc]
        seq + (1 - mask.unsqueeze(-1).float()) * NEG_INF,
        dim=1)

    return mask_max


def seq_mask(seq_len, max_len):
    """Create sequence mask.

    Parameters
    ----------
    seq_len: torch.long, shape [batch_size],
        Lengths of sequences in a batch.
    max_len: int
        The maximum sequence length in a batch.

    Returns
    -------
    mask: torch.long, shape [batch_size, max_len]
        Mask tensor for sequence.
    """

    idx = torch.arange(max_len).to(seq_len).repeat(seq_len.size(0), 1)
    mask = torch.gt(seq_len.unsqueeze(1), idx).to(seq_len)

    return mask


class DynamicLSTM(nn.Module):
    """
    Dynamic LSTM module, which can handle variable length input sequence.

    Parameters
    ----------
    input_size : input size
    hidden_size : hidden size
    num_layers : number of hidden layers. Default: 1
    dropout : dropout rate. Default: 0.5
    bidirectional : If True, becomes a bidirectional RNN. Default: False.

    Inputs
    ------
    input: tensor, shaped [batch, max_step, input_size]
    seq_lens: tensor, shaped [batch], sequence lengths of batch

    Outputs
    -------
    output: tensor, shaped [batch, max_step, num_directions * hidden_size],
         tensor containing the output features (h_t) from the last layer
         of the LSTM, for each t.
    """

    def __init__(self, input_size, hidden_size=100,
                 num_layers=1, dropout=0., bidirectional=False):
        super(DynamicLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, bias=True,
            batch_first=True, dropout=dropout, bidirectional=bidirectional)

    def forward(self, x, seq_lens):
        # sort input by descending length
        _, idx_sort = torch.sort(seq_lens, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        x_sort = torch.index_select(x, dim=0, index=idx_sort)
        seq_lens_sort = torch.index_select(seq_lens, dim=0, index=idx_sort)

        # pack input
        x_packed = pack_padded_sequence(
            x_sort, seq_lens_sort, batch_first=True)

        # pass through rnn
        y_packed, _ = self.lstm(x_packed)

        # unpack output
        y_sort, length = pad_packed_sequence(y_packed, batch_first=True)

        # unsort output to original order
        y = torch.index_select(y_sort, dim=0, index=idx_unsort)

        return y


class RankingModel(nn.Module):

    def __init__(self, pretrained_embed, loss=None):
        super(RankingModel, self).__init__()
        pretrained_embed = pretrained_embed
        padding_idx = 0
        embed_dim = 768
        num_classes = 5
        num_layers = 2
        hidden_dim = 50
        dropout = 0.2

        self.embed = nn.Embedding.from_pretrained(
                pretrained_embed.embeddings.word_embeddings.weight, freeze=False)
        # self.embed.padding_idx = padding_idx

        self.rnn = DynamicLSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            dropout=dropout, bidirectional=True)

        self.fc_att = nn.Linear(hidden_dim * 2, 1)

        self.fc1 = nn.Linear(hidden_dim * 6 * 4, hidden_dim * 6)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim * 6, hidden_dim)
        self.act2 = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.out = CoralLayer(hidden_dim, num_classes) # nn.Linear(hidden_dim, num_classes)

        self.softmax = nn.Softmax()

        if loss is None:
            self.loss = nn.BCEWithLogitsLoss()
        else:
            self.loss = loss

    def forward(self, 
                inp
        ):
        if True:
            return self.forwardv2(inp)

        post_word_seq, com_word_seq, post_seq_len, com_seq_len = inp
        # mask
        post_max_seq_len = torch.max(post_seq_len)
        post_mask = seq_mask(post_seq_len, post_max_seq_len)  # [b,msl]
        com_max_seq_len = torch.max(com_seq_len)
        com_mask = seq_mask(com_seq_len, com_max_seq_len)  # [b,msl]

        # embed
        post_e = self.drop(self.embed(post_word_seq))  # [b,msl]->[b,msl,e]
        com_e = self.drop(self.embed(com_word_seq))  # [b,msl]->[b,msl,e]

        # bi-rnn
        post_r = self.rnn(post_e, post_seq_len)  # [b,msl,e]->[b,msl,h*2]
        com_r = self.rnn(com_e, com_seq_len)  # [b,msl,e]->[b,msl,h*2]

        # attention
        post_att = self.fc_att(post_r).squeeze(-1)  # [b,msl,h*2]->[b,msl]
        post_att = mask_softmax(post_att, post_mask)  # [b,msl]
        post_r_att = torch.sum(post_att.unsqueeze(-1) * post_r, dim=1)  # [b,h*2]

        com_att = self.fc_att(com_r).squeeze(-1)  # [b,msl,h*2]->[b,msl]
        com_att = mask_softmax(com_att, com_mask)  # [b,msl]
        com_r_att = torch.sum(com_att.unsqueeze(-1) * com_r, dim=1)  # [b,h*2]

        # pooling
        post_r_avg = mask_mean(post_r, post_mask)  # [b,h*2]
        post_r_max = mask_max(post_r, post_mask)  # [b,h*2]
        post_r = torch.cat([post_r_avg, post_r_max, post_r_att], dim=-1)  # [b,h*6]

        com_r_avg = mask_mean(com_r, com_mask)  # [b,h*2]
        com_r_max = mask_max(com_r, com_mask)  # [b,h*2]
        com_r = torch.cat([com_r_avg, com_r_max, com_r_att], dim=-1)  # [b,h*6]

        # connection post and comments
        r = torch.cat((post_r, post_r * com_r, torch.abs(post_r - com_r), com_r), dim=1)

        # feed-forward
        f = self.drop(self.act1(self.fc1(r)))  # [b,h*6*4]->[b,h*6]
        f = self.drop(self.act2(self.fc2(f)))  # [b,h*6]->[b,h]
        logits = self.out(f) #.squeeze(-1)  # [b,h]->[b]

        # out = self.softmax(logits)

        return logits
    
    def forwardv2(self, inp):
        post, word_seq, seq_len = inp

        max_seq_len = torch.max(seq_len)
        mask = seq_mask(seq_len, max_seq_len)  # [b,msl]

        # e = self.drop(self.embed(word_seq))  # [b,msl]->[b,msl,e]
        e = self.embed(word_seq)  # [b,msl]->[b,msl,e]
        p = self.embed(post)

        e = e + p

        r = self.rnn(e, seq_len)  # [b,msl,e]->[b,msl,h*2]

        att = self.fc_att(r).squeeze(-1)  # [b,msl,h*2]->[b,msl]
        att = mask_softmax(att, mask)  # [b,msl]
        r_att = torch.sum(att.unsqueeze(-1) * r, dim=1)  # [b,h*2]

        r_avg = mask_mean(r, mask)  # [b,h*2]
        r_max = mask_max(r, mask)  # [b,h*2]
        r = torch.cat([r_avg, r_max, r_att], dim=-1)  # [b,h*6]

        f = self.drop(self.act2(self.fc2(r)))  # [b,h*6]->[b,h]
        logits = self.out(f).squeeze(-1)  # [b,h]->[b]

        return logits 