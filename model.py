import torch
import random

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
        hidden_states = outputs[2]
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
    
