import nltk
import torch

from transformers import BertTokenizer, BertModel, DistilBertModel
from dataset import Dataset
from model import WrapBert, BaseModel, RankingModel
from proccesing import preprocces_text, creating_features
from loss import CoralLoss

from torch.nn.utils.rnn import pad_sequence

from sklearn.decomposition import PCA

# from catboost import CatBoostClassifier
# from catboost import Pool, cv
# from catboost import CatBoost, Pool, MetricVisualizer


def setup():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    lemmatize = nltk.WordNetLemmatizer()
    procces_text = preprocces_text(tokenizer, lemmatize)

    bert = DistilBertModel.from_pretrained('distilbert-base-uncased', output_hidden_states = True)
    emmbedings = WrapBert(bert, procces_text)

    dataset = Dataset('./data', emmbedings, 'train')
    post, comments, scores = dataset[0]
    # print(text.shape, [c[0].shape for c in coms])

    # POSTS = [dataset[i][0] for i in range(40000, 50000)]# len(dataset))]
    # pca_post = PCA(50).fit(POSTS)
    # del POSTS

    # COMS = [dataset[i][1][j] for j in range(5) for i in range(40000, 50000)]
    # pca_coms = PCA(50).fit(COMS)
    # del COMS

    model = RankingModel(bert)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = CoralLoss()

    # indexed_tokens = []
    # for comment in comments:
    #     text, indexed_token, _ = procces_text(comment)
    #     indexed_tokens.append(torch.tensor(indexed_token))

    _, post_indexed_tokens, _ = procces_text(post)
    post_indexed_tokens = torch.tensor(post_indexed_tokens, dtype=torch.int32)
    post_indexed_tokens = pad_sequence([post_indexed_tokens]).permute(1, 0)

    _, com_indexed_tokens, _ = procces_text(comments[0])
    com_indexed_tokens = torch.tensor(com_indexed_tokens, dtype=torch.int32)
    com_indexed_tokens = pad_sequence([com_indexed_tokens]).permute(1, 0)

    # scores = torch.stack(scores)

    out = model(post_indexed_tokens, com_indexed_tokens, 
                torch.tensor([len(t) for t in post_indexed_tokens]), torch.tensor([len(t) for t in com_indexed_tokens]))

    optimizer.zero_grad()
    loss = criterion(out, scores[0].unsqueeze(0))
    loss.backward()
    optimizer.step()

    print(loss)

    print("SUCCESS")

    # coms_f = creating_features(text, coms)
    # print(coms_f)





if __name__ == '__main__':
    setup()