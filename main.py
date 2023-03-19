import nltk
import torch
import time

from transformers import BertTokenizer, BertModel, DistilBertModel, DistilBertTokenizer
from dataset import Dataset
from model import WrapBert, BaseModel, RankingModel
from proccesing import preprocces_text, creating_features
from loss import CoralLoss
from dataloader import get_dataloader
from process import train

from torch.nn.utils.rnn import pad_sequence

from sklearn.decomposition import PCA

# from catboost import CatBoostClassifier
# from catboost import Pool, cv
# from catboost import CatBoost, Pool, MetricVisualizer


def setup():
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    lemmatize = nltk.WordNetLemmatizer()
    procces_text = preprocces_text(tokenizer, lemmatize)

    bert = DistilBertModel.from_pretrained('distilbert-base-uncased', output_hidden_states = True)
    emmbedings = WrapBert(bert, procces_text)

    dataset = Dataset('./data', purpose='train', transform=procces_text)

    dataloader = get_dataloader(dataset)

    # POSTS = [dataset[i][0] for i in range(40000, 50000)]# len(dataset))]
    # pca_post = PCA(50).fit(POSTS)
    # del POSTS

    # COMS = [dataset[i][1][j] for j in range(5) for i in range(40000, 50000)]
    # pca_coms = PCA(50).fit(COMS)
    # del COMS

    device = 'cpu'
    model = RankingModel(bert).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
    # criterion = CoralLoss()
    criterion = torch.nn.CrossEntropyLoss()

    max_epoch = 50
    for epoch in range(max_epoch):
        epoch_time = time.time()
        loss = train(model, dataloader, optimizer, criterion, device)

        print(f"epoch {epoch} time {time.time()-epoch_time} loss {loss}")

    # out = model(post_indexed_tokens, com_indexed_tokens, 
    #             torch.tensor([len(t) for t in post_indexed_tokens]), torch.tensor([len(t) for t in com_indexed_tokens]))

    # optimizer.zero_grad()
    # loss = criterion(out, scores[0].unsqueeze(0))
    # loss.backward()
    # optimizer.step()

    # print(loss)

    print("SUCCESS")

    # coms_f = creating_features(text, coms)
    # print(coms_f)





if __name__ == '__main__':
    setup()