import nltk

from transformers import BertTokenizer, BertModel
from dataset import Dataset
from model import WrapBert, BaseModel
from proccesing import preprocces_text, creating_features

from sklearn.decomposition import PCA

# from catboost import CatBoostClassifier
# from catboost import Pool, cv
# from catboost import CatBoost, Pool, MetricVisualizer


def setup():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    lemmatize = nltk.WordNetLemmatizer()
    procces_text = preprocces_text(tokenizer, lemmatize)

    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True)
    emmbedings = WrapBert(model, procces_text)

    dataset = Dataset('./data', emmbedings, 'train')
    text, coms = dataset[0]
    print(text.shape, [c[0].shape for c in coms])

    POSTS = [dataset[i][0] for i in range(40000, 50000)]# len(dataset))]
    pca_post = PCA(50).fit(POSTS)
    del POSTS

    COMS = [dataset[i][1][j] for j in range(5) for i in range(40000, 50000)]
    pca_coms = PCA(50).fit(COMS)
    del COMS

    print("SUCCESS")

    # coms_f = creating_features(text, coms)
    # print(coms_f)





if __name__ == '__main__':
    setup()