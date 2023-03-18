import re
import torch
import nltk
nltk.download("stopwords") # поддерживает удаление стоп-слов

from nltk.corpus import stopwords

from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA

class preprocces_text():
    def __init__(self, tokenizer, lemmatize) -> None:
        self.tokenizer = tokenizer
        self.lemmatize = lemmatize
        self.STOPWORDS = stopwords.words()

    def __call__(self, text):        
        #удаляем неалфавитные символы
        text = text.lower().replace("ё", "е")
        text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '[URL]', text)
        text = re.sub('@[^\s]+', '[USER]', text)
        text = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', text)
        text = re.sub(' +', ' ', text)
        text = text.strip()
        text = "[CLS] " + text   + " [SEP]"
        # токенизируем слова
        text = self.tokenizer.tokenize(text)
        # лемматирзируем слова
        text = [self.lemmatize.lemmatize(word) for word in text if word not in self.STOPWORDS]

        indexed_tokens = self.tokenizer.convert_tokens_to_ids(text)
        segments_ids = [1] * len(text)

        assert len(text) > 0
        
        # return text, indexed_tokens, segments_ids
        return torch.tensor(indexed_tokens, dtype=torch.int32)


def creating_features(post, coms, pca=None):
    features = []
    new_features = {r: [] for r in range(5)}

    if pca:
        post_pca = pca.transform(post)

    for i in range(5):
        new_features[i] += [cosine(post, coms[i][0])]
        features.append([post, coms[i][0], cosine(post, coms[i][0])])

    return new_features # 5, k + k + n, k - size 0f text embeddings, n - additionaly features
