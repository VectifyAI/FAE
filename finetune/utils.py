from typing import Union, List
import torch

    
class FAEExample:
    def __init__(self, texts: List[str], given_embeddings, label: Union[int, float] = 0):
        self.texts = texts
        self.given_embeddings= given_embeddings
        self.label = label

    def __str__(self):
        return "<FAEExample> texts: {}".format("; ".join(self.texts))
    
    
    
def fae_smart_batching_collate(model):

    def fae_smart_batching_collate_inner(batch):
        num_texts = len(batch[0].texts)
        texts = [[] for _ in range(num_texts)]
        embeds= [[] for _ in range(num_texts)]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text)
                embeds[idx].append(example.given_embeddings[idx])

            labels.append(example.label)

        labels = torch.tensor(labels)

        sentence_features = []
        for idx in range(num_texts):
            tokenized = model.tokenize(texts[idx])
            tokenized['embeddings']=embeds[idx]
            sentence_features.append(tokenized)

            
        return sentence_features,  labels

    return fae_smart_batching_collate_inner

