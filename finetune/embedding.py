from dotenv import load_dotenv
load_dotenv()
import os
import openai
from abc import ABC, abstractmethod
from typing import List
import torch
import time
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
import pickle



OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = "text-embedding-ada-002"

class Encoder(ABC):
    """ Base interface for encoding model used to create embeddings"""
    @abstractmethod
    def embed(self, inputs: List[str]):
        """ Embed a single input """
        raise NotImplementedError



class OpenAIEncoder(Encoder):
    def __init__(self):
        super().__init__()
        self.name = 'openai'

    def embed(self, inputs: List[str]):
        print('money')
        try:
            results = openai.Embedding.create(
            api_key=OPENAI_API_KEY, input=inputs, engine=OPENAI_EMBEDDING_MODEL
            )
            return [item['embedding'] for item in results['data']]
        except openai.error.APIError as e:
            print(f"OpenAI API returned an API Error: {e}")
            pass
        except openai.error.APIConnectionError as e:
            print(f"Failed to connect to OpenAI API: {e}")
            time.sleep(5)
            self.safe_embed(inputs)
            pass
        except openai.error.RateLimitError as e:
            print(f"OpenAI API request exceeded rate limit: {e}")
            time.sleep(5)
            self.safe_embed(inputs)
            pass

        

class SentenceTransformerEncoder(Encoder):
    def __init__(self, model_name, device="cpu",):
        super().__init__()
        self.name='sentenceTransformer'
        self.device = device
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)


    def embed(self, sentences: List[str]):
        # Apply tokenizer
        with torch.no_grad():
            return self.model.encode(sentences)
        
        
        
class FAEEncoder(Encoder):
    def __init__(self, model_name, device="cpu", if_load=True):
        super().__init__()
        self.name='fae'
        self.openai_encoder=OpenAIEncoder()
        self.device = device
        self.load_openai=if_load
        self.model = SentenceTransformer(model_name)
        self.model.to(self.device)
        
        if self.load_openai:
            ## load pre-saved OpenAI embeddings
            with open('../data/val_query_openai.pkl', 'rb') as f:
                val_query_embeddings = pickle.load(f)
            with open('../data/val_corpus_openai.pkl', 'rb') as f:
                val_corpus_embeddings = pickle.load(f)

            self.val_dicts = {**val_query_embeddings, **val_corpus_embeddings}
        else:
            self.val_dicts=None
            

    def embed(self, sentences: List[str], sentences_id: List[str]):
        if self.load_openai:
            ### use the saved OpenAI embeddings
            openai_embeddings=[self.val_dicts[sentence_id] for sentence_id in sentences_id]
        else:
            ### get the embeddings from OpenAI API
            openai_embeddings=self.openai_encoder.embed(sentences)

        with torch.no_grad():
            model_embeddings=self.model.encode(sentences)
            ### Since the OpenAI embeddings are normalized, we also need to normalize the augmented embeddings
            normalized_model_embeddings=model_embeddings/np.linalg.norm(model_embeddings)
            joint_embeddings=np.concatenate((np.asarray(openai_embeddings), normalized_model_embeddings),1)
        return joint_embeddings
        
        
        


        
def batch_embed(passage_list, encoder, passage_id_list=None, batch_size=64):
    embedding_list=[]
    for i in tqdm(range(0, len(passage_list), batch_size)):
        batch_passage = passage_list[i:i+batch_size]
        batch_id=passage_id_list[i:i+batch_size]
        if encoder.name =='fae':
            embedding_list.extend(encoder.embed(batch_passage,batch_id))
        else:
            embedding_list.extend(encoder.embed(batch_passage))
    return embedding_list


def get_embedding_dict(passage_dict,encoder, load_path=None):
    if load_path!=None:
        with open(load_path, 'rb') as f:
            return pickle.load(f)
    else:
        passage_id_list=list(passage_dict.keys())
        passage_list=list(passage_dict.values())
        passage_embeddings_list = batch_embed(passage_list,encoder,passage_id_list)
            
        return dict(zip(passage_id_list,passage_embeddings_list))


