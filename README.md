# Fine-tuning Augmented Embedding

A method to fine-tune the black-box OpenAI embedding to improve the retreival performance.

See our blog post [How to Improve Your OpenAI Embeddings?
](https://vectify.ai/blog/HowToImproveYourOpenAIEmbeddings) for a detailed introduction of the FAE method.

Other training details and dataset construction follows [LlamaIndex's Blog](https://medium.com/llamaindex-blog/fine-tuning-embeddings-for-rag-with-synthetic-data-e534409a3971).

Install 
```
pip install -r requirements.txt
```

Fine-tune the BBAI/BAAI/bge-small-en model

```
python finetune/train.py
```

Fine-tune the FAE model
```
python finetune/fae_train.py
```


The comparison of the evaluation results can be found in  finetune/eval.ipynb.


Looking to improve the embedding model for your LLM application? [Get early access](https://ii2abc2jejf.typeform.com/to/BKDVoklr)  to Vectify's embedding model fine-tuning service.


Vectify AI

