# Fine-tuning Augmented Embedding

A method to fine-tune the black-box OpenAI embedding to improve the retrieval performance.

See our blog post [How to Improve Your OpenAI Embeddings?
](https://vectify.ai/blog/HowToImproveYourOpenAIEmbeddings) for a detailed introduction to the FAE method.

Other training details and dataset construction can be found in [LlamaIndex's Blog](https://medium.com/llamaindex-blog/fine-tuning-embeddings-for-rag-with-synthetic-data-e534409a3971).

Install 
```
pip install -r requirements.txt
```

Fine-tune the BAAI/bge-small-en model

```
cd ./finetune
python train.py
```

Fine-tune the FAE model
```
cd ./finetune
python fae_train.py
```


The comparison of the evaluation results can be found in the [jupyter notebook](https://github.com/VectifyAI/FAE/blob/main/finetune/eval.ipynb).


[Get early access](https://ii2abc2jejf.typeform.com/to/BKDVoklr)  to our RAG platform and embedding model fine-tuning service.


Vectify AI

