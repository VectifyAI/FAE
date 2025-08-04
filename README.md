# Fine-tuning Augmented Embedding (FAE) 🔄

A method to fine-tune the black-box OpenAI embedding to improve the retrieval performance 📈.

📝 See our paper [Enhancing Black-Box Embeddings with Model Augmented Fine-Tuning](https://arxiv.org/pdf/2402.12177) for a detailed introduction to the FAE method.

Other training details and dataset construction can be found in [LlamaIndex's Blog](https://medium.com/llamaindex-blog/fine-tuning-embeddings-for-rag-with-synthetic-data-e534409a3971).

## Install 🛠️ 
```bash
pip install -r requirements.txt
```

## Fine-tune the BAAI/bge-small-en model
```bash
cd ./finetune
python train.py
```

## Fine-tune the FAE model
```bash
cd ./finetune
python fae_train.py
```

The comparison of the evaluation results can be found in the [jupyter notebook](https://github.com/VectifyAI/FAE/blob/main/finetune/eval.ipynb) 📊.
