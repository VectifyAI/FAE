import argparse
import os
from sentence_transformers import SentenceTransformer, InputExample
from eval import evaluate
import json
from torch.utils.data import DataLoader
from loss import *
from embedding import SentenceTransformerEncoder
import numpy as np

seed=100
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
np.random.seed(seed)

parser = argparse.ArgumentParser(description="Fine-tuning the Sentence Transformer model.")
parser.add_argument("--batch_size", type=int, default=20, help="Batch size for training.")
parser.add_argument("--device", type=str, default='cuda:0', help="GPU")

MODEL_NAME="BAAI/bge-small-en"

args = parser.parse_args()

batch_size = args.batch_size
device= args.device

# Create results folder based on batch size
results_folder = f'../save/{MODEL_NAME}/'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

TRAIN_DATASET_FPATH = '../data/train_dataset.json'
VAL_DATASET_FPATH = '../data/val_dataset.json'

with open(TRAIN_DATASET_FPATH, 'r+') as f:
    train_dataset = json.load(f)

with open(VAL_DATASET_FPATH, 'r+') as f:
    val_dataset = json.load(f)

corpus = train_dataset['corpus']
queries = train_dataset['queries']
relevant_docs = train_dataset['relevant_docs']

examples = []
for query_id, query in queries.items():
    node_id = relevant_docs[query_id][0]
    text = corpus[node_id]
    example = InputExample(texts=[query, text])
    examples.append(example)

loader = DataLoader(examples, batch_size=batch_size)

model = SentenceTransformer(MODEL_NAME).to(device)
loss = MultipleNegativesRankingLoss(model=model)


MAX_EPOCHS = 20
previous_hit_rate = 0
consecutive_no_improve_epochs = 0
hit_rates = []

encoder=SentenceTransformerEncoder(model_name=MODEL_NAME)
hit_rate_temp = evaluate(val_dataset, encoder)
hit_rates.append(hit_rate_temp)
previous_hit_rate=hit_rate_temp

print(f"Epoch 0 - Hit Rate: {hit_rate_temp:.4f}")

for epoch in range(1, MAX_EPOCHS + 1):
    if epoch != 1:
        model = SentenceTransformer(os.path.join(results_folder, f'epoch_{epoch-1}')).to(device)
        loss = MultipleNegativesRankingLoss(model=model)
        
    model_output_path = os.path.join(results_folder, f'epoch_{epoch}')
    model.fit(train_objectives=[(loader, loss)], 
              epochs=1, 
              show_progress_bar=True,
              evaluation_steps=50,
              warmup_steps=int(len(loader) * 0.1), 
              output_path=model_output_path)
    
    encoder=SentenceTransformerEncoder(model_name=model_output_path)
    hit_rate_temp = evaluate(val_dataset, encoder)
    hit_rates.append(hit_rate_temp)

    print(f"Epoch {epoch} - Hit Rate: {hit_rate_temp:.4f}")

    if hit_rate_temp > previous_hit_rate:
        previous_hit_rate = hit_rate_temp
        consecutive_no_improve_epochs = 0
    else:
        consecutive_no_improve_epochs += 1

    if consecutive_no_improve_epochs >= 4:
        print("Early stopping due to no improvement in hit rate.")
        break

        
print(hit_rates)
with open('../results/bge-small-en_eval.npy', 'wb') as f:
    np.save(f, hit_rates)
