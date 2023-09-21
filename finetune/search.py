import faiss


def create_index(embeddings, use_gpu):
    index = faiss.IndexFlatIP(len(embeddings[0]))
    if use_gpu:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = True
        index = faiss.index_cpu_to_all_gpus(index, co=co)
    index.add(embeddings)
    return index


def batch_search(index,
                 query,
                 topk: int = 200,
                 batch_size: int = 64):
    all_scores, all_inxs = [], []
    for start_index in range(0, len(query), batch_size):
        batch_query = query[start_index:start_index + batch_size]
        batch_scores, batch_inxs = index.search(batch_query, k=topk)
        all_scores.extend(batch_scores.tolist())
        all_inxs.extend(batch_inxs.tolist())
    return all_scores, all_inxs