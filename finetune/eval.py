from search import create_index, batch_search
from embedding import *

def evaluate(val_dataset, encoder, if_load=True):
    val_corpus = val_dataset['corpus']
    val_queries = val_dataset['queries']
    val_relevant_docs = val_dataset['relevant_docs']
    
    
    if encoder.name=='openai' and if_load==True:
        ### Use the saved OpenAI embeddings, set load_path=None if you want to re-get them from the OpenAI API
        val_query_embeddings=get_embedding_dict(val_queries,encoder, load_path='../data/val_query_openai.pkl')
        val_corpus_embeddings=get_embedding_dict(val_corpus,encoder, load_path='../data/val_corpus_openai.pkl')
    else:
        print('Embed queries...')
        val_query_embeddings=get_embedding_dict(val_queries,encoder)
        print('Embed corpus...')
        val_corpus_embeddings=get_embedding_dict(val_corpus,encoder)
    
    val_index=create_index(np.asarray(list(val_corpus_embeddings.values())),use_gpu=False)

    retrieved_passages = batch_search(val_index, np.asarray(list(val_query_embeddings.values())), topk= 5, batch_size = 64)[1]

    val_corpus_key_list=list(val_corpus.keys())
    hit_num=0
    for query_index, (query_id, query) in enumerate(val_query_embeddings.items()):
        retrieved_indexes=retrieved_passages[query_index]
        retrieved_ids = [val_corpus_key_list[retrieved_index] for retrieved_index in retrieved_indexes]
        expected_id = val_relevant_docs[query_id][0]
        if expected_id in retrieved_ids:
            hit_num+=1

    return hit_num/len(val_queries)
