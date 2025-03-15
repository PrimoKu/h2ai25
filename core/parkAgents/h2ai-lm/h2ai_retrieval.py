import json
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')
from rank_bm25 import BM25Okapi
import os, pickle
import numpy as np
from tqdm import tqdm

def process_all_data(json_list):
    """
    Get all BM25
    json_list is the list of json file path (abs path)
    """
    for j in tqdm(json_list):
        with open(j, 'r') as file:
            data = json.load(file)

        keys = list(data.keys())
        corpus = [word_tokenize(str(data[key]).lower()) for key in keys]
        bm25 = BM25Okapi(corpus)
        directory_path = os.path.dirname(j)
        file_name = os.path.basename(j)
        with open(
            os.path.join(directory_path,file_name+".bm25_with_keys.pkl"), 'wb'
        ) as file:
            pickle.dump((bm25,keys), file)

def relevent_retrieve(prompt, bm25_pkl, json_data, n):
    """
    Retieve most relevent entries
    bm25_pkl is a tuple pair (bm25, keys)
    n is top n
    """
    query_tokens = word_tokenize(prompt.lower())
    (bm25, keys) = bm25_pkl
    scores = bm25.get_scores(query_tokens)
    top_n_indices = np.argsort(scores)[::-1][:n]
    top_n_keys = [keys[i] for i in top_n_indices]
    o = {}
    for k in top_n_keys:
        o[k] = json_data[k]

    return o

def main():
    """
    Test class
    """
    json_list = [
        "expert_knowledge/expert_knowledge.json",
    ]
    process_all_data(json_list)
    
if __name__ == '__main__':
    main()