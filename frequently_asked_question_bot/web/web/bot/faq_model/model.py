"""
Model
-----
This module defines AI-dependent functions.
"""

import torch
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from transformers import AutoTokenizer, AutoModel
from datasets import load_from_disk
import pickle

from pathlib import Path
from rank_bm25 import BM25Okapi

from numpy import dot
from numpy.linalg import norm
import bot.faq_model.embed as embed

model = ['Luyu/co-condenser-marco-retriever', 'cointegrated/LaBSE-en-ru',
         'OpenMatch/cocodr-large-msmarco','sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
         'DeepPavlov/distilrubert-base-cased-conversational']
BM25=len(model)
cur_index = 0

p_path = './pickle/'
data_path = './ruonly_dataset/'

tokenizer = [AutoTokenizer.from_pretrained(model[i]) for i in range(len(model))]
encoder = [AutoModel.from_pretrained(model[i]) for i in range(len(model))]
for i in range(len(model)):
    encoder[i].to(embed.device)


ds = load_from_disk(data_path)


dox = list(ds['document_plaintext'])
questions = list(ds['question_text']) + list(ds['question_text_2'])
q_len = len(questions)//2

def load_model(model_name,tokenizer,encoder):
    """ Loads model from pickle"""
    emb_file = Path(p_path+"emb_list_"+model_name.split("/")[1])

    if emb_file.is_file():
        with open(p_path+"emb_list_"+model_name.split("/")[1], "rb") as e_f:   # Unpickling
            emb_list = pickle.load(e_f)
    else:
        emb_list = []
        CHL = 100
        iter_n = len(questions)//CHL
        for idx in range(iter_n):
            q_emb = embed.encode(questions[idx*CHL:(idx+1)*CHL],tokenizer,encoder,model_name)
            #q_emb = np.squeeze(q_emb)
            emb_list.extend(q_emb)
            print(F"\rQuestion: {idx}",end='')
        print("\n")
        if len(questions)%CHL > 0:
            q_emb = embed.encode(questions[iter_n*CHL:len(questions)],tokenizer,encoder,model_name)
            emb_list.extend(q_emb)
        #print(emb_list)
            print(F"\rQuestion: {len(questions)%CHL}")
        with open(p_path+"emb_list_"+model_name.split("/")[1], "wb") as e_f:   #Pickling
            pickle.dump(emb_list, e_f)
    return emb_list


tokenized_corpus = [doc.split(" ") for doc in questions]
bm25 = BM25Okapi(tokenized_corpus)

emb_list = [load_model(model[i],tokenizer[i],encoder[i]) for i in range(len(model))]

def find_similar_question(question: str):
    """Return the most similar question from the faq database."""

    print("cur_model:",cur_index)

    if cur_index == BM25:
        doc = bm25.get_top_n(question.split(" "),questions,n=1)[0]

        for i,q in enumerate(questions):
            if doc[:30] in q:
                    res = (q,i%q_len)
                    break
        return res

    q_emb = embed.encode(question,tokenizer[cur_index],encoder[cur_index],model[cur_index])
    emb = np.squeeze(q_emb)
    emb_with_scores = tuple(zip(questions,list(range(q_len))+list(range(q_len)), map(lambda x: embed.cos(x,emb), emb_list[cur_index])))
    res= sorted(filter(lambda x: x[2] > 0.1, emb_with_scores), key=lambda x: x[2])[-1]
    result = (res[0],res[1])
    #print("result:",result)
    #print("sorted:",sorted_tuple[:5])
    return result


# def find_similar_question(question: str) -> str | None:
#     """Return the most similar question from the faq database."""
#     questions = list(map(lambda x: "<Q>" + x, faq.keys()))
#     q_emb, *faq_emb = model.encode(["<Q>" + question] + questions)
#
#     emb_with_scores = tuple(zip(questions, map(lambda x: np.linalg.norm(x - q_emb), faq_emb)))
#
#     sorted_embeddings = tuple(sorted(filter(lambda x: x[1] < 10, emb_with_scores), key=lambda x: x[1]))
#
#     if len(sorted_embeddings) > 0:
#         return sorted_embeddings[0][0].removeprefix("<Q>")
#     return None
