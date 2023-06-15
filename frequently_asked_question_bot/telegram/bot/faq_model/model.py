# """
# Model
# -----
# This module defines AI-dependent functions.
# """

import os
import torch
from pathlib import Path
from tqdm.autonotebook import tqdm
import string

import numpy as np
from transformers import AutoTokenizer, AutoModel
from sentence_transformers.cross_encoder import CrossEncoder
from datasets import load_from_disk
import pickle

from pathlib import Path
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords

import faq_model.embed as embed

#nltk.download('stopwords')
ru_stopwords = stopwords.words('russian')

model = ['Luyu/co-condenser-marco-retriever', 'cointegrated/LaBSE-en-ru',
         'sentence-transformers/multi-qa-distilbert-cos-v1','sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
         'DeepPavlov/distilrubert-base-cased-conversational','OpenMatch/cocodr-base-msmarco']

BM25=len(model)

cur_index = 1

data_path = '/home/yuri/ruonly_dataset/'
p_path = '/home/yuri/pickle/'
#p_path = '/Users/yuriypeshkichev/Projects/itmo internship/pickle/'
#p_path = '/app/pickle/'
#data_path = '/Users/yuriypeshkichev/Projects/itmo internship/ruonly_dataset/'
#data_path = '/app/ru_dataset_3/'
model_save_path = '/home/yuri/saved_models/' #Path to save model benchmarks

tokenizer = [AutoTokenizer.from_pretrained(model[i]) for i in range(len(model))]
encoder = [AutoModel.from_pretrained(model[i]) for i in range(len(model))]
for i in range(len(model)):
    encoder[i].to(embed.device)

ce_path = max(Path(model_save_path).glob('*/'), key=os.path.getmtime)
print(ce_path)
ce_model = CrossEncoder(ce_path, num_labels=1, max_length = 512)

ds = load_from_disk(data_path)

dox = list(ds['document_plaintext'])
questions = list(ds['question_text']) + list(ds['question_text_2'])

q_len = len(dox)

def find_best_answer(example,m_name,toker,encer):

    start = example['passage_answer_candidates']['plaintext_start_byte']
    end = example['passage_answer_candidates']['plaintext_end_byte']
    cand_list = [example['document_plaintext'].encode("utf-8")[s:e].decode("utf-8", errors="replace") for s,e in zip(start,end)]
    if m_name == "BM25":
        example["best_answer"] = bm25.get_top_n(example['question_text'].split(" "),cand_list,n=1)[0]
    else:

        CHL = 100
        iter_n = len(cand_list)//CHL
        emb_list = []
        flag = 0
        for idx in range(iter_n):
            if flag == 0:
                cand_emb = embed.encode([example['question_text']]+cand_list[idx*CHL:(idx+1)*CHL],toker,encer,m_name)
                flag = 1
            else:
                cand_emb = embed.encode(cand_list[idx*CHL:(idx+1)*CHL],toker,encer,m_name)
            emb_list.extend(cand_emb)
        if len(cand_list)%CHL > 0:
            if flag == 0:
                cand_emb = embed.encode([example['question_text']]+cand_list[iter_n*CHL:len(cand_list)],toker,encer,m_name)
            else:
                cand_emb = embed.encode(cand_list[iter_n*CHL:len(cand_list)],toker,encer,m_name)
            emb_list.extend(cand_emb)
          
        
        emb_with_scores = tuple(zip(list(range(len(start))), map(lambda x: embed.cos(x,emb_list[0]), emb_list[1:])))
        res= sorted(emb_with_scores, key=lambda x: x[1])[-1]
        #example["best_answer"] = cand_list[res[0]]
        
    return cand_list[res[0]]


def load_model(model_name,tokenizer,encoder):
    """ Loads model from pickle"""
    
    emb_file = Path(p_path+"emb_list_"+model_name.split("/")[1])
    
    if emb_file.is_file():
        with open(p_path+"emb_list_"+model_name.split("/")[1], "rb") as e_f:   # Unpickling
            emb_list = pickle.load(e_f)
    else:

        emb_list = []
        CHL = 100
        iter_n = 2*q_len//CHL
        for idx in range(iter_n):
            q_emb = embed.encode(questions[idx*CHL:(idx+1)*CHL],tokenizer,encoder,model_name)
            #q_emb = np.squeeze(q_emb)
            emb_list.extend(q_emb)
            print(F"\rQuestion: {idx}",end='')
        print("\n")
        if 2*q_len%CHL > 0:
            q_emb = embed.encode(questions[iter_n*CHL:2*q_len],tokenizer,encoder,model_name)
            emb_list.extend(q_emb)
        #print(emb_list)
            print(F"\rQuestion: {2*q_len%CHL}")
        with open(p_path+"emb_list_"+model_name.split("/")[1], "wb") as e_f:   #Pickling
            pickle.dump(emb_list, e_f)
    return emb_list


def load_labse():

    labse_name = 'cointegrated/LaBSE-en-ru'
    labse_index = model.index(labse_name)
    
    labse_file = Path(p_path+"answer_list_LaBSE-en-ru")
    
    if labse_file.is_file():
        with open(p_path+"answer_list_LaBSE-en-ru", "rb") as a_f:
            labse_list = pickle.load(a_f)
    else:
        labse_list = []
        for ind in range(len(ds)):
            labse_list.append(find_best_answer(ds[ind],labse_name,tokenizer[labse_index],encoder[labse_index]))
            print(F"\rDocument: {ind}",end='')
    
        with open(p_path+"answer_list_LaBSE-en-ru", "wb") as a_f:   #Pickling
            pickle.dump(labse_list, a_f)            
    return labse_list

# We lower case our text and remove stop-words from indexing
def bm25_tokenizer(text):
    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(string.punctuation)

        if len(token) > 0 and token not in ru_stopwords:
            tokenized_doc.append(token)
    return tokenized_doc

tokenized_corpus = []
for passage in tqdm(questions):
    tokenized_corpus.append(bm25_tokenizer(passage))

bm25 = BM25Okapi(tokenized_corpus)

emb_list = [load_model(model[i],tokenizer[i],encoder[i]) for i in range(len(model))]



labse_dox = load_labse()

def test_ce(query,doc):
    return ce_model.predict([query,doc], convert_to_numpy=True, show_progress_bar=False)


def find_similar_questions(question: str):
    """Return a list of similar questions from the database."""

    print("cur_index:",cur_index)
    
    if cur_index == BM25:
        doc = bm25.get_top_n(question.split(" "),questions,n=30)
        #print("doc:",doc[:100])

        res_tuple = ()
        seen = []
        for d in doc:
            if d not in seen:
                print("ddd:",d)
                print("seen:",seen)
                for i,q in enumerate(questions):
                    if q not in seen:
                        if d[:20] in q:
                            res_tuple += ((q,i%q_len),)
                            seen.append(q)
                            break
            if len(res_tuple) == 5:
                break

        #print("tuple:",res_tuple)
        return res_tuple

                           
                           
    q_emb = embed.encode(question,tokenizer[cur_index],encoder[cur_index],model[cur_index])
    emb = np.squeeze(q_emb)                       
    #print(emb)
    
    emb_with_scores = tuple(list(range(q_len))+list(range(q_len)), map(lambda x: embed.cos(x,emb), emb_list[cur_index]))
    #print("Here!!!!")
    res_list = []
    seen = []
    if model[cur_index] == "Luyu/co-condenser-marco-retriever":
        top_list = []
        for el in sorted(filter(lambda x: x[1] > 0.1, emb_with_scores), key=lambda x: x[1])[-30:]:
            top_list.append((el[0],test_ce(question,dox[el[0]])))

        sorted_tuple = sorted(top_list, key=lambda el: el[1],reverse=True)
    else:
        sorted_tuple = sorted(filter(lambda x: x[1] > 0.1, emb_with_scores), key=lambda x: x[1], reverse=True)
        #print("sorted:",sorted_tuple[:5])
    for item in sorted_tuple:
        if item[0] not in res_list:
            res_list.append(item[0])
    return res_list[:5]

