"""
Model
-----
This module defines AI-dependent functions.
"""

import numpy as np

import faq_model.embed as embed
from faq_model.utils import questions, dox, model, tokenizer, encoder, emb_list, q_len, bm25, test_ce

cur_index = 1

def find_similar_questions(question: str):
    """Returns a list of upto 5 similar questions from the database."""

    #print("cur_index:",cur_index)

    if "bm25" in model[cur_index]:
        doc = bm25.get_top_n(question.split(" "),questions,n=30)
        #print("doc:",doc[:100])

        res_list = []
        seen = []
        for d in doc:
            if d not in seen:
                #print("ddd:",d)
                #print("seen:",seen)
                for i,q in enumerate(questions):
                    if q not in seen:
                        if d[:20] in q:
                            res_list.append(i%q_len)
                            seen.append(q)
                            break
            if len(res_list) == 5:
                break

        #print("tuple:",res_tuple)
        return res_list

    q_emb = embed.encode(question,tokenizer[cur_index],encoder[cur_index],model[cur_index])
    emb = np.squeeze(q_emb)
    #print(emb)

    emb_with_scores = tuple(zip(list(range(q_len))+list(range(q_len)), map(lambda x: embed.cos(x,emb), emb_list[cur_index])))
    #print("Here!!!!")
    res_list = []
    seen = []
    if model[cur_index] == "Luyu/co-condenser-marco-retriever": # For this model we apply crossencoder re-ranking
        top_list = []
        for el in sorted(filter(lambda x: x[1] > 0.1, emb_with_scores), key=lambda x: x[1])[-10:]:
            top_list.append((el[0],test_ce(question,dox[el[0]])))

        sorted_tuple = sorted(top_list, key=lambda el: el[1],reverse=True)
    else:
        sorted_tuple = sorted(filter(lambda x: x[1] > 0.1, emb_with_scores), key=lambda x: x[1], reverse=True)
        #print("sorted:",sorted_tuple[:5])
    for item in sorted_tuple:
        if item[0] not in res_list:
            res_list.append(item[0])
    return res_list[:5]
