"""
Model
-----
This module defines AI-dependent functions.
"""
import numpy as np
import bot.faq_model.embed as embed
from bot.faq_model.utils import BM25, questions, dox, model, tokenizer, encoder, emb_list, q_len, bm25, test_ce

cur_index = 0

def find_similar_question(question: str):
    """Return the most similar question from the faq database."""

    #print("cur_model:",cur_index)

    if cur_index == BM25:
        doc = bm25.get_top_n(question.split(" "),questions,n=1)[0]

        for i,q in enumerate(questions):
            if doc[:30] in q:
                return i%q_len


    
    q_emb = embed.encode(question,tokenizer[cur_index],encoder[cur_index],model[cur_index])
    emb = np.squeeze(q_emb)
    emb_with_scores = tuple(zip(list(range(q_len))+list(range(q_len)), map(lambda x: embed.cos(x,emb), emb_list[cur_index])))
    
    if model[cur_index] == "Luyu/co-condenser-marco-retriever": # For this model we apply crossencoder re-ranking
        top_list = []
        for el in sorted(filter(lambda x: x[1] > 0.1, emb_with_scores), key=lambda x: x[1])[-10:]:
            top_list.append((el[0],test_ce(question,dox[el[0]])))
    
        result = sorted(top_list, key=lambda el: el[1],reverse=True)[0][0]
    else:
        result = sorted(filter(lambda x: x[1] > 0.1, emb_with_scores), key=lambda x: x[1])[-1][0]

    #print("result:",result)
    #print("sorted:",sorted_tuple[:5])
    return result
