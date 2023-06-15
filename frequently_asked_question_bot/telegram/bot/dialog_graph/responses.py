"""
Responses
---------
This module defines different responses the bot gives.
"""
from typing import cast

from dff.script import Context
from dff.pipeline import Pipeline
from dff.script.core.message import Button
from dff.messengers.telegram import TelegramMessage, TelegramUI, ParseMode
import faq_model.model as faq
from spacy.lang.ru import Russian


nlp = Russian()
nlp.add_pipe('sentencizer')

def suggest_similar_questions(ctx: Context, _: Pipeline):
    """Suggest questions similar to user's query by showing buttons with those questions."""
    if ctx.validation:  # this function requires non-empty fields and cannot be used during script validation
        return TelegramMessage()
    last_request = ctx.last_request
    if last_request is None:
        raise RuntimeError("No last requests.")
    if last_request.annotations is None:
        raise RuntimeError("No annotations.")
    similar_questions = last_request.annotations.get("similar_questions")
    if similar_questions is None:
        raise RuntimeError("Last request has no text.")
    if len(similar_questions) == 0:  # question is not similar to any questions
        return TelegramMessage(
                text="I don't have an answer to that question. Here's a list of questions I know an answer to:",
                ui=TelegramUI(buttons=[Button(text=q, payload=str(i)) for i,q in enumerate(questions[:5])]),
            )
    return TelegramMessage(
                    text="I found similar questions in my database                                                                                                   :",
                    ui=TelegramUI(buttons=[Button(text=faq.questions[q], payload=str(q)) for q in similar_questions], row_width = 1)
                )





# only select necessary pipeline components to speed up processing


def answer_question(ctx: Context, _: Pipeline):

    """Answer a question asked by a user by pressing a button."""
    if ctx.validation:  # this function requires non-empty fields and cannot be used during script validation
        return TelegramMessage()
    print("INN2")
    last_request = ctx.last_request
    if last_request is None:
        raise RuntimeError("No last requests.")

    last_request = cast(TelegramMessage, last_request)
    if last_request.callback_query is None:
        raise RuntimeError("No callback query")
    
    print("index:",last_request.callback_query)
    
    if faq.model[faq.cur_index] == 'cointegrated/LaBSE-en-ru': # with labse model we collected best answer fragments from tydi-qa
        doc = faq.labse_dox[int(last_request.callback_query)]
    else:    
        doc = nlp(faq.dox[int(last_request.callback_query)]) # showing only part of the whole document, that's why splitting into sentences 
        sentences = [sent.text.strip() for sent in doc.sents]
        chunk = 30
        while len("".join(sentences[:chunk])) > 4000:
               chunk -= 1
        doc = " ".join(sentences[:chunk])
    return TelegramMessage(text="<b>This is my answer:</b>\n"+doc,parse_mode=ParseMode.HTML)


def change_model(ctx: Context, _: Pipeline):
    """Change retriever model by user's request"""
    if ctx.validation:  # this function requires non-empty fields and cannot be used during script validation
        return TelegramMessage()
    
    last_request = ctx.last_request
    model_name = last_request.text.split(" ")[2]
    print("model_name:",model_name)
    print("cur_index:",faq.cur_index)
    if model_name == "bm25":
        if faq.cur_index == faq.BM25:
            return TelegramMessage(text="Current model is BM25")
        else:
            faq.cur_index = faq.BM25
            print("cur_index changed to:",faq.cur_index)
    else:
        if faq.cur_index != faq.BM25:
            
            if model_name in faq.model[faq.cur_index]:
                return TelegramMessage(text=F"Current model is {faq.model[faq.cur_index]}")
    
        for i in range(len(faq.model)):
            if faq.model[i].split("/")[1] in last_request.text:
                faq.cur_index = i
                model_name = faq.model[i]
                break
    return TelegramMessage(text=F"Model changed to {model_name}")