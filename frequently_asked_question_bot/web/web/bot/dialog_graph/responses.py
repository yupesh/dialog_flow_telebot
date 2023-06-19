"""
Responses
---------
This module defines different responses the bot gives.
"""
import sys
import os
sys.path.insert(1,'./')
sys.path.insert(2,'../')

from dff.script import Context
from dff.script import Message
from dff.pipeline import Pipeline
#from ..faq_model.model import faq
import bot.faq_model.utils as faq
import bot.faq_model.model as fm

from spacy.lang.ru import Russian

nlp = Russian()
nlp.add_pipe('sentencizer')

def get_bot_answer(question: str, doc: str) -> Message:
    """Forms bot answer from doc for asked question"""
    if doc != "":
        message = F"Q: {question} <br> A: {doc}"
    else:
        message = question
    #print(message)
    return Message(text=message)


# def get_bot_answer(question: str) -> Message:
#     """The Message the bot will return as an answer if the most similar question is `question`."""
#     return Message(text=f"Q: {question} <br> A: {faq[question]}")


FALLBACK_ANSWER = Message(
    text=F"I do not have an answer to that question. Here is a list of questions I know an answer to",
)
"""Fallback answer that the bot returns if user's query is not similar to any of the questions."""


FIRST_MESSAGE = Message(
    text="Welcome! Ask your question. Type *change model* to try different retrievers"
)

FALLBACK_NODE_MESSAGE = Message(
    text="Something went wrong.\n"
         "You may continue asking me questions."
)


def answer_similar_question(ctx: Context, _: Pipeline):
    print("last_request:",ctx.last_request)
    """Answers with the most similar question to user's query. Or changes current retriever's model"""
    if ctx.validation:  # this function requires non-empty fields and cannot be used during script validation
        return Message()
    last_request = ctx.last_request
    if last_request is None:
        raise RuntimeError("No last requests.")
    if last_request.annotations is None:
        raise RuntimeError("No annotations.")

    #print("text:",last_request.text)

    if last_request.text == "change model":
        fm.cur_index = (fm.cur_index+1)%len(faq.model)
        model_name = faq.model[fm.cur_index]
        return get_bot_answer(F"model changed to {model_name}","")

    similar_index = last_request.annotations.get("similar_question")
    #print("similar_index:",similar_index)

    similar_question  = faq.questions[similar_index]
    if faq.model[fm.cur_index] == 'cointegrated/LaBSE-en-ru': # We collected fragments answers from tydi-qa for this model
        doc = faq.labse_dox[similar_index]
    else:
        doc = faq.dox[similar_index]
        doc = nlp(doc) # showing only part of the whole document, that's why splitting into sentences
        sentences = [sent.text.strip() for sent in doc.sents]
        chunk = 30
        while len("".join(sentences[:chunk])) > 2000:
               chunk -= 1
        doc = " ".join(sentences[:chunk])

    if similar_question is None:  # question is not similar to any of the questions
        return FALLBACK_ANSWER
    else:
        return get_bot_answer(similar_question,doc)
