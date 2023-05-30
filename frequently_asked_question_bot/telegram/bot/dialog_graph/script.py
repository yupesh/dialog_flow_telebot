"""
Script
--------
This module defines a script that the bot follows during conversation.
"""
from dff.script import RESPONSE, TRANSITIONS, LOCAL
import dff.script.conditions as cnd
from dff.messengers.telegram import (
    TelegramMessage,
    TelegramUI
)
from faq_model.model import model
from dff.script.core.message import Button

from .responses import answer_question, suggest_similar_questions, change_model
from .conditions import received_button_click, received_text, received_change

buttons = [Button(text="switch to "+model[i].split("/")[1]) for i in range(len(model))] + [Button(text="switch to bm25")]

script = {
    "service_flow": {
        "start_node": {
            TRANSITIONS: {("qa_flow", "welcome_node"): cnd.exact_match(TelegramMessage(text="/start"))},
        },
        "fallback_node": {
            RESPONSE: TelegramMessage(text="Something went wrong. Use `/restart` to start over."),
            TRANSITIONS: {("qa_flow", "welcome_node"): cnd.exact_match(TelegramMessage(text="/restart"))},
        },
    },
    "qa_flow": {
        LOCAL: {
            TRANSITIONS: {
                ("qa_flow", "change_model"): received_change,
                ("qa_flow", "suggest_questions"): received_text,
                ("qa_flow", "answer_question"): received_button_click,
                
            },
        },
        "welcome_node": {
            RESPONSE: TelegramMessage(text="Welcome! Ask your question",
                   #ui=TelegramUI(buttons=buttons, is_inline = False, row_width=1
                   ui=TelegramUI(buttons=buttons, is_inline = False
                                )
                      ),
        },
        "suggest_questions": {
            RESPONSE: suggest_similar_questions,
        },
        "answer_question": {
            RESPONSE: answer_question,
        },
        "change_model": {
            RESPONSE: change_model,
        },
    },
}
