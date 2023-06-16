"""
Conditions
-----------
This module defines conditions for transitions between nodes.
"""
from typing import cast

from dff.script import Context
from dff.pipeline import Pipeline
from dff.messengers.telegram import TelegramMessage


def received_text(ctx: Context, _: Pipeline):
    """Return true if the last update from user contains text."""
    last_request = ctx.last_request

    #print("text:",last_request.text)
    if last_request.text is not None:
        if "switch to" not in last_request.text and "/restart" not in last_request.text:
            return True
    return False
        

def received_change(ctx: Context, _: Pipeline):
    """Return true if the last update from user contains changing request."""
    last_request = ctx.last_request

    #print("change:",last_request.text)
    if last_request.text is not None:
        if "switch to" in last_request.text:
            return True
    return False


def received_button_click(ctx: Context, _: Pipeline):
    """Return true if the last update from user is a button press."""
    if ctx.validation:  # Regular `Message` doesn't have `callback_query` field, so this fails during validation
        return False
    last_request = cast(TelegramMessage, ctx.last_request)

    return vars(last_request).get("callback_query") is not None
