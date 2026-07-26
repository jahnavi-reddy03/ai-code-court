import os
from openai import OpenAI
from utils.prompts import PROSECUTOR_SYSTEM

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def opening_argument(code: str) -> str:
    """GPT-4 tears the code apart for the first time."""
    resp = _get_client().chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": PROSECUTOR_SYSTEM},
            {"role": "user", "content": f"Here is the defendant's code:\n\n```\n{code}\n```\n\nDeliver your opening argument."},
        ],
        temperature=0.8,
        max_tokens=500,
    )
    return resp.choices[0].message.content.strip()


def rebuttal(code: str, defense_argument: str) -> str:
    """GPT-4 responds to the defense's opening argument."""
    resp = _get_client().chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": PROSECUTOR_SYSTEM},
            {"role": "user", "content": (
                f"The defendant's code:\n\n```\n{code}\n```\n\n"
                f"The defense just argued:\n\n\"{defense_argument}\"\n\n"
                "Deliver your rebuttal."
            )},
        ],
        temperature=0.8,
        max_tokens=600,
    )
    return resp.choices[0].message.content.strip()
