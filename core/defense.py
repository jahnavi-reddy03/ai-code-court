import os
import time
import google.generativeai as genai
from utils.prompts import DEFENSE_SYSTEM

_configured = False


def _ensure_configured():
    global _configured
    if not _configured:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        _configured = True


def _get_model() -> genai.GenerativeModel:
    _ensure_configured()
    return genai.GenerativeModel(
        model_name="gemini-flash-latest",
        system_instruction=DEFENSE_SYSTEM,
    )


def _generate_with_retry(prompt: str, max_tokens: int, attempts: int = 5) -> str:
    last_error = None
    for i in range(attempts):
        try:
            resp = _get_model().generate_content(
                prompt,
                generation_config={"temperature": 0.8, "max_output_tokens": max_tokens},
                request_options={"timeout": 25},
            )
            return resp.text.strip()
        except Exception as e:
            last_error = e
            time.sleep(2)
    raise last_error


def opening_argument(code: str, prosecutor_argument: str) -> str:
    prompt = (
        f"Here is your client's code:\n\n```\n{code}\n```\n\n"
        f"The prosecutor just argued:\n\n\"{prosecutor_argument}\"\n\n"
        "Deliver your opening argument in defense of this code."
    )
    return _generate_with_retry(prompt, max_tokens=1500)


def rebuttal(code: str, prosecutor_rebuttal: str) -> str:
    prompt = (
        f"Your client's code:\n\n```\n{code}\n```\n\n"
        f"The prosecutor's rebuttal:\n\n\"{prosecutor_rebuttal}\"\n\n"
        "Deliver your closing rebuttal."
    )
    return _generate_with_retry(prompt, max_tokens=1500)



