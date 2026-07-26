import os
import re
from dataclasses import dataclass
from groq import Groq
from utils.prompts import JUDGE_SYSTEM

FALLBACK_MODEL = "llama-3.1-8b-instant"


@dataclass
class Verdict:
    verdict: str
    reasoning: str
    sentence: str
    one_liner: str
    raw: str


_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


def _parse(raw: str) -> Verdict:
    def grab(label: str, next_labels: list[str]) -> str:
        pattern = rf"{label}:\s*(.*?)(?=(?:{'|'.join(next_labels)}):|\Z)"
        match = re.search(pattern, raw, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    labels = ["VERDICT", "REASONING", "SENTENCE", "ONE-LINER"]
    return Verdict(
        verdict=grab("VERDICT", labels[1:]),
        reasoning=grab("REASONING", labels[2:]),
        sentence=grab("SENTENCE", labels[3:]),
        one_liner=grab("ONE-LINER", []),
        raw=raw,
    )


def deliver_verdict(code: str, transcript: str) -> Verdict:
    resp = _get_client().chat.completions.create(
        model=FALLBACK_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": (
                f"The code on trial:\n```\n{code}\n```\n\n"
                f"Courtroom transcript:\n{transcript}\n\n"
                "Deliver your verdict now, using the exact format specified."
            )},
        ],
        temperature=0.7,
        max_tokens=400,
    )
    raw = resp.choices[0].message.content.strip()
    return _parse(raw)