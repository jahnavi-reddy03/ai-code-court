"""
Runs one full trial end-to-end: opening arguments, one rebuttal round, verdict.
Kept model-call-order explicit here (not hidden in a loop) because the courtroom
"beats" matter for the UI - each step gets rendered to the user as it happens.
"""
from dataclasses import dataclass, field
from core import prosecutor, defense, judge
import json
import os


@dataclass
class TrialTranscript:
    code: str
    prosecutor_opening: str = ""
    defense_opening: str = ""
    prosecutor_rebuttal: str = ""
    defense_rebuttal: str = ""
    verdict: judge.Verdict | None = None
    steps: list[str] = field(default_factory=list)  # log of what happened, for the UI

    def as_text(self) -> str:
        """Flattened transcript, used as context for the judge."""
        return "\n\n".join(filter(None, [
            f"PROSECUTOR (opening): {self.prosecutor_opening}",
            f"DEFENSE (opening): {self.defense_opening}",
            f"PROSECUTOR (rebuttal): {self.prosecutor_rebuttal}",
            f"DEFENSE (rebuttal): {self.defense_rebuttal}",
        ]))


def _save_for_training(t: TrialTranscript):
    """Auto-logs every completed trial as a training example for the judge fine-tune."""
    os.makedirs("finetuning/dataset", exist_ok=True)
    record = {
        "code": t.code,
        "prosecutor": f"{t.prosecutor_opening}\n\n{t.prosecutor_rebuttal}".strip(),
        "defense": f"{t.defense_opening}\n\n{t.defense_rebuttal}".strip(),
        "verdict": t.verdict.raw,
    }
    with open("finetuning/dataset/collected_verdicts.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")


def run_trial(code: str, on_step=None) -> TrialTranscript:
    """
    Runs the full trial synchronously and returns the transcript.
    on_step: optional callback(step_name: str, text: str) fired after each beat,
             so the Streamlit UI can render the courtroom live instead of waiting
             for the whole trial to finish.
    """
    t = TrialTranscript(code=code)

    def emit(step, text):
        t.steps.append(step)
        if on_step:
            on_step(step, text)

    t.prosecutor_opening = prosecutor.opening_argument(code)
    emit("prosecutor_opening", t.prosecutor_opening)

    t.defense_opening = defense.opening_argument(code, t.prosecutor_opening)
    emit("defense_opening", t.defense_opening)

    t.prosecutor_rebuttal = prosecutor.rebuttal(code, t.defense_opening)
    emit("prosecutor_rebuttal", t.prosecutor_rebuttal)

    t.defense_rebuttal = defense.rebuttal(code, t.prosecutor_rebuttal)
    emit("defense_rebuttal", t.defense_rebuttal)

    t.verdict = judge.deliver_verdict(code, t.as_text())
    emit("verdict", t.verdict.raw)
    _save_for_training(t)

    return t