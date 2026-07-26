"""
Turns finetuning/dataset/judge_verdicts_sample.jsonl (raw code + arguments + verdict)
into the prompt/completion format used by train_qlora.py.

Grow the sample file with more real examples before training - 3 rows is a schema
demo, not a training set. Aim for 150-300+ examples for a QLoRA fine-tune that
actually shifts the model's voice. You can bootstrap more examples by running the
prompted (non-fine-tuned) courtroom pipeline itself and hand-editing the outputs
you like best.
"""

import json
from pathlib import Path

from utils.prompts import JUDGE_SYSTEM

SRC = Path(__file__).parent / "dataset" / "collected_verdicts.jsonl"
OUT = Path(__file__).parent / "dataset" / "judge_train.jsonl"


def build_prompt(row: dict) -> str:
    return (
        f"{JUDGE_SYSTEM}\n\n"
        f"The code on trial:\n```\n{row['code']}\n```\n\n"
        f"Courtroom transcript:\n"
        f"PROSECUTOR: {row['prosecutor']}\n\n"
        f"DEFENSE: {row['defense']}\n\n"
        "Deliver your verdict now, using the exact format specified."
    )


def main():
    rows = [json.loads(line) for line in SRC.read_text().splitlines() if line.strip()]
    with OUT.open("w") as f:
        for row in rows:
            record = {"prompt": build_prompt(row), "completion": row["verdict"]}
            f.write(json.dumps(record) + "\n")
    print(f"Wrote {len(rows)} training examples to {OUT}")


if __name__ == "__main__":
    main()
