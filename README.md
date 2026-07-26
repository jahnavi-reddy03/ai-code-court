# AI Code Court ⚖️

Paste your bad code. Get dragged in front of a jury of language models.

- **Prosecutor** (GPT-4) — reads your code and tears it apart, line by line, no mercy.
- **Defense Attorney** (Claude) — has to defend every questionable decision you made, even the `except: pass`.
- **Judge** (fine-tuned Llama 3 8B) — listens to both sides and hands down a verdict, sentence, and one-liner roast.

Live demo: (add Streamlit link once deployed)

## Why this exists

Code review tools tell you what's wrong. They don't make it *entertaining*. This turns a review into
a debate with a beginning, middle, and end — and a judge who's been fine-tuned specifically on what
a good/bad verdict sounds like, so it's not just "GPT-4 pretending to be a judge."

## Architecture

```
User pastes code
      │
      ▼
┌─────────────────┐
│   courtroom.py   │  orchestrates the whole trial
└─────────────────┘
      │
      ├──► Prosecutor (OpenAI GPT-4)   — opening argument
      ├──► Defense (Anthropic Claude)  — opening argument
      ├──► Prosecutor rebuttal
      ├──► Defense rebuttal
      └──► Judge (fine-tuned Llama 3 8B, via HF Inference Endpoint or local) — verdict
      │
      ▼
Streamlit renders it as a live transcript
```

Each role is its own module with its own system prompt and its own model client, so swapping any one
model out later (e.g. GPT-4 → GPT-4o, or hosting Llama differently) doesn't touch the other two.

## Project layout

```
ai-code-court/
├── app.py                  # Streamlit UI — the courtroom itself
├── core/
│   ├── prosecutor.py       # GPT-4 role
│   ├── defense.py          # Claude role
│   ├── judge.py            # Fine-tuned Llama 3 8B role
│   └── courtroom.py        # runs the trial, manages turn order + transcript
├── utils/
│   └── prompts.py          # all system prompts live here, not scattered in code
├── finetuning/
│   ├── prepare_dataset.py  # builds QLoRA training set from verdict examples
│   ├── train_qlora.py      # Colab-ready QLoRA fine-tune script
│   └── dataset/
│       └── judge_verdicts_sample.jsonl
├── tests/
│   └── test_courtroom.py
├── .env.example
├── requirements.txt
└── .gitignore
```

## Setup

```bash
git clone https://github.com/jahnavi-reddy03/ai-code-court.git
cd ai-code-court
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your keys, this file is gitignored
streamlit run app.py
```

## Fine-tuning the judge

The judge isn't just a system-prompted Llama — it's QLoRA fine-tuned on a small dataset of
code-review verdicts (code snippet + prosecutor argument + defense argument → verdict) so it
develops a consistent voice: authoritative, funny, decisive, and willing to actually pick a side
instead of hedging like a base model does.

See `finetuning/` for the dataset prep and training script (built to run on a free/paid Colab GPU).
Until the fine-tuned adapter is trained and hosted, `core/judge.py` falls back to a heavily
prompted base Llama so the app runs end-to-end from day one.

## Roadmap

- [ ] v0: all three roles working with prompted (non-fine-tuned) models, basic Streamlit transcript UI
- [ ] v1: collect/curate verdict dataset, run QLoRA fine-tune on Colab
- [ ] v2: swap judge.py to call the fine-tuned adapter (HF Inference Endpoint)
- [ ] v3: polish UI — typing animation, gavel sound, shareable verdict card for LinkedIn
- [ ] v4: "verdict card" auto-generated as an image for social sharing
