# AI Code Court ⚖️

Paste your bad code. Get dragged in front of a jury of language models.

- **Prosecutor** (GPT-4o) — reads your code and tears it apart, line by line, no mercy.
- **Defense Attorney** (Gemini) — has to defend every questionable decision you made, even the `except: pass`.
- **Judge** (Llama 3.1 8B, served via Groq) — listens to both sides and hands down a verdict, sentence, and one-liner roast.

Live demo: (add Streamlit link once deployed)

## Why this exists

Code review tools tell you what's wrong. They don't make it *entertaining*. This turns a review into a debate with a beginning, middle, and end.

## Architecture

User pastes code
│
▼
┌─────────────────┐
│ courtroom.py │ orchestrates the whole trial
└─────────────────┘
│
├──► Prosecutor (OpenAI GPT-4o) — opening argument
├──► Defense (Google Gemini) — opening argument
├──► Prosecutor rebuttal
├──► Defense rebuttal
└──► Judge (Groq / Llama 3.1 8B) — verdict
│
▼
Streamlit renders it as a live transcript


Each role is its own module with its own system prompt and its own model client, so swapping any one model out later doesn't touch the other two — the defense role actually started as Claude and moved to Gemini partway through building this, and the swap only took a few minutes because of that separation.

## Project layout

ai-code-court/
├── app.py # Streamlit UI — the courtroom itself
├── core/
│ ├── prosecutor.py # GPT-4o role
│ ├── defense.py # Gemini role
│ ├── judge.py # Groq / Llama 3.1 8B role
│ └── courtroom.py # runs the trial, manages turn order + transcript
├── utils/
│ └── prompts.py # all system prompts live here, not scattered in code
├── finetuning/
│ ├── prepare_dataset.py # builds QLoRA training set from verdict examples
│ ├── train_qlora.py # Colab QLoRA fine-tune script, actually run and working
│ └── dataset/
│ ├── judge_verdicts_sample.jsonl
│ └── collected_verdicts.jsonl # 50 real examples auto-collected from live trials
├── tests/
│ └── test_courtroom.py
├── .env.example
├── requirements.txt
└── .gitignore


## Setup

**Windows:**
```cmd
git clone https://github.com/jahnavi-reddy03/ai-code-court.git
cd ai-code-court
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
copy .env.example .env
streamlit run app.py
```

**Mac/Linux:**
```bash
git clone https://github.com/jahnavi-reddy03/ai-code-court.git
cd ai-code-court
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

You'll need three keys in your `.env`: `OPENAI_API_KEY`, `GEMINI_API_KEY`, and `GROQ_API_KEY`.

## The judge, actually fine-tuned

The judge isn't just a system-prompted model — a real QLoRA fine-tune was trained on 50 courtroom transcripts collected live from this exact app (code snippet + prosecutor argument + defense argument → verdict), so it learned to sound consistently deadpan and decisive.

**The adapter is trained, tested, and published:**
[huggingface.co/jahnavi0803/llama3-8b-code-judge](https://huggingface.co/jahnavi0803/llama3-8b-code-judge)

Training results: loss dropped from 0.71 to 0.62 over 3 epochs, mean token accuracy reached 87%. Confirmed working — a real generated example is on the model card.

**Where it currently runs:** the live app above uses the *base* Llama 3.1 8B via Groq (chosen for speed and reliability). The fine-tuned adapter needs paid GPU hosting to serve live, which is a separate infrastructure decision from training — so for now it's published and fully documented, and anyone can load and run it themselves using the code on the model page.

Full training and dataset-prep code is in `finetuning/`.