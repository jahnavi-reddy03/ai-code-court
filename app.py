import streamlit as st
from dotenv import load_dotenv
from core.courtroom import run_trial

load_dotenv()

st.set_page_config(page_title="AI Code Court ⚖️", page_icon="⚖️", layout="centered")

ROLE_STYLE = {
    "prosecutor_opening": ("🔴 Prosecutor (GPT-4)", "opening argument"),
    "defense_opening": ("🔵 Defense (Gemini)", "opening argument"),
    "prosecutor_rebuttal": ("🔴 Prosecutor (GPT-4)", "rebuttal"),
    "defense_rebuttal": ("🔵 Defense (Gemini)", "rebuttal"),
}

st.title("⚖️ AI Code Court")
st.caption("Paste your code. GPT-4 prosecutes it, Gemini defends it, a fine-tuned Llama 3 judges it.")

code = st.text_area(
    "Exhibit A — your code",
    height=280,
    placeholder="def do_thing(x):\n    try:\n        return x.process()\n    except:\n        pass",
)

if st.button("Put it on trial", type="primary", disabled=not code.strip()):
    transcript_area = st.container()
    verdict_area = st.empty()

    def on_step(step, text):
        if step == "verdict":
            return  # rendered separately below
        label, phase = ROLE_STYLE[step]
        with transcript_area:
            st.markdown(f"**{label}** · _{phase}_")
            st.write(text)
            st.divider()

    with st.spinner("Court is now in session..."):
        trial = run_trial(code, on_step=on_step)

    v = trial.verdict
    with verdict_area.container():
        st.subheader("🔨 Verdict")
        st.markdown(f"**{v.verdict}**")
        st.write(v.reasoning)
        st.markdown(f"**Sentence:** {v.sentence}")
        st.info(f"\u201c{v.one_liner}\u201d")

    st.caption("Screenshot the verdict above, it's built to be shared.")
