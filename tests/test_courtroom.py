"""
Mocks all three model calls so tests run without API keys or network access.
Run with: pytest tests/
"""

from unittest.mock import patch
from core import courtroom, judge


SAMPLE_CODE = "def f(x):\n    try:\n        return 1/x\n    except:\n        pass"

FAKE_VERDICT_RAW = (
    "VERDICT: Guilty of Bad Code\n"
    "REASONING: Bare except swallows real errors.\n"
    "SENTENCE: Refactor Before Merge\n"
    "ONE-LINER: \"Silence isn't golden when it's hiding a ZeroDivisionError.\""
)


@patch("core.judge.deliver_verdict")
@patch("core.defense.rebuttal", return_value="Defense rebuttal text")
@patch("core.prosecutor.rebuttal", return_value="Prosecutor rebuttal text")
@patch("core.defense.opening_argument", return_value="Defense opening text")
@patch("core.prosecutor.opening_argument", return_value="Prosecutor opening text")
def test_run_trial_full_flow(mock_p_open, mock_d_open, mock_p_reb, mock_d_reb, mock_verdict):
    mock_verdict.return_value = judge._parse(FAKE_VERDICT_RAW)

    steps_seen = []
    result = courtroom.run_trial(SAMPLE_CODE, on_step=lambda step, text: steps_seen.append(step))

    assert result.prosecutor_opening == "Prosecutor opening text"
    assert result.defense_rebuttal == "Defense rebuttal text"
    assert result.verdict.verdict == "Guilty of Bad Code"
    assert result.verdict.sentence == "Refactor Before Merge"
    assert steps_seen == [
        "prosecutor_opening", "defense_opening",
        "prosecutor_rebuttal", "defense_rebuttal", "verdict",
    ]


def test_judge_parse_handles_full_format():
    v = judge._parse(FAKE_VERDICT_RAW)
    assert v.verdict == "Guilty of Bad Code"
    assert "Bare except" in v.reasoning
    assert v.sentence == "Refactor Before Merge"
    assert "ZeroDivisionError" in v.one_liner
