"""One-time script to compile DSPy programs from training data.

Run once before deploying the agent service. The compiled program
is saved to compiled/triage_program.json and loaded at runtime
without any LLM calls.

Usage:
    export GROQ_API_KEY="gsk_..."
    python compile_trainset.py
"""

import json
import logging
import os
import sys
from pathlib import Path

import dspy
from dspy.teleprompt import BootstrapFewShot

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
log = logging.getLogger("compile")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    log.error("GROQ_API_KEY not set")
    sys.exit(1)

COMPILED_DIR = Path("compiled")
COMPILED_DIR.mkdir(exist_ok=True)
TRAINSET_PATH = Path("data/triage_trainset.json")


class TriageSignature(dspy.Signature):
    """Classify a customer message for safety, intent, urgency, sentiment, and auto-resolvability."""
    query: str = dspy.InputField()
    safety: str = dspy.OutputField(desc="SAFE or UNSAFE")
    intent: str = dspy.OutputField()
    urgency: int = dspy.OutputField(desc="1-10")
    sentiment: str = dspy.OutputField(desc="angry, frustrated, confused, neutral, satisfied")
    auto_resolvable: bool = dspy.OutputField()


class TriageProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought(TriageSignature)

    def forward(self, query: str):
        result = self.classify(query=query)
        # Return a dspy.Prediction so the metric can access attributes
        return dspy.Prediction(
            safety=result.safety,
            intent=result.intent,
            urgency=int(result.urgency),
            sentiment=result.sentiment,
            auto_resolvable=bool(result.auto_resolvable),
        )


def triage_metric(example, pred, trace=None):
    """Weighted accuracy metric. pred is a dspy.Prediction."""
    score = 0.0
    if pred.safety == example.safety:
        score += 0.3
    if pred.intent == example.intent:
        score += 0.3
    if abs(pred.urgency - example.urgency) <= 1:
        score += 0.2
    if pred.sentiment == example.sentiment:
        score += 0.1
    if pred.auto_resolvable == example.auto_resolvable:
        score += 0.1
    return score


def main():
    log.info("Loading trainset from %s", TRAINSET_PATH)
    with open(TRAINSET_PATH) as f:
        data = json.load(f)

    trainset = []
    for item in data:
        trainset.append(
            dspy.Example(
                query=item["query"],
                safety=item["safety"],
                intent=item["intent"],
                urgency=item["urgency"],
                sentiment=item["sentiment"],
                auto_resolvable=item["auto_resolvable"],
            ).with_inputs("query")
        )

    log.info("Loaded %d training examples", len(trainset))

    # LiteLLM respects this env var for rate limiting
    os.environ["LITELLM_RATE_LIMIT"] = "5"  # Max 5 concurrent requests

    lm = dspy.LM(
        model="groq/openai/gpt-oss-safeguard-20b",
        api_key=GROQ_API_KEY,
        temperature=0.0,
        max_tokens=512,
    )

    # Configure the default LM for the optimizer
    dspy.configure(lm=lm)

    program = TriageProgram()

    log.info("Compiling with BootstrapFewShot (this may take a few minutes)...")
    optimizer = BootstrapFewShot(
        metric=triage_metric,
        max_bootstrapped_demos=4,
        max_labeled_demos=16,
        max_errors=10,  # Tolerate rate-limit errors on some examples
    )
    compiled = optimizer.compile(program, trainset=trainset)

    output_path = COMPILED_DIR / "triage_program.json"
    compiled.save(str(output_path))
    log.info("Saved compiled program to %s", output_path)
    log.info("Done.")


if __name__ == "__main__":
    main()