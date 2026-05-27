"""
DSPy programs – TriageProgram for guardrail + classification.
Compiles with BootstrapFewShot and loads from disk when possible.
Only the classifier is compiled; the resolver uses a system prompt.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import dspy
from dspy.teleprompt import BootstrapFewShot

log = logging.getLogger("agent-service")

COMPILED_DIR = Path(os.getenv("DSPY_COMPILED_DIR", "compiled"))
COMPILED_DIR.mkdir(parents=True, exist_ok=True)
TRAINSET_PATH = Path(os.getenv("DSPY_TRAINSET_PATH", "data/triage_trainset.json"))


class TriageSignature(dspy.Signature):
    """Classify a customer message for safety, intent, urgency, sentiment, and auto-resolvability."""
    query: str = dspy.InputField()
    safety: str = dspy.OutputField(desc="SAFE or UNSAFE")
    intent: str = dspy.OutputField(desc="wrong_item_delivered, damaged_product, late_delivery, refund_status, cancellation_request, return_request, warranty_claim, payment_issue, account_issue, general_inquiry, complaint")
    urgency: int = dspy.OutputField(desc="1-10, 10 = financial harm, safety risk, legal threat")
    sentiment: str = dspy.OutputField(desc="angry, frustrated, confused, neutral, satisfied")
    auto_resolvable: bool = dspy.OutputField(desc="False if urgency>=8, amount>10000, legal threat, or customer demands human")


class TriageProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought(TriageSignature)

    def forward(self, query: str) -> dict:
        result = self.classify(query=query)
        return {
            "safety": result.safety,
            "intent": result.intent,
            "urgency": int(result.urgency),
            "sentiment": result.sentiment,
            "auto_resolvable": bool(result.auto_resolvable),
        }


def triage_metric(example, pred, trace=None) -> float:
    score = 0.0
    if pred.safety == example.safety: score += 0.3
    if pred.intent == example.intent: score += 0.3
    if abs(pred.urgency - example.urgency) <= 1: score += 0.2
    if pred.sentiment == example.sentiment: score += 0.1
    if pred.auto_resolvable == example.auto_resolvable: score += 0.1
    return score


def load_trainset(path: Path | None = None) -> list[dspy.Example]:
    path = path or TRAINSET_PATH
    if not path.exists():
        log.warning("Trainset file not found: %s — using inline fallback", path)
        return _fallback_trainset()

    with open(path) as f:
        data = json.load(f)

    examples = []
    for item in data:
        example = dspy.Example(
            query=item["query"],
            safety=item["safety"],
            intent=item["intent"],
            urgency=item["urgency"],
            sentiment=item["sentiment"],
            auto_resolvable=item["auto_resolvable"],
        ).with_inputs("query")
        examples.append(example)

    log.info("Loaded %d examples from %s", len(examples), path)
    return examples


def _fallback_trainset() -> list[dspy.Example]:
    return [
        dspy.Example(query="I want to return a damaged phone", safety="SAFE", intent="damaged_product", urgency=5, sentiment="frustrated", auto_resolvable=True).with_inputs("query"),
        dspy.Example(query="You people are useless, I'll sue you", safety="UNSAFE", intent="complaint", urgency=10, sentiment="angry", auto_resolvable=False).with_inputs("query"),
        dspy.Example(query="Where is my refund? It's been 15 days.", safety="SAFE", intent="refund_status", urgency=8, sentiment="angry", auto_resolvable=False).with_inputs("query"),
    ]


def load_or_compile_triage(lm: dspy.LM, force: bool = False) -> TriageProgram:
    """Load a compiled TriageProgram or compile a new one.

    Sets the global default LM via dspy.configure so that the loaded program works.
    """
    compiled_path = COMPILED_DIR / "triage_program.json"

    # Set the global default LM - this is what the compiled program expects
    dspy.configure(lm=lm)

    program = TriageProgram()

    if compiled_path.exists() and not force:
        program.load(str(compiled_path))
        log.info("Loaded compiled TriageProgram from %s", compiled_path)
        return program

    if not compiled_path.exists() or force:
        trainset = load_trainset()
        log.info("Compiling TriageProgram with %d examples...", len(trainset))
        optimizer = BootstrapFewShot(metric=triage_metric, max_bootstrapped_demos=4, max_labeled_demos=16)
        compiled = optimizer.compile(program, trainset=trainset)
        compiled.save(str(compiled_path))
        log.info("TriageProgram compiled and saved to %s", compiled_path)
        return compiled

    return program