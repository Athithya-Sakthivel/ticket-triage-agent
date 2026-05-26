"""
DSPy programs – TriageProgram for guardrail + classification.
Compiles with BootstrapFewShot and loads from disk when possible.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

import dspy
from dspy.teleprompt import BootstrapFewShot

log = logging.getLogger("agent-service")

COMPILED_DIR = Path(os.getenv("DSPY_COMPILED_DIR", "/app/compiled"))
COMPILED_DIR.mkdir(parents=True, exist_ok=True)


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


def load_or_compile_triage(lm: dspy.LM, force: bool = False) -> TriageProgram:
    """Load compiled program from disk, or compile with a small trainset."""
    compiled_path = COMPILED_DIR / "triage_program.json"
    program = TriageProgram()
    program.classify.lm = lm   # assign the safeguard LM

    if compiled_path.exists() and not force:
        program.load(str(compiled_path))
        log.info("Loaded compiled TriageProgram from %s", compiled_path)
        return program


    optimizer = BootstrapFewShot(metric=triage_metric, max_bootstrapped_demos=3)
    compiled = optimizer.compile(program, trainset=trainset)
    compiled.save(str(compiled_path))
    log.info("Compiled TriageProgram and saved to %s", compiled_path)
    return compiled