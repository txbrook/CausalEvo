"""A deterministic synthetic execution example; no LLM or patient data."""
import argparse
import json
from pathlib import Path

from causalevo import Case, Outcome, Proposal, run_step


RULES = [
    "For a toy task, apply reusable procedure alpha before producing an answer.",
    "For a toy task, apply reusable procedure beta before producing an answer.",
    "For a toy task, apply reusable procedure gamma before producing an answer.",
    "For a toy task, apply reusable procedure delta before producing an answer.",
    "For a toy task, apply reusable procedure epsilon before producing an answer.",
    "For a toy task, apply reusable procedure zeta before producing an answer.",
    "For a toy task, apply reusable procedure eta before producing an answer.",
    "For a toy task, apply reusable procedure theta before producing an answer.",
]
WEIGHTS = dict(zip(RULES, [3, 2, 1, 0, -1, -2, 5, -4]))


def actor(case, memory, seed):
    # This simulator exposes known utilities only to the synthetic environment.
    # The CausalEvo selector observes binary correctness, never WEIGHTS.
    threshold = int(case.question)
    score = sum(WEIGHTS[rule] for rule in memory)
    answer = str(int(score >= threshold))
    return Outcome(text=f"Toy environment output: {answer}", correct=int(answer == case.target))


def proposer(memory, teaching_case, attempt, seed):
    # Stand-in for a model proposal; these are not learned medical procedures.
    additions = RULES[6:] if memory else RULES[:6]
    return Proposal("\n".join(f"RULE {i + 1}: {rule}" for i, rule in enumerate(additions)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="Optional path for the complete synthetic trace")
    args = parser.parse_args()
    memory, history = [], []
    for t in range(2):
        teaching = Case(f"teaching-{t}", "6", "1")
        feedback = [Case(f"feedback-{t}-{j}", str(threshold), "1")
                    for j, threshold in enumerate([5, 6, 7])]
        result = run_step(memory, teaching, feedback, actor, proposer, step_index=t)
        memory = result["after"]
        history.append(result)
        print(f"Step {t + 1}: selected={result['decision']['selected']}, "
              f"changed={result['changed']}, actor_calls={result['actor_calls']}")
    output = {"kind": "synthetic_offline_demo", "api_calls": 0,
              "empirical_llm_result": False, "history": history, "final_memory": memory}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output, indent=2) + "\n")
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
