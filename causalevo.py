"""CausalEvo: fixed-capacity exchanges, a cycle fit, and rule selection.

Self-contained implementation of the CausalEvo decision logic. Model execution
and grading are callbacks. This module does not make network requests.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import json
import random
import re
from typing import Callable, Sequence

N, K = 6, 4


@dataclass(frozen=True)
class Case:
    case_id: str
    question: str
    target: str


@dataclass(frozen=True)
class Outcome:
    text: str
    correct: int
    valid: bool = True
    finish: str = "stop"

    @property
    def reward(self) -> int:
        if self.correct not in (0, 1):
            raise ValueError("The evaluator must return binary correctness.")
        return int(self.correct) if self.valid and self.finish == "stop" else 0


@dataclass(frozen=True)
class Proposal:
    text: str
    finish: str = "stop"


Actor = Callable[[Case, list[str], int], Outcome]
Proposer = Callable[[list[str], Case, Outcome, int], Proposal]


def parse_rules(proposal: Proposal, count: int) -> list[str] | None:
    """Proposal contract: ordered RULE blocks, 5--80 words each."""
    if proposal.finish != "stop":
        return None
    chunks = re.split(r"(?im)^\s*RULE\s+([1-6])\s*:\s*", proposal.text.strip())
    if len(chunks) != 2 * count + 1 or chunks[0].strip():
        return None
    if chunks[1::2] != [str(i + 1) for i in range(count)]:
        return None
    rules = [part.strip() for part in chunks[2::2]]
    if len(set(rules)) != count or not all(5 <= len(r.split()) <= 80 for r in rules):
        return None
    return rules


def candidate_pool(memory: Sequence[str], proposed: Sequence[str]) -> list[str] | None:
    if len(memory) not in (0, K) or len(proposed) != (2 if memory else N):
        raise ValueError("Use four old plus two new rules, or six initial rules.")
    draft = list(memory) + list(proposed)
    return draft if len(set(draft)) == N else None


def design(seed: int) -> list[dict]:
    """Six directed edges; each comparison keeps three rules in common."""
    rng = random.Random(seed)
    order = list(range(N))
    rng.shuffle(order)
    case_indices = [0, 0, 1, 1, 2, 2]
    rng.shuffle(case_indices)
    edges = []
    for e in range(N):
        i, j = order[e], order[(e + 1) % N]
        background = sorted(rng.sample([a for a in range(N) if a not in (i, j)], K - 1))
        edges.append({"i": i, "j": j, "background": background,
                      "plus_ids": sorted(background + [i]),
                      "minus_ids": sorted(background + [j]),
                      "case_index": case_indices[e]})
    return edges


def fit_cycle(edges: Sequence[dict], differences: Sequence) -> dict:
    """Solve min ||d - A b||^2 subject to sum(b)=0, using exact arithmetic.

    Row e of A has +1 at i_e and -1 at j_e. On a directed cycle,
    the residual is the mean edge difference, so no matrix library is needed.
    """
    if len(edges) != N or len(differences) != N:
        raise ValueError("Exactly six edges and six differences are required.")
    if {e["i"] for e in edges} != set(range(N)) or any(
        edges[e]["j"] != edges[(e + 1) % N]["i"] for e in range(N)
    ):
        raise ValueError("Edges must form one ordered directed cycle.")
    d = [Fraction(value) for value in differences]
    residual = sum(d) / N
    b = {edges[0]["i"]: Fraction(0)}
    for e in range(N - 1):
        b[edges[e]["j"]] = b[edges[e]["i"]] - (d[e] - residual)
    center = sum(b.values()) / N
    values = [b[i] - center for i in range(N)]
    return {"values": [float(v) for v in values],
            "exact_values": [str(v) for v in values],
            "cycle_residual_mean": float(residual),
            "residual_sum_squares": float(N * residual * residual)}


def select(exact_values: Sequence, old_count: int, seed: int) -> list[int]:
    """Maximize linear credit; break exact ties by retaining old rules."""
    if len(exact_values) != N or old_count not in (0, K):
        raise ValueError("Expected six credits and either zero or four old rules.")
    values = [Fraction(value) for value in exact_values]
    order = list(range(N))
    random.Random(seed).shuffle(order)
    tie = {i: position for position, i in enumerate(order)}
    chosen = sorted(range(N), key=lambda i: (-values[i], -int(i < old_count), tie[i]))[:K]
    return sorted(chosen)


def decision(edges: Sequence[dict], differences: Sequence, old_count: int,
             seed: int, shuffled_credit: bool = False) -> dict:
    fit = fit_cycle(edges, differences)
    permutation = list(range(N))
    if shuffled_credit:
        random.Random(seed + 100000).shuffle(permutation)
    values = [fit["exact_values"][i] for i in permutation]
    return {"fit": fit, "permutation": permutation, "used_exact_values": values,
            "selected": select(values, old_count, seed)}


def run_step(memory: Sequence[str], teaching_case: Case,
             feedback_cases: Sequence[Case], actor: Actor, proposer: Proposer,
             step_index: int = 0, shuffled_credit: bool = False) -> dict:
    """One CausalEvo update. A valid proposal uses 1 teaching + 12 feedback calls.

    The actor must send only the question and rules to its model, then grade
    locally. The proposer may inspect the labeled teaching case, never the
    feedback cases. Transport exceptions propagate; they are not resampled.
    """
    if len(memory) not in (0, K) or len(set(memory)) != len(memory):
        raise ValueError("Memory must contain zero or four distinct rules.")
    if len(feedback_cases) != 3 or len({c.case_id for c in feedback_cases}) != 3:
        raise ValueError("Use three distinct feedback cases.")
    if teaching_case.case_id in {c.case_id for c in feedback_cases}:
        raise ValueError("The teaching case must be separate from feedback cases.")
    if step_index < 0:
        raise ValueError("step_index is zero-based and nonnegative.")
    before = list(memory)
    attempt = actor(teaching_case, before[:], 125000 + step_index)
    _ = attempt.reward
    proposal = proposer(before[:], teaching_case, attempt, 126000 + step_index)
    proposed = parse_rules(proposal, 2 if before else N)
    draft = candidate_pool(before, proposed) if proposed is not None else None
    record = {"step": step_index + 1, "before": before,
              "teaching_case_id": teaching_case.case_id,
              "teaching_attempt": asdict(attempt), "proposal": asdict(proposal),
              "draft": draft, "edges": [], "decision": None, "after": before[:],
              "valid_update": False, "actor_calls": 1, "proposal_calls": 1}
    if draft is not None:
        edges = design(181000 + step_index)
        jobs = [(e, side) for e in range(N) for side in ("plus", "minus")]
        random.Random(182000 + step_index).shuffle(jobs)
        observations = {}
        # Evaluate both sides with the same case and seed, in shuffled call order.
        for e, side in jobs:
            edge = edges[e]
            case = feedback_cases[edge["case_index"]]
            rules = [draft[i] for i in edge[side + "_ids"]]
            seed = 183000 + 10 * step_index + e
            outcome = actor(case, rules, seed)
            _ = outcome.reward
            observations[e, side] = outcome
            record["actor_calls"] += 1
        differences = []
        for e, edge in enumerate(edges):
            plus, minus = observations[e, "plus"], observations[e, "minus"]
            delta = plus.reward - minus.reward
            differences.append(delta)
            record["edges"].append({**edge, "case_id": feedback_cases[edge["case_index"]].case_id,
                                    "seed": 183000 + 10 * step_index + e,
                                    "plus": asdict(plus), "minus": asdict(minus), "difference": delta})
        selected = decision(edges, differences, len(before), 184000 + step_index, shuffled_credit)
        after = [draft[i] for i in selected["selected"]]
        if len(json.dumps(after, ensure_ascii=False).encode()) > 8192:
            raise ValueError("The selected state exceeds the 8,192-byte limit.")
        record.update(decision=selected, after=after, valid_update=True)
    record["changed"] = record["before"] != record["after"]
    record["retained_old_rules"] = sum(rule in record["after"] for rule in before)
    return record
