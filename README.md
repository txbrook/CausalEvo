# CausalEvo

This repository provides the core implementation for **[CausalEvo: Causal Skill Self-Evolution for Health LLM Agents](https://arxiv.org/abs/2610.21016)**, by Yuchen Zhang, accepted at the GenAI for Health Workshop at NeurIPS 2026.

[Project page](https://txbrook.github.io/CausalEvo/) · [Paper](https://arxiv.org/abs/2610.21016)

CausalEvo maintains four textual rules. Each evolution step pools the incumbent rules with two proposals, compares four-rule subsets through six paired exchanges, and fits relative rule values by least squares. It retains the four highest-valued rules, favoring incumbents on exact ties. The first step proposes six rules from an empty state.

## Run

Python 3.10 or newer; standard library only.

```sh
git clone https://github.com/txbrook/CausalEvo.git
cd CausalEvo
python3 demo.py
python3 -m unittest -v test_causalevo
```

The demo runs two evolution steps in a deterministic synthetic environment, with no API calls. To save its trace, run `python3 demo.py --out trace.json`.

## Core

`causalevo.py` provides:

- `design`: a randomly relabeled six-rule cycle with three shared background rules per exchange. Each of three feedback cases serves two edges.
- `fit_cycle`: exact, zero-sum least-squares credits using rational arithmetic.
- `select`: top-four selection with incumbent preference on ties.
- `run_step`: teaching, proposal validation, paired evaluation, and skill update. Set `shuffled_credit=True` for the credit-assignment control.

A valid step uses one teaching actor call, one proposal call, and twelve feedback actor calls. An invalid or duplicate proposal retains the current state and skips feedback calls.

## Connect a model

Supply two callbacks to `run_step`:

```python
actor(case, rules, seed) -> Outcome
proposer(rules, teaching_case, teaching_attempt, seed) -> Proposal
```

The actor sends `case.question` and the supplied rules to the model, then grades the response locally against `case.target`. The reference target is for grading and must not enter the actor's model prompt. Return binary correctness through `Outcome`; invalid or non-normal completions receive zero reward. Handle model transport failures in the callback if they should count as zero; uncaught exceptions stop the step.

The proposer receives only the labeled teaching case, its actor response, and the current rules. Return numbered `RULE 1: ...` blocks: six rules for an empty state or two new rules otherwise, each 5–80 whitespace-separated words. The feedback cases are reserved for paired scoring.

Pass one teaching case and three distinct feedback cases to each step. Reuse `result["after"]` as the next skill state. `result` also contains observations, fitted credits, selected indices, and callback counts. The controller uses the same case and seed for both sides of each exchange; model execution and task-specific grading are supplied by the callbacks.
