"""Offline checks of mathematical properties and update behavior."""
from fractions import Fraction
from itertools import combinations
import unittest

from causalevo import Case, Outcome, Proposal, candidate_pool, design, fit_cycle, parse_rules, run_step, select


OLD = [f"Procedure {i}: preserve this reusable rule when it remains useful." for i in range(4)]
NEW = ["Procedure four: check applicable units before performing the requested calculation.",
       "Procedure five: preserve precision until the final reported numeric answer."]


class CausalEvoChecks(unittest.TestCase):
    def test_paired_design_has_equal_capacity_and_balanced_cases(self):
        for seed in range(20):
            edges = design(seed)
            for e in edges:
                self.assertEqual(len(e['plus_ids']), 4)
                self.assertEqual(len(e['minus_ids']), 4)
                self.assertEqual(set(e['plus_ids']) & set(e['minus_ids']), set(e['background']))
            self.assertEqual(sorted(e['case_index'] for e in edges), [0, 0, 1, 1, 2, 2])

    def test_cycle_recovers_centered_additive_effects(self):
        beta = [Fraction(x, 10) for x in [1, -2, 7, -5, 3, 6]]
        edges = design(11)
        d = [beta[e['i']] - beta[e['j']] for e in edges]
        fitted = list(map(Fraction, fit_cycle(edges, d)['exact_values']))
        self.assertEqual(fitted, [x - sum(beta) / 6 for x in beta])

    def test_cycle_fit_satisfies_least_squares_normal_equations(self):
        edges, d = design(13), [1, 0, -1, 1, 1, 0]
        fitted = list(map(Fraction, fit_cycle(edges, d)['exact_values']))
        residuals = [Fraction(delta) - (fitted[e['i']] - fitted[e['j']]) for e, delta in zip(edges, d)]
        self.assertEqual(sum(fitted), 0)
        self.assertEqual(residuals, [Fraction(sum(d), 6)] * 6)

    def test_exact_ties_retain_incumbent(self):
        self.assertEqual(select(['0'] * 6, 4, 9), [0, 1, 2, 3])
        self.assertEqual(candidate_pool(OLD, NEW)[:4], OLD)
        self.assertIsNone(candidate_pool(OLD, [OLD[0], NEW[0]]))

    def test_truncated_proposal_is_a_noop_without_feedback_calls(self):
        calls = []
        def actor(case, memory, seed):
            calls.append(case.case_id)
            return Outcome('FINAL: 1', 1)
        def propose(memory, case, attempt, seed):
            return Proposal('RULE 1: an incomplete output', finish='length')
        feedback = [Case(str(i), 'question', '1') for i in range(3)]
        result = run_step(OLD, Case('teaching', 'question', '1'), feedback, actor, propose)
        self.assertEqual(result['after'], OLD)
        self.assertFalse(result['valid_update'])
        self.assertEqual(calls, ['teaching'])

    def test_invalid_actor_output_counts_as_zero(self):
        self.assertEqual(Outcome('broken', 1, valid=False).reward, 0)
        self.assertEqual(Outcome('partial', 1, finish='length').reward, 0)

    def test_valid_proposal_contract(self):
        text = '\n'.join(f'RULE {i + 1}: {s}' for i, s in enumerate(NEW))
        self.assertEqual(parse_rules(Proposal(text), 2), NEW)
        self.assertIsNone(parse_rules(Proposal('Preface\n' + text), 2))

    def test_complete_step_pairs_calls_and_retains_useful_rules(self):
        calls, proposal_calls = [], []
        memory = OLD[:]

        def actor(case, rules, seed):
            calls.append((case.case_id, tuple(rules), seed))
            # One new rule has a known positive effect in this synthetic task.
            return Outcome('synthetic answer', int(NEW[0] in rules))

        def propose(rules, case, attempt, seed):
            proposal_calls.append((rules, case.case_id, attempt.correct))
            return Proposal('\n'.join(f'RULE {i + 1}: {s}' for i, s in enumerate(NEW)))

        feedback = [Case(f'feedback-{i}', 'question', '1') for i in range(3)]
        result = run_step(memory, Case('teaching', 'question', '1'), feedback, actor, propose)

        self.assertEqual(memory, OLD)
        self.assertEqual(proposal_calls, [(OLD, 'teaching', 0)])
        self.assertEqual(result['actor_calls'], 13)
        self.assertEqual(result['proposal_calls'], 1)
        self.assertTrue(result['valid_update'])
        self.assertIn(NEW[0], result['after'])
        self.assertEqual(len(set(result['after'])), 4)
        self.assertEqual(result['retained_old_rules'], 3)
        for case in feedback:
            self.assertEqual(sum(c[0] == case.case_id for c in calls[1:]), 4)
        for seed in {c[2] for c in calls[1:]}:
            pair = [c for c in calls[1:] if c[2] == seed]
            self.assertEqual(len(pair), 2)
            self.assertEqual(pair[0][0], pair[1][0])
            self.assertEqual(len(set(pair[0][1]) & set(pair[1][1])), 3)

    def test_projection_is_not_a_global_combinatorial_guarantee(self):
        # Exact background averaging still cannot recover arbitrary interactions.
        value = {s: Fraction(0) for s in combinations(range(6), 4)}
        value[(0, 1, 2, 3)] = Fraction(9, 10)
        value[(0, 1, 4, 5)] = Fraction(6, 10)
        value[(0, 2, 4, 5)] = Fraction(4, 10)
        edges, d = design(7), []
        for e in edges:
            backgrounds = list(combinations([i for i in range(6) if i not in (e['i'], e['j'])], 3))
            contrasts = [value[tuple(sorted(bg + (e['i'],)))] - value[tuple(sorted(bg + (e['j'],)))] for bg in backgrounds]
            d.append(sum(contrasts) / len(contrasts))
        fitted = fit_cycle(edges, d)
        chosen = tuple(select(fitted['exact_values'], 0, 17))
        self.assertEqual(value[chosen], 0)
        self.assertEqual(max(value.values()), Fraction(9, 10))


if __name__ == '__main__':
    unittest.main()
