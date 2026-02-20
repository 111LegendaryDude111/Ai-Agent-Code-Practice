from __future__ import annotations

import unittest

from interview_orchestrator.evaluation_dataset import (
    build_offline_dataset,
    check_scoring_consistency,
    evaluate_offline_dataset,
    run_offline_dataset_consistency_check,
)


class OfflineDatasetEvaluationTests(unittest.TestCase):
    def test_build_offline_dataset_includes_required_solution_types(self) -> None:
        samples = build_offline_dataset()
        sample_ids = {sample.sample_id for sample in samples}
        self.assertSetEqual(
            sample_ids,
            {
                "correct_solution",
                "inefficient_solution",
                "security_bad_solution",
            },
        )

    def test_evaluate_offline_dataset_is_deterministic(self) -> None:
        first = evaluate_offline_dataset()
        second = evaluate_offline_dataset()
        self.assertEqual(first, second)

    def test_evaluate_offline_dataset_scores_follow_expected_order(self) -> None:
        results = evaluate_offline_dataset()
        indexed = {result.sample_id: result for result in results}

        correct = indexed["correct_solution"]
        inefficient = indexed["inefficient_solution"]
        security_bad = indexed["security_bad_solution"]

        self.assertGreater(correct.final_score, inefficient.final_score)
        self.assertGreater(inefficient.final_score, security_bad.final_score)
        self.assertLess(inefficient.performance_score, correct.performance_score)
        self.assertLess(security_bad.security_score, correct.security_score)
        self.assertLessEqual(security_bad.security_score, 20.0)

        self.assertAlmostEqual(correct.final_score, 99.61, places=2)
        self.assertAlmostEqual(inefficient.final_score, 91.03, places=2)
        self.assertAlmostEqual(security_bad.final_score, 89.03, places=2)

    def test_check_scoring_consistency_returns_true_for_builtin_results(self) -> None:
        results = evaluate_offline_dataset()
        self.assertTrue(check_scoring_consistency(results))

    def test_run_offline_dataset_consistency_check_returns_true(self) -> None:
        self.assertTrue(run_offline_dataset_consistency_check())


if __name__ == "__main__":
    unittest.main()
