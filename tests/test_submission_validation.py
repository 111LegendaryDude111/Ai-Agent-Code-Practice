from __future__ import annotations

import unittest

from interview_bot.main import MAX_SUBMISSION_LENGTH, _validate_submission_text


class SubmissionValidationTests(unittest.TestCase):
    def test_rejects_none_message(self) -> None:
        submission, error = _validate_submission_text(None)

        self.assertIsNone(submission)
        self.assertEqual(
            error,
            "Отправьте код в текстовом сообщении.",
        )

    def test_rejects_empty_message(self) -> None:
        submission, error = _validate_submission_text("   \n\t   ")

        self.assertIsNone(submission)
        self.assertEqual(
            error,
            "Пустой ввод. Отправьте непустой фрагмент кода.",
        )

    def test_rejects_too_large_message(self) -> None:
        oversized_payload = "x" * (MAX_SUBMISSION_LENGTH + 1)
        submission, error = _validate_submission_text(oversized_payload)

        self.assertIsNone(submission)
        self.assertIsNotNone(error)
        if error is None:
            self.fail("Expected length validation error.")

        self.assertIn(str(MAX_SUBMISSION_LENGTH), error)

    def test_accepts_valid_message(self) -> None:
        submission, error = _validate_submission_text("  print('ok')  ")

        self.assertEqual(submission, "print('ok')")
        self.assertIsNone(error)


if __name__ == "__main__":
    unittest.main()
