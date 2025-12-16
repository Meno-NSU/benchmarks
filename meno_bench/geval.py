from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from deepeval.models import DeepEvalBaseLLM
from meno_bench.models import TestResult


class GEvalJudgeBase:
    def __init__(self, model: DeepEvalBaseLLM | str):
        pass

    def eval(self, case: LLMTestCase) -> dict[str, TestResult]:
        pass


class GEvalStandardJudge:
    def __init__(self, model: DeepEvalBaseLLM | str):
        self.correctness = GEval(
            name="Correctness",
            evaluation_steps=[
                "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
                "You should also heavily penalize omission of detail",
                "Vague language, or contradicting OPINIONS, are OK",
            ],
            evaluation_params=[
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            model=model,
        )
        self.clarity = GEval(
            name="Clarity",
            evaluation_steps=[
                "Evaluate whether the response uses clear and direct language.",
                "Check if the explanation avoids jargon or explains it when used.",
                "Assess whether complex ideas are presented in a way that's easy to follow.",
                "Identify any vague or confusing parts that reduce understanding.",
            ],
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            model=model,
        )

    def eval(self, case: LLMTestCase) -> dict[str, TestResult]:
        self.correctness.measure(case)
        self.clarity.measure(case)
        return {
            "correctness": TestResult(
                score=self.correctness.score, reason=self.correctness.reason
            ),
            "clarity": TestResult(score=self.clarity.score, reason=self.clarity.reason),
        }
