from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from deepeval.models import DeepEvalBaseLLM
from meno_bench.models import TestResult
from deepeval.metrics.g_eval import Rubric


class GEvalJudgeBase:
    def __init__(self, model: DeepEvalBaseLLM | str):
        pass

    def eval(self, case: LLMTestCase) -> dict[str, TestResult]:
        pass


class GEvalStandardJudge:
    def __init__(self, model: DeepEvalBaseLLM | str, strict: bool = False):
        self.correctness = GEval(
            name="Correctness",
            evaluation_steps=[
                "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
                "You should also heavily penalize omission of detail",
                "Vague language, or contradicting OPINIONS, or Russian, or English languages are OK",
            ],
            evaluation_params=[
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            model=model,
            strict_mode=strict,
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
            strict_mode=strict,
        )
        self.correctness2 = GEval(
            name="Correctness",
            criteria=(
                "Evaluate the factual accuracy and completeness of the actual output compared to the expected output. "
                "Pay close attention to whether the information is missing, perfectly aligned, or overly verbose."
            ),
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            rubric=[
                Rubric(score_range=(0, 1), expected_outcome="Malformed answer: Output is gibberish, contains technical errors, or is unreadable."),
                Rubric(score_range=(2, 2), expected_outcome="Entirely wrong answer: Output contradicts the truth or provides completely unrelated information."),
                Rubric(score_range=(3, 4), expected_outcome="Answer without general information: Output misses the core premise and fails to provide the primary answer."),
                Rubric(score_range=(5, 6), expected_outcome="Answer without some details: Core answer is present but significant supporting facts or steps are missing."),
                Rubric(score_range=(7, 7), expected_outcome="Answer without little details: Almost complete, but misses minor nuances, specific numbers, or secondary names."),
                Rubric(score_range=(8, 8), expected_outcome="Correct answer with too many details: Factually perfect but excessively wordy or includes irrelevant fluff that distracts from the answer."),
                Rubric(score_range=(9, 9), expected_outcome="Fully correct answer: Perfectly matches the expected output in facts and scope."),
                Rubric(score_range=(10, 10), expected_outcome="Correct answer with additional details: Factually perfect and includes helpful, relevant context beyond the expected output."),
            ]
        )
        self.clarity2 = GEval(
            name="Clarity2",
            evaluation_steps=[
                "Оцените, насколько текст написан живым и понятным языком. Избегает ли автор избыточного канцелярита и тяжелых словесных конструкций?",
                "Проверьте использование терминологии: если используются узкоспециализированные термины, дано ли им краткое пояснение?",
                "Оцените логическую последовательность: вытекает ли каждое следующее предложение из предыдущего? Нет ли резких скачков мысли?",
                "Выявите двусмысленности: могут ли фразы быть истолкованы двояко из-за неудачного порядка слов или грамматики?",
                "Текст должен быть лаконичным: удаление лишних слов ('водности') без потери смысла повышает оценку."
            ],
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
            model=model,
            strict_mode=strict,
        )

    def eval(self, case: LLMTestCase) -> dict[str, TestResult]:
        self.correctness.measure(case)
        self.clarity.measure(case)
        self.correctness2.measure(case)
        self.clarity2.measure(case)
        return {
            "correctness": TestResult(
                score=self.correctness.score, reason=self.correctness.reason
            ),
            "clarity": TestResult(score=self.clarity.score, reason=self.clarity.reason),
            "correctness2": TestResult(
                score=self.correctness2.score, reason=self.correctness2.reason
            ),
            "clarity2": TestResult(score=self.clarity2.score, reason=self.clarity2.reason),
        }
