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
    def __init__(self, model: DeepEvalBaseLLM | str, strict: bool = False):
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
            name="Correctness2",
            evaluation_steps=[
                "Сверьте факты в 'actual output' с 'expected output'. Любое прямое противоречие — повод для минимального балла.",
                "Оцените полноту ответа: если в 'actual output' отсутствуют ключевые детали, упомянутые в эталоне, существенно снизьте оценку.",
                "Проверьте на наличие 'галлюцинаций': добавление фактов, которых нет в эталоне и которые невозможно логически вывести из контекста.",
                "Допускаются различия в стиле изложения или личных оценках, если они не искажают фактическую суть эталонного ответа.",
                "Игнорируйте незначительные грамматические ошибки, если они не меняют смысл фактов."
            ],
            evaluation_params=[
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            model=model,
            strict_mode=strict,
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
