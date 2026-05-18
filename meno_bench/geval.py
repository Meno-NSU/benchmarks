from deepeval.metrics import GEval, AnswerRelevancyMetric, BiasMetric, ToxicityMetric, NonAdviceMetric, BaseMetric
from deepeval.test_case import LLMTestCase, SingleTurnParams
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
        self.metrics: dict[str, BaseMetric] = {
            "correctness": GEval(
                name="Correctness",
                evaluation_steps=[
                    "Check whether the facts in 'actual output' contradicts any facts in 'expected output'",
                    "You should also heavily penalize omission of detail",
                    "Vague language, or contradicting OPINIONS, or Russian, or English languages are OK",
                ],
                evaluation_params=[
                    SingleTurnParams.ACTUAL_OUTPUT,
                    SingleTurnParams.EXPECTED_OUTPUT,
                ],
                model=model,
                strict_mode=strict,
            ),
            "clarity": GEval(
                name="Clarity",
                evaluation_steps=[
                    "Evaluate whether the response uses clear and direct language.",
                    "Check if the explanation avoids jargon or explains it when used.",
                    "Assess whether complex ideas are presented in a way that's easy to follow.",
                    "Identify any vague or confusing parts that reduce understanding.",
                ],
                evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
                model=model,
                strict_mode=strict,
            ),
            "correctness_ru": GEval(
                name="CorrectnessRu",
                evaluation_steps=[
                    "Сверьте факты в 'actual output' с 'expected output'. Любое прямое противоречие — повод для минимального балла.",
                    "Оцените полноту ответа: если в 'actual output' отсутствуют ключевые детали, упомянутые в эталоне, существенно снизьте оценку.",
                    "Проверьте на наличие 'галлюцинаций': добавление фактов, которых нет в эталоне и которые невозможно логически вывести из контекста.",
                    "Допускаются различия в стиле изложения или личных оценках, если они не искажают фактическую суть эталонного ответа.",
                    "Игнорируйте незначительные грамматические ошибки, если они не меняют смысл фактов.",
                ],
                evaluation_params=[
                    SingleTurnParams.ACTUAL_OUTPUT,
                    SingleTurnParams.EXPECTED_OUTPUT,
                ],
                model=model,
                strict_mode=strict,
            ),
            "clarity_ru": GEval(
                name="ClarityRu",
                evaluation_steps=[
                    "Оцените, насколько текст написан живым и понятным языком. Избегает ли автор избыточного канцелярита и тяжелых словесных конструкций?",
                    "Проверьте использование терминологии: если используются узкоспециализированные термины, дано ли им краткое пояснение?",
                    "Оцените логическую последовательность: вытекает ли каждое следующее предложение из предыдущего? Нет ли резких скачков мысли?",
                    "Выявите двусмысленности: могут ли фразы быть истолкованы двояко из-за неудачного порядка слов или грамматики?",
                    "Текст должен быть лаконичным: удаление лишних слов ('водности') без потери смысла повышает оценку.",
                ],
                evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
                model=model,
                strict_mode=strict,
            ),
            "correctness_rubrics": GEval(
                name="Correctness",
                criteria=(
                    "Evaluate the factual accuracy and completeness of the actual output compared to the expected output. "
                    "Pay close attention to whether the information is missing, perfectly aligned, or overly verbose."
                ),
                evaluation_params=[
                    SingleTurnParams.ACTUAL_OUTPUT,
                    SingleTurnParams.EXPECTED_OUTPUT,
                ],
                rubric=[
                    Rubric(
                        score_range=(0, 1),
                        expected_outcome="Malformed answer: Output is gibberish, contains technical errors, or is unreadable.",
                    ),
                    Rubric(
                        score_range=(2, 2),
                        expected_outcome="Entirely wrong answer: Output contradicts the truth or provides completely unrelated information.",
                    ),
                    Rubric(
                        score_range=(3, 4),
                        expected_outcome="Answer without general information: Output misses or omits the core facts and/or fails to provide the primary answer.",
                    ),
                    Rubric(
                        score_range=(5, 6),
                        expected_outcome="Answer without some details: Core answer is present but significant supporting facts or steps are missing.",
                    ),
                    Rubric(
                        score_range=(7, 7),
                        expected_outcome="Answer without little details: Almost complete, but misses minor nuances, specific numbers, or secondary names.",
                    ),
                    # Rubric(
                    #     score_range=(8, 8),
                    #     expected_outcome="Correct answer with too many details: Factually perfect but excessively wordy or includes irrelevant fluff that distracts from the answer.",
                    # ),
                    Rubric(
                        score_range=(8, 9),
                        expected_outcome="Fully correct answer: Perfectly matches the expected output in facts and scope.",
                    ),
                    Rubric(
                        score_range=(10, 10),
                        expected_outcome="Correct answer with additional details: Factually perfect and includes helpful, relevant context beyond the expected output.",
                    ),
                ],
                model=model,
            ),
            "relevance": AnswerRelevancyMetric(model=model),
            "bias": BiasMetric(model=model),
            "non_advice": NonAdviceMetric(model=model),
        }

    def eval(self, case: LLMTestCase) -> dict[str, TestResult]:
        out: dict[str, TestResult] = {}
        for metric_name, metric in self.metrics.items():
            metric.measure(case)
            out[metric_name] = TestResult(
                score=metric.score, reason=metric.reason
            )
        return out
