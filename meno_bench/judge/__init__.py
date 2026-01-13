from meno_bench.settings import JudgeSettings, GoogleJudgeSettings, OpenAIJudgeSettings, GigaSettings
from meno_bench.models import TestCasesFileFull, TestOut, TestMetricsResults
from meno_bench.judge.summary import get_summary
import json
import tqdm
import traceback


def judge_cases(settings: JudgeSettings, test_cases: TestCasesFileFull) -> list[TestOut]:
    match settings:
        case GoogleJudgeSettings():
            # import here for better performance
            from meno_bench.judge.google import get_model as get_google_model
            model = get_google_model(settings)
        case OpenAIJudgeSettings():
            # import here for better performance
            from meno_bench.judge.openai_api import get_model as get_api_model
            model = get_api_model(settings)
        case GigaSettings():
            from meno_bench.judge.gig import get_model as get_api_model
            model = get_api_model(settings)
        case _:
            raise Exception("Unknown judge settings")
    # import here for better performance in inference as cli tool
    from meno_bench.geval import GEvalStandardJudge
    from deepeval.test_case import LLMTestCase
    geval = GEvalStandardJudge(model)

    results: list[TestOut] = []
    try:
        for case in tqdm.tqdm(test_cases):
            llm_test_case = LLMTestCase(
                input=case["question"],
                actual_output=case["model_answer"],
                expected_output=case["ground_truth"],
            )

            results.append(
                TestOut(
                    result=TestMetricsResults(**geval.eval(llm_test_case)),
                    case=case,
                )
            )
    except Exception as e:
        print(e)
        print(traceback.print_exc())
    return results


def judge(settings: OpenAIJudgeSettings | GoogleJudgeSettings):
    with open(settings.file, "r") as f:
        test_cases: TestCasesFileFull = json.load(f)
    
    results = judge_cases(settings, test_cases)

    out_file = settings.file.with_stem(f"{settings.file.stem}_judged")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    summary = get_summary(results)
    sum_file = settings.file.with_stem(f"{settings.file.stem}_summary")
    with open(sum_file, "w") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    print(summary)
