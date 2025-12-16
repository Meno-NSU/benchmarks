from meno_bench.settings import InferenceSettings
from meno_bench.models import TestCasesFile, TestCasesFileFull, TestCaseFromFileFull
import json
import tqdm
import requests


def retrieve(address: str, model: str, text: str) -> str | None:
    messages = [
        {"role": "user", "content": text},
    ]

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "user": "test",
    }

    response = requests.post(address, json=payload)
    if response.status_code != 200:
        return None
    return response.json()["choices"][0]["message"]["content"]


def inference_cases(
    settings: InferenceSettings, cases: TestCasesFile
) -> TestCasesFileFull:
    results: TestCasesFileFull = []
    try:
        for jsoned_test in tqdm.tqdm(cases):
            if "model_answer" not in jsoned_test or jsoned_test["model_answer"] is None:
                results.append(
                    TestCaseFromFileFull(
                        question=jsoned_test["question"],
                        ground_truth=jsoned_test["ground_truth"],
                        model_answer=retrieve(
                            settings.address, settings.model, jsoned_test["question"]
                        ),
                    )
                )
    except Exception as e:
        print(e)
    return results


def inference(settings: InferenceSettings):
    with open(settings.file, "r") as f:
        jsoned_tests: TestCasesFile = json.load(f)

    result = inference_cases(settings, jsoned_tests)

    out_file = settings.file.with_stem(f"{settings.file.stem}_out")
    with open(out_file, "w") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
