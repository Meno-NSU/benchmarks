from meno_bench.settings import InferenceSettings
from meno_bench.models import TestCasesFile, TestCasesFileFull, TestCaseFromFileFull
import json
import tqdm
import requests
from time import time
from pathlib import Path


def retrieve(
    address: str, model: str, text: str
) -> tuple[str, float] | tuple[None, None]:
    messages = [
        {"role": "user", "content": text},
    ]

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "user": "test",
    }

    time_s = time()
    response = requests.post(address, json=payload)
    time_s = time() - time_s
    if response.status_code != 200:
        return None, None
    return response.json()["choices"][0]["message"]["content"], time_s


def inference_cases(
    settings: InferenceSettings, cases: TestCasesFile
) -> TestCasesFileFull:
    results: TestCasesFileFull = []
    try:
        for jsoned_test in tqdm.tqdm(cases):
            if "model_answer" not in jsoned_test or jsoned_test["model_answer"] is None:
                model_answer, time_s = retrieve(
                    settings.address, settings.model, jsoned_test["question"]
                )
                if model_answer is None:
                    raise Exception("Could not retrieve model answer")
                results.append(
                    TestCaseFromFileFull(
                        question=jsoned_test["question"],
                        ground_truth=jsoned_test["ground_truth"],
                        model_answer=model_answer,
                        time_s=time_s,
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
