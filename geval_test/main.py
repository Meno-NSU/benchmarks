from geval_test.settings import get_settings, InferenceSettings
from geval_test.inference import inference
from geval_test.judge import judge


def main():
    settings = get_settings("judge_google_proxy.env")
    if isinstance(settings, InferenceSettings):
        inference(settings)
    else:
        judge(settings)


if __name__ == "__main__":
    main()
