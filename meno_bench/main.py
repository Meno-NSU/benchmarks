from meno_bench.settings import get_settings, InferenceSettings
from meno_bench.inference import inference
from meno_bench.judge import judge


def main():
    settings = get_settings()
    if isinstance(settings, InferenceSettings):
        inference(settings)
    else:
        judge(settings)


if __name__ == "__main__":
    main()
