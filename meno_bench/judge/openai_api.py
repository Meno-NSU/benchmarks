from deepeval.models import DeepEvalBaseLLM
from openai import OpenAI, AsyncOpenAI
from meno_bench.judge.proxy import get_http_client, get_async_http_client
from meno_bench.settings import OpenAIJudgeSettings


class OpenAIAPINetworkModel(DeepEvalBaseLLM):
    def __init__(self, base_url: str, model_name: str, api_key: str, proxy: str | None = None):
        self.model_name = model_name
        http_client = get_http_client(proxy)
        self.client = OpenAI(api_key=api_key, base_url=base_url, http_client=http_client)
        http_async_client = get_async_http_client(proxy)
        self.async_client = AsyncOpenAI(api_key=None, base_url=base_url, http_client=http_async_client)

    def load_model(self):
        return self

    def generate(self, prompt: str) -> str:
        return (
            self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=1000,
            )
            .choices[0]
            .text
        )

    async def a_generate(self, prompt: str) -> str:
        resp = await self.async_client.completions.create(
            model=self.model_name,
            prompt=prompt,
            max_tokens=1000,
        )
        return resp.choices[0].text

    def get_model_name(self):
        return self.model_name


def get_model(settings: OpenAIJudgeSettings) -> OpenAIAPINetworkModel:
    return OpenAIAPINetworkModel(settings.address, settings.model, settings.api_key, settings.proxy)
