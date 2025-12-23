from gigachat import GigaChat
from meno_bench.settings import JudgeSettings
from deepeval.models import DeepEvalBaseLLM


class GigaNetworkModel(DeepEvalBaseLLM):

    def __init__(self, api_key: str, model_name: str):
        self.client = GigaChat(
            credentials=api_key,
            scope="GIGACHAT_API_PERS",
            model=model_name,
            # http_options={
            #     "httpx_client": http_client,
            #     "httpx_async_client": async_http_client,
            # },
        )
        self.model_name = model_name
        

    def load_model(self):
        return self

    def generate(self, prompt: str) -> str:
        return self.client.chat(prompt).choices[0].message.content
    
    async def a_generate(self, prompt: str) -> str:
        resp = await self.client.achat(prompt)
        return resp.choices[0].message.content

    def get_model_name(self):
        return self.model_name


def get_model(settings: JudgeSettings) -> GigaNetworkModel:
    return GigaNetworkModel(settings.api_key, settings.model)
