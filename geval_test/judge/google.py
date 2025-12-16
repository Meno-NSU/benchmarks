from google import genai
from google.genai.errors import ServerError
from geval_test.settings import JudgeSettings
import asyncio
import httpx
from deepeval.models import DeepEvalBaseLLM
from geval_test.judge.proxy import get_http_client, get_async_http_client
import time


def get_client(api_key: str, proxy: str | None = None) -> genai.Client:
    http_client = get_http_client(proxy)
    async_http_client = get_async_http_client(proxy)

    return  genai.Client(
        api_key=api_key,
        http_options={
            "httpx_client": http_client,
            "httpx_async_client": async_http_client,
            "api_version": "v1beta"
        },
    )


class GeminiNetworkModel(DeepEvalBaseLLM):

    def __init__(self, client: genai.Client, model_name: str, live: bool | None = None):
        self.model_name = model_name
        self.client = client
        if live is not None:
            self.live = live
        else:
            self.live = "live" in model_name
        
        if self.live:
            self.config = {
                "response_modalities": ["TEXT"],
            }
        else:
            self.config = {}

    def load_model(self):
        return self

    def generate(self, prompt: str) -> str:
        if self.live:
            return asyncio.run(self.a_generate(prompt))
        else:
            return self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            ).text
    
    async def a_generate(self, prompt: str) -> str:
        if self.live:
            async with self.client.aio.live.connect(model=self.model_name, config=self.config) as session:
                await session.send(input=prompt, end_of_turn=True)
                msgs = []
                async for msg in session.receive():
                    msgs.append(msg.text)
                return "".join(msg for msg in msgs if msg is not None)
        else:
            try:
                resp = self.client.models.generate_content(model=self.model_name, contents=prompt)
            except ServerError as e:
                print("503 encountered. Sleeping 60s...")
                if e.code == 503:
                    time.sleep(60)
            return resp.text


    def get_model_name(self):
        return self.model_name


def get_model(settings: JudgeSettings) -> GeminiNetworkModel:
    client = get_client(settings.api_key, settings.proxy)
    return GeminiNetworkModel(client, settings.model, settings.use_gemini_live)
