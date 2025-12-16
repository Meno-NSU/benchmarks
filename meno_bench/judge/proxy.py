import httpx

def get_http_client(proxy: str | None = None) -> httpx.Client:
    return httpx.Client(proxy=proxy)

def get_async_http_client(proxy: str | None = None) -> httpx.AsyncClient:
    return httpx.AsyncClient(proxy=proxy)
