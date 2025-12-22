# This is a Library and a CLI tool to test rag system

# Installation

Just install as [uv](https://docs.astral.sh/uv/getting-started/installation/) tool:
```bash
uv tool install https://github.com/Meno-NSU/benchmarks.git
```

# Usage

This tool can work with command line arguments or corresponding environment variables. Also any env file can be supplied by `-e` flag.

This tool provides 2 modes: Inference and Judge.

## Inference

To inference you need a json file with format:
```json
[
  {
    "question": "...",
    "ground_truth": "...",
  },
  {
    "question": "...",
    "ground_truth": "...",
  },
  "..."
]
```

Example of env file for inference is `inference.example.env`:
```bash
FILE="correct_json.json"
ADDRESS="http://10.1.15.44:9006/v1/chat/completions"
MODEL="menon-1"
INFERENCE=1
```

or command line variant:

```bash
meno-bench -i -f correct_json.json -a http://10.1.15.44:9006/v1/chat/completions -m menon-1
```

The address is backend endpoint for inference. It must accept messages in format:
```json
{
    "model": "...",
    "messages": [
        {
            "role": "user",
            "content": "..."
        }
    ],
    "stream": false,
    "user": "test"
}
```

#### The output will be written to file `[INPUT FILE NAME]_out.json`

## Judge

There are two options to use judge: via google API or via openai API interface.

### OpenAI API interface

this is standard interface for llms. Required config is `judge_openai_api.example.env`:
```bash
FILE="correct_json_out.json"
ADDRESS="http://192.168.3.213:9111/v1"
MODEL="meno-tiny-0.1"
API_KEY=""
```

or command line variant:

```bash
meno-bench -f correct_json_out.json -a http://192.168.3.213:9111/v1 -m meno-tiny-0.1 -k ""
```

It must accept file from inference stage.

### Google API

This is mode to judge via google gemini API. Required config is `judge_google_proxy.example.env`:

```bash
FILE="correct_json_out.json"
PROXY="socks5://127.0.0.1:9150"
MODEL="gemini-2.5-flash"
INFERENCE=0
USE_GEMINI=1
USE_GEMINI_LIVE=0
API_KEY=""
```

or command line variant:

```bash
meno-bench -f correct_json_out.json -p socks5://127.0.0.1:9150 -m gemini-2.5-flash -g -k ""
```

If you want to force use live mode, set `USE_GEMINI_LIVE=1` or add `-u` argument. Since google swapping live models and often removes pure live text-to-text models, it is not recommended to use it. Also if model name contain `live` keyword it will be treated as live model.

#### The output will be written to file `[INPUT FILE NAME]_judged.json`

#### The summary will be written to file `[INPUT FILE NAME]_summary.json`