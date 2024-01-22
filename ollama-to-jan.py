import argparse
import json
import os
import re
from datetime import datetime

import requests


OLLAMA_HOST = "http://localhost:11434"
JAN_MODEL_DIR = os.path.expanduser("~/jan/models")
DEFAULT_JAN_MODEL_PARAMETER = {
    "temperature": 0.7,
    "top_p": 0.95,
    "stream": True,
    "max_tokens": 2048,
    "stop": [],
    "frequency_penalty": 0,
    "presence_penalty": 0
}


def call_ollama_api(endpoint, data=None):
    url = OLLAMA_HOST + endpoint
    if data:
        response = requests.post(url, json=data)
    else:
        response = requests.get(url)
    response.raise_for_status()
    return response.json()


def parse_ollama_model_path(modelfile: str):
    model_file = re.search("[\^\n]FROM (?P<path>.+)", modelfile)["path"]
    return model_file


def parse_ollama_parameters(parameters: str):
    # Ollama: https://github.com/jmorganca/ollama/blob/main/docs/modelfile.md
    # For Jan there are
    #   engine parameters: https://nitro.jan.ai/features/load-unload/#table-of-parameters
    #   model parameters: https://nitro.jan.ai/api-reference/#tag/Chat-Completion/operation/createChatCompletion
    para_map = {
        "num_gpu": "ngl",
        "num_ctx": "ctx_len",
        "num_thread": "cpu_threads",
        "temperature": "temperature",
        "top_p": "top_p",
        "num_predict": "max_tokens",
        "stop": "stop",
    }
    params = {"stop": []}
    for match in re.findall("[\^\n](?P<key>[a-z])\s+(?P<value>.*)", parameters):
        key, value = match["key"], match["value"]
        if key == "stop":
            params[key].append(value)
        elif key in para_map:
            params[para_map[key]] = value
    return params


def transfrom_template(template: str):
    # TODO: transform more complex template
    return (
        template.replace("{{ .System }}", "{system_prompt}")
                .replace("{{ .Prompt }}", "{prompt}")
    )


def sync(
    model_filter: str = "",
):
    for model in call_ollama_api("/api/tags").get("models"):
        tag_name = model.get("name")
        
        if model_filter and not re.search(model_filter, tag_name):
            continue
        
        model_name = tag_name.replace(":", "-")
        model_info = call_ollama_api("/api/show", {"name": tag_name})
        model_path = parse_ollama_model_path(model_info.get("modelfile"))
        model_parameters = parse_ollama_parameters(model_info.get("parameters"))
        template = transfrom_template(model_info.get("template"))

        model_json = {
            "object": "model",
            "version": 1,
            "format": "gguf",
            "source_url": "N/A",
            "id": tag_name,
            "name": model_name.title().replace("-", " "),
            "created": int(datetime.fromisoformat(model.get("modified_at")).timestamp() * 1000),
            "description": f"{model_name} - linked from Ollama",
            "settings": {
                "prompt_template": template,
            },
            "parameters": {**DEFAULT_JAN_MODEL_PARAMETER, **model_parameters},
            "metadata": {
                "author": "User",
                "tags": [],
                "size": model.get("size")
            },
            "engine": "nitro"
        }
        
        try:
            assert os.path.isfile(model_path)
            os.mkdir(os.path.join(JAN_MODEL_DIR, model_name))
            os.link(model_path, os.path.join(JAN_MODEL_DIR, model_name, model_name))
            with open(os.path.join(JAN_MODEL_DIR, model_name, "model.json"), "w") as fp:
                json.dump(model_json, fp, indent=2)
        except Exception as err:
            print(err)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync Ollama models to Jan.")
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="",
        help="The model(s) to sync. Use regexp to sync matched ones and empty to sync all.",
    )
    args = parser.parse_args()
    sync(args.model)