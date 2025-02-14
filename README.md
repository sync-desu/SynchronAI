<h1 align="center">SynchronAI</h1>


## What it is:
- A simple take on the **Retrieval Augmented Generation** (RAG) technique used with Large Language Models (LLMs).
- Uses **[LangChain](https://python.langchain.com/api_reference/)** (build application level workflows with LLMs) and **[Ollama](https://ollama.com/)** (library and tool for serving open-source LLMs on your local machine for personal use cases).


## Pre-Requisites:
- **Python** (tested on ver. `3.10.11`)
    - and the modules mentioned in `requirements.txt`
- **[Ollama](https://ollama.com/)** (ver. `latest`) 
    - and the following Language Models:
        - `nomic-embed-text:latest` (for text embedding)
        - `qwen2-math:7b` (for scientific & mathematical reasoning)
- Read additional information at the end...


## Installation Guide:
- Clone via:
    - HTTP or GitHub CLI
    - Download ZIP
    - Open on Desktop Application
    - Download release `.zip` package
- In the main directory, run `pip install -r requirements.txt` to install all the pre-requisite python modules.
- Download and install [Ollama](https://ollama.com/) and all of its pre-requisites using the following commands:
    - `ollama serve`: to ensure the ollama service is running on your system
    - `ollama pull <model_name>`: to download a language model locally (download pre=requisite models)


## Additional Information:
- Consider downgrading from the `7B` parameter `qwen2-math` model as it may not run on your system as intended.
- It **may damage** your system if you do not have a GPU which can provide sufficient compute power.
- **Specifications of the system this software was tested on:**
```
CPU: Intel Core i7 13620H
GPU: Nvidia RTX 4060 8GB VRAM GDDR6 (mobile)
RAM: 16GB single-channel DDR5 (@ 4800 MT/s)
```