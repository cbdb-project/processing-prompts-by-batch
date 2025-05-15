# Processing Prompts by Batch (probat)

This project provides a Python script to interact with multiple Large Language Models (LLMs) from various APIs like Google Gemini, Deepseek, OpenAI (Harvard), Anthropic, GPT-4Free, and Qwen. The script allows batch processing of prompts, splitting lengthy inputs into manageable chunks, and retrieving responses from the specified LLM.

## NOTICE

We are currently using the Deep Seek V3 API as the default. 

## Branch

### [Jiajun Zou](https://github.com/jzou19957)'s G4F implement

https://github.com/jzou19957/Unlimited-Excel-Processing-through-GPT-3.5-API

## Features

- Multi-API Support: Easily switch between different language models including Google Gemini, Deepseek, OpenAI (Harvard), Anthropic, GPT-4Free, and Qwen by setting api_choice.
- Chunk Splitting: Automatically splits long prompts into smaller chunks based on specified separators to ensure they fit within token limits.
- Text Cleaning: Converts newline characters to `<br />` for formatted output.
- Configurable Parameters: Adjust batch size, timeout, and token length thresholds for efficient processing.
- Batch Processing: Processes multiple prompts at a time, writing outputs to an external file.

## Requirements

  - Python 3.9 or newer
  - Dependencies for specific APIs:
    - `google.generativeai` for Google Gemini
    - `openai` for Deepseek and Qwen
    - `anthropic` for Anthropic API
    - `requests` for OpenAI (Harvard)
    - `g4f` for GPT-4Free
    - `qwen_vl` for qwen2.5-72b-instruct
    - `gemini_vl` for gemini-2.0-flash

## Installation

1. Before running the script, ensure you have Python installed on your system, and then install the required SDK using pip:

For `Deep Seek` and `Qwen` users:
```bash
pip install -U openai
```

For `Gemini` users:
```bash
pip install -U -q "google"
pip install -U -q "google.genai"
```

For `anthropic` users:
```bash
pip install anthropic
```

For `GPT-4Free` users:
```bash
pip install g4f
```

2. You can obtain a API key from the LLM API website(For exampel, deep seek users can visit https://platform.deepseek.com). After acquiring your API key, save the key to `api_key.txt` in the root directory of the current repository.

![image](https://github.com/cbdb-project/processing-prompts-by-batch/assets/8538710/f38a0f0f-732d-4f71-bdbd-b2054831b92d)

## Usage

1. Place your text prompts in a file named `prompts.txt`, with one prompt per line.
2. (Optional) Add a prefix prompt to the `prompt_prefix.txt` file to include a prefix for each prompt if needed.
3. Change `api_choice = "deepseek"` in `probat.py` to the LLM you want to use and save the file. The available options are: gemini, deepseek, openai_harvard, anthropic, call_g4f, and qwen.
4. Run probat.py.
5. The script will process all prompts and save the outputs in `output.txt`. If this file already exists, it will be overwritten.

## Configuration

You can adjust the following configurations at the beginning of the script:

  - `TEMP_BATCH_SIZE`: Number of prompts processed in each batch (default is 10).
  - `TIMEOUT`: The base delay time between API calls (default is 0.5 seconds).
  - `TIMEOUT_OFFSET`: The offset for randomizing delay time (default is 0.5).
  - `LEN_THRESHOLD`: Character length threshold for splitting prompts (default is 2000).
  - `SEPARATOR_LIST`: List of separators to use when splitting long prompts.


## Disclaimer

Always adhere to the API provider's usage policies and guidelines.

## License

[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International license](https://creativecommons.org/licenses/by-nc-sa/4.0/)
