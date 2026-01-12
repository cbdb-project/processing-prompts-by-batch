import os
import random
import time
import re
import base64

print("Program starting...")
print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
start_time = time.time()

# Configuration
PROMPT_BATCH = 1  # number of prompts to concatenate with \n
TEMP_BATCH_SIZE = 10 # number of prompts to generate in a single batch. Don't need to change this for most cases
TIMEOUT = 0.5
TIMEOUT_OFFSET = 0.5
SEPARATOR_LIST = [".", "。", ",", ", ", "\\n", "\n"]
LEN_THRESHOLD = 7500
# api_choice: gemini, deepseek, openai, openai_harvard, openai_harvard_reimbursed, anthropic, call_g4f, qwen, volcengine, qwen_vl, gemini_vl...
api_choice = "deepseek"
# Thinking configuration
# Harvard OpenAI Direct may not support reasoning parameter yet
ENABLE_THINKING = False  # Set to True to enable thinking mode, False to disable (default)

with open("api_key.txt", "r") as file:
    api_key_str = file.read()


# Helper function to format response with reasoning
def format_with_reasoning(reasoning, content):
    """Format response with reasoning content if available."""
    if reasoning:
        return f"[Reasoning]\n{reasoning}\n\n[Answer]\n{content}"
    return content


## Google Gemini
def gemini(text):
    # Model-specific configuration
    temperature = 0.7
    top_p = 0.9
    thinking_budget = 1024  # Token budget for thinking mode

    # Build config with thinking control
    config = types.GenerateContentConfig(
        temperature=temperature,
        top_p=top_p,
        thinking_config=types.ThinkingConfig(
            thinking_budget=thinking_budget if ENABLE_THINKING else 0
        )
    )

    response = client.models.generate_content(
        model=gemini_model,
        contents=text,
        config=config
    )

    # Extract text parts and check for thinking content
    text_parts = []
    thinking_parts = []
    for candidate in response.candidates:
        for part in candidate.content.parts:
            if hasattr(part, 'thought') and part.thought and ENABLE_THINKING:
                thinking_parts.append(part.text)
            elif hasattr(part, 'text') and part.text:
                text_parts.append(part.text)

    result = ''.join(text_parts) if text_parts else response.text
    reasoning = ''.join(thinking_parts) if ENABLE_THINKING and thinking_parts else None
    return format_with_reasoning(reasoning, result)


## Deepseek V3
def deepseek(text):
    response = client.chat.completions.create(
        model=deepseek_model,
        messages=[{"role": "user", "content": text}],
        max_tokens=max_tokens,
    )

    message = response.choices[0].message
    reasoning = message.reasoning_content if (ENABLE_THINKING and hasattr(message, 'reasoning_content')) else None
    return format_with_reasoning(reasoning, message.content)

def opneai(text):
    # Reasoning effort: "none" (default for gpt-5.2), "low", "medium", "high"
    reasoning_effort = "medium" if ENABLE_THINKING else "none"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text},
        ],
        "max_tokens": max_tokens,
    }

    # Add reasoning config for models that support it (o-series and gpt-5+)
    if model.startswith("o") or model.startswith("gpt-5"):
        payload["reasoning"] = {"effort": reasoning_effort}

    response = client.chat.completions.create(**payload)
    message = response.choices[0].message
    reasoning = message.reasoning if (ENABLE_THINKING and hasattr(message, 'reasoning')) else None
    return format_with_reasoning(reasoning, message.content)


## OpenAI Harvard
def openai_harvard(text):
    # Reasoning effort: "none" (default for gpt-5.2), "low", "medium", "high"
    reasoning_effort = "medium" if ENABLE_THINKING else "none"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text},
        ],
        "max_tokens": max_tokens,
    }

    # Add reasoning config for models that support it (o-series and gpt-5+)
    if model.startswith("o") or model.startswith("gpt-5"):
        payload["reasoning"] = {"effort": reasoning_effort}

    response = requests.post(
        "https://go.apis.huit.harvard.edu/ais-openai-direct/v1/chat/completions",
        headers=headers,
        json=payload,
    )
    message = response.json()["choices"][0]["message"]
    reasoning = message.get("reasoning") if ENABLE_THINKING else None
    return format_with_reasoning(reasoning, message["content"])

# OpenAI Harvard Reimbursed
def openai_harvard_reimbursed(text):
    # Harvard API may not support reasoning parameter yet
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens,
    }

    # Note: reasoning parameter is not supported by Harvard's API endpoint
    # If needed in the future, uncomment the following:
    # if ENABLE_THINKING and model.startswith("o"):
    #     payload["reasoning"] = {"effort": "medium"}

    response = requests.post(
        "https://go.apis.huit.harvard.edu/ais-openai-direct-limited-schools/v1/chat/completions",
        headers=headers,
        json=payload,
    )

    # Check for errors in response
    response_data = response.json()
    if "choices" not in response_data:
        print(f"API Error Response: {response_data}")
        raise Exception(f"API returned error: {response_data.get('error', {}).get('message', 'Unknown error')}")

    message = response_data["choices"][0]["message"]
    return message["content"]

# Claude
def anthropic(text):
    # Claude supports extended thinking with the thinking parameter
    thinking_config = {"type": "enabled", "budget_tokens": 10000} if ENABLE_THINKING else {"type": "disabled"}

    # Use streaming to avoid timeout errors for long requests
    full_response = ""
    thinking_content = ""

    with client.messages.stream(
        model="claude-sonnet-4-5-20250929",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": text}],
        thinking=thinking_config,
    ) as stream:
        for event in stream:
            if hasattr(event, 'type'):
                if event.type == "content_block_delta":
                    if hasattr(event.delta, 'type'):
                        if event.delta.type == "thinking_delta" and ENABLE_THINKING:
                            thinking_content += event.delta.thinking
                        elif event.delta.type == "text_delta":
                            full_response += event.delta.text

    return format_with_reasoning(thinking_content if thinking_content else None, full_response)


# gpt4free
def call_g4f(text, max_retries=3, retry_delay=5):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": text},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            if e.response.status == 503:
                print(
                    f"Received a error, attempt {attempt + 1} of {max_retries}. Retrying in {retry_delay} seconds..."
                )
                time.sleep(retry_delay)
            else:
                raise
    raise Exception("Maximum retries reached, the service may be unavailable.")


def qwen(text):
    completion = client.chat.completions.create(
        model=qwen_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text},
        ],
        temperature=0.8,
        top_p=0.8,
    )

    message = completion.choices[0].message
    # QwQ model may include reasoning in the response, try to extract it
    reasoning = None
    if ENABLE_THINKING and hasattr(message, 'reasoning'):
        reasoning = message.reasoning

    return format_with_reasoning(reasoning, message.content)

def volcengine(text):
    response = client.chat.completions.create(
        model=volcengine_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text},
        ],
    )

    message = response.choices[0].message
    reasoning = message.reasoning_content if (ENABLE_THINKING and hasattr(message, 'reasoning_content')) else None
    return format_with_reasoning(reasoning, message.content)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def qwen_vl(img_path):
    base64_image = encode_image(img_path)
    completion = client.chat.completions.create(
        model="qwen3-vl-plus",
        messages=[{
            "role": "user", 
            "content": [
                {"type": "text", "text": prompt_prefix},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }]
    )
    return completion.choices[0].message.content

def gemini_vl(img_path):
    with open(img_path, "rb") as image_file:
        image_data = image_file.read()

    response = client.models.generate_content(
        model=gemini_model,
        contents=[
            prompt_prefix,
            types.Part.from_bytes(data=image_data, mime_type="image/png")
        ]
    )
    # Extract only text parts from the response to avoid warnings about non-text parts
    text_parts = []
    for candidate in response.candidates:
        for part in candidate.content.parts:
            if hasattr(part, 'text') and part.text:
                text_parts.append(part.text)
    return ''.join(text_parts) if text_parts else response.text

def clean_text(text):
    text = text.replace("\n", "<br />")
    return text


def find_nearest_separator(s, index, separators):
    # Create a regex pattern to match any of the separators
    pattern = "|".join(re.escape(sep) for sep in separators)

    # Find all occurrences of the separators in the string
    matches = [(m.start(), m.end()) for m in re.finditer(pattern, s)]

    # Filter matches to find the nearest ones on the left and right
    left_match = max(
        (m for m in matches if m[0] <= index), default=None, key=lambda x: x[0]
    )
    right_match = min(
        (m for m in matches if m[0] > index), default=None, key=lambda x: x[0]
    )

    return left_match, right_match


def split_by_threshold(prompt, separators, len_threshold):
    chunks = []
    start = 0
    n = 1

    while start < len(prompt):
        # Calculate the index where we want to find the nearest separator
        target_index = n * len_threshold

        if target_index >= len(prompt):
            chunks.append(prompt[start:])
            break

        left_sep, right_sep = find_nearest_separator(prompt, target_index, separators)

        if not left_sep and not right_sep:
            # If no separators are found, just cut at the target index
            print(
                f"No valid separator found near the target index {target_index} for prompt: {prompt[start:]}\n"
            )
            chunks.append(prompt[start:target_index])
            start = target_index
        else:
            # Choose the separator closest to the target index
            if left_sep and right_sep:
                if abs(left_sep[0] - target_index) <= abs(right_sep[0] - target_index):
                    sep_index = left_sep[1]
                else:
                    sep_index = right_sep[0]
            elif left_sep:
                sep_index = left_sep[1]
            else:
                sep_index = right_sep[0]

            chunks.append(prompt[start:sep_index])
            start = sep_index

        n += 1

    return chunks

def llm_api(text, api_choice):
    if api_choice in api_functions:
        return api_functions[api_choice](text)
    else:
        raise ValueError(f"Unknown API choice: {api_choice}")

api_functions = {
    "gemini": gemini,
    "deepseek": deepseek,
    "openai": opneai,
    "openai_harvard": openai_harvard,
    "anthropic": anthropic,
    "call_g4f": call_g4f,
    "qwen": qwen,
    "volcengine": volcengine,
    "qwen_vl": qwen_vl,
    "openai_harvard_reimbursed": openai_harvard_reimbursed,
    "gemini_vl": gemini_vl,
}

# Initialize LLM
if api_choice == "gemini" or api_choice == "gemini_vl":
    from google import genai
    from google.genai import types

    # Model selection - gemini supports thinking via thinking_config parameter
    gemini_model = "gemini-flash-latest"
    client = genai.Client(api_key=api_key_str)
elif api_choice == "deepseek":
    from openai import OpenAI

    # Model selection based on ENABLE_THINKING
    deepseek_model = "deepseek-reasoner" if ENABLE_THINKING else "deepseek-chat"
    client = OpenAI(api_key=api_key_str, base_url="https://api.deepseek.com/")
    max_tokens = 7500
elif api_choice == "openai":
    from openai import OpenAI

    # Model selection - reasoning is configured via payload in opneai() function
    client = OpenAI(api_key=api_key_str)
    model = "gpt-5.2"
    max_tokens = 30000
elif api_choice == "openai_harvard":
    import requests

    # Model selection - reasoning is configured via payload in openai_harvard() function
    model = "gpt-5.2"
    max_tokens = 30000
    headers = {
        "api-key": api_key_str,
        "Content-Type": "application/json",
        "Accept-Encoding": "gzip, deflate, identity",
    }
elif api_choice == "openai_harvard_reimbursed":
    import requests

    # Model selection - currently no reasoning support for Harvard reimbursed API
    model = "gpt-5.2"
    max_tokens = 30000
    headers = {
        "api-key": api_key_str,
        "Content-Type": "application/json",
        "Accept-Encoding": "gzip, deflate, identity",
    }
elif api_choice == "anthropic":
    import anthropic

    # Model selection - thinking is configured via thinking parameter in anthropic() function
    client = anthropic.Anthropic(api_key=api_key_str)
    max_tokens = 64000

elif api_choice == "qwen":
    from openai import OpenAI

    # Model selection based on ENABLE_THINKING (QwQ for reasoning, qwen3 for standard)
    qwen_model = "qwq-32b-preview" if ENABLE_THINKING else "qwen3-235b-a22b-instruct-2507"
    os.environ["DASHSCOPE_API_KEY"] = api_key_str
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
elif api_choice == "volcengine":
    from openai import OpenAI

    # Model selection based on ENABLE_THINKING
    volcengine_model = "deepseek-reasoner" if ENABLE_THINKING else "deepseek-chat"
    os.environ["ARK_API_KEY"] = api_key_str
    client = OpenAI(
        api_key=os.getenv("ARK_API_KEY"),
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )
elif api_choice == "call_g4f":
    from g4f.client import Client

    # Model selection - g4f uses fixed model, no thinking mode support
    client = Client()
elif api_choice == "qwen_vl":
    from openai import OpenAI

    # Model selection - vision model doesn't support thinking mode
    os.environ["QWEN_VL_API_KEY"] = api_key_str
    client = OpenAI(
        api_key=os.getenv("QWEN_VL_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

if os.path.exists("output.txt"):
    os.remove("output.txt")

prompt_list = []

prompt_prefix = ""
with open("prompt_prefix.txt", "r", encoding="utf-8-sig") as f:
    prompt_prefix = f.read().replace("\n", "\\n")
    prompt_prefix = prompt_prefix.strip()

# if the model name not end with _vl
if api_choice[-3:] != "_vl":
    with open("prompts.txt", "r", encoding="utf-8-sig") as f:
        lines = f.readlines()
        temp_batch = []
        for line in lines:
            line = line.strip()
            temp_batch.append(line)
            if len(temp_batch) == PROMPT_BATCH:
                prompt_list.append(["\n".join(temp_batch)])
                temp_batch = []
        # Add any remaining prompts
        if temp_batch:
            prompt_list.append(["\n".join(temp_batch)])

    print(f"Total batches: {len(prompt_list)} found. Starting generation...")

    temp_output_list = []

    if len(prompt_list) < TEMP_BATCH_SIZE:
        TEMP_BATCH_SIZE = len(prompt_list)
    for i in range(0, len(prompt_list), TEMP_BATCH_SIZE):
        temp_batch = prompt_list[i : i + TEMP_BATCH_SIZE]
        for prompt in temp_batch:
            timeout = TIMEOUT + random.random() * 0.1 * TIMEOUT_OFFSET
            time.sleep(timeout)
            # output_record = call_g4f(prompt[0])
            # output_record = anthropic(prompt[0])
            prompt_chunks = []
            if len(prompt[0]) > LEN_THRESHOLD:
                prompt_chunks = split_by_threshold(prompt[0], SEPARATOR_LIST, LEN_THRESHOLD)
            else:
                prompt_chunks = [prompt[0]]
            # print(prompt_chunks)
            output_record = ""
            for prompt_chunk in prompt_chunks:
                try:
                    prompt_with_prefix = prompt_prefix + prompt_chunk
                    if output_record == "":
                        output_record = llm_api(prompt_with_prefix, api_choice)
                    else:
                        output_record = (
                            output_record
                            + "<SEP>"
                            + llm_api(prompt_with_prefix, api_choice)
                        )
                except Exception as e:
                    print(f"Error: {e}")
                    output_record = "Error"
                    break
            # output_record = gemini(prompt[0])
            output_record = clean_text(output_record)
            temp_output_list.append(output_record)
        with open("output.txt", "a", encoding="utf-8") as f:
            for output_record in temp_output_list:
                f.write(output_record + "\n")
        temp_output_list = []
        if (i + TEMP_BATCH_SIZE) < len(prompt_list):
            print(f"Finished {i + TEMP_BATCH_SIZE}/{len(prompt_list)} prompts on {time.strftime('%H:%M:%S')}")
        else:
            print(f"Finished {len(prompt_list)}/{len(prompt_list)} prompts on {time.strftime('%H:%M:%S')}")
else:
    image_list = []
    # read from image paths from img folder
    for root, dirs, files in os.walk("img"):
        for file in files:
            image_list.append(os.path.join(root, file))
    image_list.sort()
    print(f"Total images: {len(image_list)} found. Starting generation...")
    counter = 0
    for img_path in image_list:
        counter += 1
        if counter % 10 == 0:
            print(f"Finished {counter}/{len(image_list)} images on {time.strftime('%H:%M:%S')}")
        timeout = TIMEOUT + random.random() * 0.1 * TIMEOUT_OFFSET
        time.sleep(timeout)
        try:
            output_record = llm_api(img_path, api_choice)
        except Exception as e:
            print(f"Error: {e}")
            output_record = "Error"
        output_record = clean_text(output_record)
        with open("output.txt", "a", encoding="utf-8") as f:
            f.write(output_record + "\n")
print("Finished all prompts")
elapsed_time = time.time() - start_time
print(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
