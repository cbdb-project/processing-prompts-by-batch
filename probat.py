import os
import random
import time
import re
import base64

# Configuration
PROMPT_BATCH = 1  # number of prompts to concatenate with \n
TEMP_BATCH_SIZE = 10 # number of prompts to generate in a single batch. Don't need to change this for most cases
TIMEOUT = 0.5
TIMEOUT_OFFSET = 0.5
SEPARATOR_LIST = [".", "。", ",", ", ", "\\n", "\n"]
LEN_THRESHOLD = 2000
# api_choice: gemini, deepseek, openai, openai_harvard, openai_harvard_reimbursed, anthropic, call_g4f, qwen, volcengine, qwen_vl, gemini_vl...
api_choice = "deepseek"

with open("api_key.txt", "r") as file:
    api_key_str = file.read()


## Google Gemini
def gemini(text):
    # response = client.generate_content(text)
    response = client.models.generate_content(
    model = gemini_model,
    contents= text
)
    # return to_markdown(response.text)
    return response.text


## Deepseek V3
def deepseek(text):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": text},
        ],
        # temperature =  0.7,
        # top_p = 0.1,
    )
    return response.choices[0].message.content

def opneai(text):
    response = client.chat.completions.create(
        model= model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text},
        ],
        max_tokens= max_tokens,
        # temperature =  0.7,
        # top_p = 0.1,
    )
    return response.choices[0].message.content


## OpenAI Harvard
def openai_harvard(text):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text},
        ],
        "max_tokens": max_tokens,
        # "temperature": 0.7,
        # "top_p": 0.1,
    }
    response = requests.post(
        "https://go.apis.huit.harvard.edu/ais-openai-direct/v1/chat/completions",
        headers=headers,
        json=payload,
    )
    response_json = response.json()
    # print(response_json)
    return response_json["choices"][0]["message"]["content"]

# OpenAI Harvard Reimbursed
def openai_harvard_reimbursed(text):
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": text},
        ],
    }
    response = requests.post(
        "https://go.apis.huit.harvard.edu/ais-openai-direct-limited-schools/v1/chat/completions",
        headers=headers,
        json=payload,
    )
    response_json = response.json()
    return response_json["choices"][0]["message"]["content"]

# Glaude-3
def anthropic(text):
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100000,
        messages=[{"role": "user", "content": text}],
    )
    return message.content[0].text


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
        model="qwen3-235b-a22b-instruct-2507",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text},
        ],
        temperature=0.8,
        top_p=0.8,
    )
    return completion.choices[0].message.content

def volcengine(text):
    response = client.chat.completions.create(
        model="deepseek-v3-241226",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text},
        ],
    )
    return response.choices[0].message.content

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def qwen_vl(img_path):
    base64_image = encode_image(img_path)
    completion = client.chat.completions.create(
        model="qwen-vl-plus-latest",
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
    return response.text

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

    gemini_model = "gemini-2.0-flash"
    client = genai.Client(api_key=api_key_str)
elif api_choice == "deepseek":
    from openai import OpenAI

    client = OpenAI(api_key=api_key_str, base_url="https://api.deepseek.com/")
elif api_choice == "openai":
    from openai import OpenAI

    client = OpenAI(api_key=api_key_str)
    model = "gpt-5"
    max_tokens = 30000
elif api_choice == "openai_harvard":
    import requests

    model = "gpt-5"
    max_tokens = 30000
    headers = {
        "api-key": api_key_str,
        "Content-Type": "application/json",
        "Accept-Encoding": "gzip, deflate, identity",
    }
elif api_choice == "openai_harvard_reimbursed":
    import requests

    model = "gpt-5"
    max_tokens = 30000
    headers = {
        "api-key": api_key_str,
        "Content-Type": "application/json",
        "Accept-Encoding": "gzip, deflate, identity",
    }
elif api_choice == "anthropic":
    import anthropic

    client = anthropic.Anthropic(api_key=api_key_str)

elif api_choice == "qwen":
    from openai import OpenAI

    os.environ["DASHSCOPE_API_KEY"] = api_key_str
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
elif api_choice == "volcengine":
    from openai import OpenAI

    os.environ["ARK_API_KEY"] = api_key_str
    client = OpenAI(
        api_key=os.getenv("ARK_API_KEY"),
        base_url="https://ark.cn-beijing.volces.com/api/v3",
    )
elif api_choice == "call_g4f":
    from g4f.client import Client

    client = Client()
elif api_choice == "qwen_vl":
    from openai import OpenAI

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
