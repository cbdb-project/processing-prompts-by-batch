import os
import random
import time
import re

with open("api_key.txt", "r") as file:
    api_key_str = file.read()


# # Google Gemini
# import google.generativeai as genai

# genai.configure(api_key=api_key_str)
# model = genai.GenerativeModel("gemini-pro")


# def gemini(text):
#     response = model.generate_content(text)
#     # return to_markdown(response.text)
#     return response.text


## Deepseek V2
from openai import OpenAI


def deepseek(text):
    client = OpenAI(api_key=api_key_str, base_url="https://api.deepseek.com/")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": text},
        ],
    )
    return response.choices[0].message.content


# # Glaude-3
# import anthropic
# client = anthropic.Anthropic(
#     api_key=api_key_str,
# )
# def anthropic(text):
#     message = client.messages.create(
#         model="claude-3-opus-20240229",
#         max_tokens=1024,
#         messages=[{"role": "user", "content": text}],
#     )
#     return message.content[0].text

# # gpt4free
# def call_g4f(text, max_retries=3, retry_delay=5):
#     from g4f.client import Client

#     client = Client()
#     for attempt in range(max_retries):
#         try:
#             response = client.chat.completions.create(
#                 model="gpt-3.5-turbo",
#                 messages=[
#                     {"role": "user", "content": text},
#                 ],
#             )
#             return response.choices[0].message.content
#         except Exception as e:
#             if e.response.status == 503:
#                 print(
#                     f"Received a error, attempt {attempt + 1} of {max_retries}. Retrying in {retry_delay} seconds..."
#                 )
#                 time.sleep(retry_delay)
#             else:
#                 raise
#     raise Exception("Maximum retries reached, the service may be unavailable.")


def clean_text(text):
    text = text.replace("\n", "<br />")
    return text

def find_nearest_separator(s, index, separators):
    # Create a regex pattern to match any of the separators
    pattern = '|'.join(re.escape(sep) for sep in separators)
    
    # Find all occurrences of the separators in the string
    matches = [(m.start(), m.end()) for m in re.finditer(pattern, s)]
    
    # Filter matches to find the nearest ones on the left and right
    left_match = max((m for m in matches if m[0] <= index), default=None, key=lambda x: x[0])
    right_match = min((m for m in matches if m[0] > index), default=None, key=lambda x: x[0])
    
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
            print(f"No valid separator found near the target index {target_index} for prompt: {prompt[start:]}\n")
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

# configuration
TEMP_BATCH_SIZE = 10
TIMEOUT = 0.5
TIMEOUT_OFFSET = 0.5
SEPARATOR_LIST = [".","。",",",", ", "\\n", "\n"]
LEN_THRESHOLD = 2000

if os.path.exists("output.txt"):
    os.remove("output.txt")

prompt_list = []
with open("prompts.txt", "r", encoding="utf-8-sig") as f:
    lines = f.readlines()
    for line in lines:
        prompt_list.append([line.strip()])

prompt_prefix = ""
with open("prompt_prefix.txt", "r", encoding="utf-8-sig") as f:
    prompt_prefix = f.read().replace("\n", "\\n")
    prompt_prefix = prompt_prefix.strip()

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
                    output_record = deepseek(prompt_with_prefix)
                else:
                    output_record = output_record + "<SEP>" + deepseek(prompt_with_prefix)
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
        print(f"Finished {i + TEMP_BATCH_SIZE}/{len(prompt_list)} prompts")
    else:
        print(f"Finished {len(prompt_list)}/{len(prompt_list)} prompts")
print("Finished all prompts")
