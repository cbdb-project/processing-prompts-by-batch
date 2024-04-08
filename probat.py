import os
import random
import time
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import anthropic

with open("api_key.txt", "r") as file:
    api_key_str = file.read()
    client = anthropic.Anthropic(
        api_key=api_key_str,
    )


def anthropic(text):
    message = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        messages=[{"role": "user", "content": text}],
    )
    return message.content[0].text


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


# configuration
TEMP_BATCH_SIZE = 10
TIMEOUT = 3
TIMEOUT_OFFSET = 1

if os.path.exists("output.txt"):
    os.remove("output.txt")

prompt_list = []
with open("prompts.txt", "r", encoding="utf-8-sig") as f:
    lines = f.readlines()
    for line in lines:
        prompt_list.append([line.strip()])

temp_output_list = []
if len(prompt_list) < TEMP_BATCH_SIZE:
    TEMP_BATCH_SIZE = len(prompt_list)
for i in range(0, len(prompt_list), TEMP_BATCH_SIZE):
    temp_batch = prompt_list[i : i + TEMP_BATCH_SIZE]
    for prompt in temp_batch:
        timeout = TIMEOUT + random.random() * 0.1 * TIMEOUT_OFFSET
        time.sleep(timeout)
        # output_record = call_g4f(prompt[0])
        output_record = anthropic(prompt[0])
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
