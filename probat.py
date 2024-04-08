import os
import random
import time


def call_g4f(text):
    # pip install -U g4f
    from g4f.client import Client

    text_utf8 = text.encode("utf-8")
    client = Client()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": text_utf8,
            }
        ],
    )
    return response.choices[0].message.content


def clean_text(text):
    text = text.replace("\n", "<br />")
    return text


# configuration
TEMP_BATCH_SIZE = 10
TIMEOUT = 0.5
TIMEOUT_OFFSET = 0.5

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
        output_record = call_g4f(prompt[0])
        output_record = clean_text(output_record)
        temp_output_list.append(output_record)
        with open("output.txt", "a+", encoding="utf-8-sig") as f:
            f.write(output_record + "\n")
    print(f"Finished {i + TEMP_BATCH_SIZE}/{len(prompt_list)} prompts")
    temp_output_list = []
print("Finished all prompts")
