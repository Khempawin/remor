"Qwen-7B-Review-ICLR-lora-sft"
# Use a pipeline as a high-level helper
from transformers import pipeline, AutoTokenizer
from datasets import load_dataset
from typing import TypedDict
from tqdm import tqdm

import pandas as pd

ds = load_dataset("pawin205/iclr-2017-2020-peer-review-with-thinking-trace", split="short")

print(len(ds))

## load tokenizer (same for all models handled here)
TOKENIZER = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

def get_tokenized_length(message):
    return len(TOKENIZER(message)["input_ids"])

messages = [
    {"role": "user", "content": "Who are you?"},
]

pipe = pipeline("text-generation", model="pawin205/Qwen-7B-Review-ICLR-90th-sft")

MAX_COMPLETION_TOKENS = 4096

# Generate reviews
class GeneratedReview(TypedDict):
    prompt: str
    review: str
    prompt_length: int=0
    review_length: int=0

results = list()

for record in tqdm(ds):
    #print(record)
    response = pipe(record["conversations"][:1], max_new_tokens=MAX_COMPLETION_TOKENS)
    review_full = response[0]["generated_text"][-1]["content"]
    record_result = GeneratedReview(
            prompt=record["conversations"][0]["content"],
            review=review_full,
            prompt_length=record["prompt_length"],
            review_length=get_tokenized_length(review_full)
    )
    results.append(record_result)
    

review_df = pd.DataFrame(results)

review_df.to_parquet("generated_review_90th_sft.parquet", engine="pyarrow")

print(review_df.head(10))
