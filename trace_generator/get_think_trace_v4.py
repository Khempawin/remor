from botocore.exceptions import ClientError
from tqdm import tqdm
# from datasets import load_dataset
from transformers import AutoTokenizer
from ratelimit import limits, sleep_and_retry

import boto3
import pandas as pd
import concurrent.futures
# import re
import pickle
from pathlib import Path


"""

Get thinking trace from sonnet 3.7 of ICLR 2017 - 2020
Input as a dataframe
Columns:
    'year': year of conference
    'note': metadata from openreview.net
    'pdf_url': url to original PDF
    'full_text': Parsed and Organized Full Text from pdf_url
    'title': Title of Paper
    'abstract': Abstract of Paper
    'user_message': Input prompt to generate review
    'prompt_length': Length of input prompt after tokenized by google-bert/bert-base-uncased. This is used to determine if the prompt is longer than supported in the model.
    'file_id': File name that was obtained from pdf_url
    'review_count': Number of existing reviews related to submission

"""


EXISTING_RESULT_ID = set()
TOKENIZER = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
OUTPUT_PATH = "output_iclr"
AWS_ACCESS_KEY_ID = "<replace_with_access_key_id>"
AWS_SECRET_ACCESS_KEY = "<replace_with_secret_access_key"
AWS_REGION_NAME = "<replace_with_region_name>"


@sleep_and_retry
@limits(calls=32, period=60)
def generated_reasoning_for_review(user_message):
    # Initialize the Bedrock client
    client = boto3.client(
        'bedrock-runtime', 
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION_NAME)

    # Specify the model ID. For the latest available models, see:
    # https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html
    model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    # model_id = "us.deepseek.r1-v1:0"

    # Create the message with the user's prompt
    conversation = [
        {
            "role": "user",
            "content": [{"text": user_message}],
        }
    ]

    # Configure reasoning parameters with a 2000 token budget
    reasoning_config = {
        "thinking": {
            "type": "enabled",
            "budget_tokens": 2000
        }
    }

    try:
        # Send message and reasoning configuration to the model
        response = client.converse(
            modelId=model_id,
            messages=conversation,
            additionalModelRequestFields=reasoning_config
        )

        # Extract the list of content blocks from the model's response
        content_blocks = response["output"]["message"]["content"]

        reasoning = None
        text = None

        # Process each content block to find reasoning and response text
        for block in content_blocks:
            if "reasoningContent" in block:
                if "reasoningText" in block["reasoningContent"]:
                    reasoning = block["reasoningContent"]["reasoningText"]["text"]
                else:
                    reasoning = block["reasoningContent"]
                    print("ERROR", block["reasoningContent"])
                    exit(1)
            if "text" in block:
                text = block["text"]

        return reasoning, text

    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        exit(1)


def process_record(record):
    save_name = "{}.pickle".format(record["file_id"])
    save_path = f"{OUTPUT_PATH}/{save_name}"
    
    if save_name in EXISTING_RESULT_ID:
        prompt_length = record.get("prompt_length") 
        with open(save_path, "rb") as f:
            record = pickle.load(f)
        record["prompt_length"] = prompt_length
        return record
    
    if record.get("prompt_length") > 64_000:
        record["thinking_trace_sonnet"] = "ERROR"
        record["review_sonnet"] = "ERROR"
        return record
    
    review = generated_reasoning_for_review(record["user_message"])
    record["thinking_trace_sonnet"] = review[0]
    record["review_sonnet"] = review[1]
    
    
    with open(save_path, "wb") as f:
        pickle.dump(record, f)
    return record


def main():
    # Get Existing Results
    output_path = Path(OUTPUT_PATH)
    
    for entry in output_path.glob("*.pickle"):
        EXISTING_RESULT_ID.add(entry.name)
    
    if(len(EXISTING_RESULT_ID) != 0):
        print("Found some existing results: Skipping those records")
    
    tqdm.pandas()
    
    # Load submission dataframe
    with open("iclr_2017_2020_submissions.pickle", "rb") as f:
        papers = pickle.load(f)
    
    papers_list = papers.to_dict(orient="records")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        results = list(tqdm(executor.map(process_record, papers_list), total=len(papers_list)))
    
    # Save results as pickle since openreview.openreview.Note is not serializable by pyarrow
    with open("think_trace_iclr_2017_2020.pickle", "wb") as f:
        pickle.dump(pd.DataFrame(results), f)


if __name__ == "__main__":
    main()
