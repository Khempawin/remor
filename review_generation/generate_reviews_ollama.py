from experiment_util import generate_review_from_dataframe
from tqdm import tqdm
from datasets import load_dataset


import nltk

# Initialize tqdm for pandas
tqdm.pandas()

nltk.download('wordnet')
nltk.download('omw-1.4')

data_filtered = load_dataset("pawin205/iclr-2017-2020-peer-review-with-thinking-trace", split="train").to_pandas()

# Generate reviews
model_name = "qwen-7b-grpo-u"

qwen_7b_grpo_u = generate_review_from_dataframe(
   data_filtered, 
   model_name, 
   save_file_path="qwen_7b_grpo_u.pickle",
   save_to_file=True
)

# Generate reviews
model_name = "qwen-7b-grpo-h"

qwen_7b_grpo_h = generate_review_from_dataframe(
    data_filtered,
    model_name,
    save_file_path="qwen_7b_grpo_h.pickle",
    save_to_file=True
)
