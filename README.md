# REMOR : Automated Peer Review Generation with LLM Reasoning and Multi-Objective Reinforcement Learning

## Notes
- distilbert_models can be downloaded from [Link](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3002238)
- for `grpo_training` and `review_generation`, please download and extract the contents of distilbert_models in their directories.

## Loading reward function and review generation
### Environment Setup
```
conda create -n grpo python=3.11
conda activate grpo
pip install openreview-py
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install --user -U nltk
pip install ollama pandas matplotlib numpy seaborn tqdm
pip install git+https://github.com/huggingface/transformers.git@main
```

## Calculating human-aligned weights
### Environment Setup
```
conda create -n human-value-calculation python=3.12  

conda activate human-value-calculation 
  
pip install cvxpy ecos scikit-learn numpy pandas pyarrow

```

Calculation can be done by the notebook file optim_cv.ipynb located in [/human_value_calculation](/human_value_calculation)

## Training

### SFT via LLaMA-Factory
#### Environment Setup
```
conda create -n llama-factory python=3.11  
conda activate llama-factory  

git clone https://github.com/hiyouga/LLaMA-Factory.git  
cd LLaMA-Factory  
pip install -e ".[torch,metrics]"  
pip install deepspeed==0.16.2  
pip install liger-kernel
```

### GRPO via TRL
#### Environment Setup
```
conda create -n grpo python=3.11
conda activate grpo
pip install openreview-py
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

pip install --user -U nltk

pip install deepspeed

pip install trl[peft]
pip install bitsandbytes loralib
pip install git+https://github.com/huggingface/transformers.git@main

pip install git+https://github.com/huggingface/accelerate.git@main

pip install git+https://github.com/huggingface/trl.git
```

#### Launching TRL training scripts
```
conda activate grpo
accelerate launch trl_iclr_90th_u.py
``` 

## Accessing Data from Huggingface.co
Below is a simple snippet to access the dataset via huggingface datasets.
```
from datasets import load_dataset

ds = load_dataset("pawin205/iclr-2017-2020-peer-review-with-thinking-trace")
```

## Accessing Models from Huggingface.co
Below is a simple snippet to load the model via huggingface.
```
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("pawin205/Qwen-7B-Review-ICLR-GRPO-U")
model = AutoModelForCausalLM.from_pretrained("pawin205/Qwen-7B-Review-ICLR-GRPO-U")
```