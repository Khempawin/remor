# REMOR : Automated Peer Review Generation with LLM Reasoning and Multi-Objective Reinforcement Learning

## Notes
- distilbert_models can be downloaded from [Link](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3002238)

## Loading reward function
### Environment Setup


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


### GRPO via TRL
#### Environment Setup

## Accessing Data from Huggingface.co

## Accessing Models from Huggingface.co
