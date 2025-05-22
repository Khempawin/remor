# Install LLaMA-Factory
Follow the guide in [LLaMA-Factory GitHub](https://github.com/hiyouga/LLaMA-Factory)

# Replace Dataset Info configuration
Replace the file "data/dataset_info.json" with the "dataset_info.json" in this directory

# Add training files and merging LoRA weight configuration
Copy "qwen_7b_iclr_90th_percentile.yaml" and "qwen_7b_iclr_90th_merge_lora_sft.yaml" to the "example" directory found in the cloned LLaMA-Factory directory.

# Training
At the directory of the cloned LLaMA-Factory directory. Activate the python environment with LLaMA-Factory installed. Then execute the following command:
```
llamafactory-cli train examples/qwen_7b_iclr_90th_percentile.yaml
```

# Exporting model with merged LoRA weights
After the training is done, merge the LoRA weights to get the standalone model with the following command:
```
llamafactory-cli export examples/qwen_7b_iclr_90th_merge_lora_sft.yaml
```

The final model can be found at "output/Qwen-7B-Review-ICLR-90th-sft" inside the cloned LLaMA-Factory directory.