from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline, pipeline, AutoModelForCausalLM, Qwen2ForCausalLM
from typing import List, TypedDict, Literal
from importlib.metadata import version
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import sent_tokenize

import re
import nltk
import torch


print("transformers version: {}".format(version('transformers')))
print("pandas version: {}".format(version('pandas')))


nltk.download("punkt_tab")
nltk.download('wordnet')
nltk.download('omw-1.4')


class ReviewSentenceCriteriaCount(TypedDict):
    criticism: int
    example: int
    importance_and_relevance: int
    materials_and_methods: int
    praise: int
    presentation_and_reporting: int
    results_and_discussion: int
    suggestion_and_solution: int
    total: int

# Load aspect classifier models
JIF_MODELS = [
        "criticism",
        "example",
        "importance_and_relevance",
        "materials_and_methods",
        "praise",
        "presentation_and_reporting",
        "results_and_discussion",
        "suggestion_and_solution"
    ]

## load tokenizer (same for all models handled here)
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

## load classifier models
classifier_dict = dict()

for model_idx in JIF_MODELS:
    current_model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert_models/" + model_idx, num_labels = 2
    )
    classifier:TextClassificationPipeline = pipeline(
        "text-classification",
        model=current_model,
        tokenizer=tokenizer,
        device="cuda:0"
    )
    classifier_dict[model_idx] = classifier
    
def classify_review_sentences(sentences: List[str]) -> ReviewSentenceCriteriaCount:
    criteria_count = ReviewSentenceCriteriaCount(
        criticism=0,
        example=0,
        importance_and_relevance=0,
        materials_and_methods=0,
        praise=0,
        presentation_and_reporting=0,
        results_and_discussion=0,
        suggestion_and_solution=0,
        total=0
    )
    
    tokenizer_kwarg= { "padding":True, "truncation":True, "max_length":512 }

    for model_idx in JIF_MODELS:
        # run classification pipeline
        results = classifier_dict[model_idx](sentences, **tokenizer_kwarg)
        positive_criteria_count = len(list(filter(lambda x: x["label"] == "LABEL_1", results)))
        criteria_count[model_idx] = positive_criteria_count
    criteria_count["total"] = len(sentences)
    
    return criteria_count


def calculate_meteor_score_review_full_text(review, full_text):
    if(full_text == "ERROR"):
        return 0.0

    # Tokenize your inputs
    candidate = nltk.word_tokenize(review)

    reference_list = [nltk.word_tokenize(full_text)]

    # Calculate METEOR score
    return meteor_score(reference_list, candidate)


def calc_reward(sentence_criteria_count: ReviewSentenceCriteriaCount) -> float:
    if (sentence_criteria_count["total"] == 0):
        return -10
    reward = 0
    for metric in JIF_MODELS:
        reward += sentence_criteria_count[metric] / sentence_criteria_count["total"]
    return reward


def calc_ratio_with_aspect(sentences: List[str], aspect: Literal["criticism",
                                                                            "example",
                                                                            "importance_and_relevance",
                                                                            "materials_and_methods",
                                                                            "praise",
                                                                            "presentation_and_reporting",
                                                                            "results_and_discussion",
                                                                            "suggestion_and_solution"]
    ) -> float:

    if(len(sentences) == 0):
        return -2

    tokenizer_kwarg= { "padding":True, "truncation":True, "max_length":512 }
    classification_result = classifier_dict[aspect](sentences, **tokenizer_kwarg)
    ratio = len(list(filter(lambda x: x["label"] == "LABEL_1", classification_result))) / float(len(sentences))
    
    return ratio


def reward_aspect(completions: List[str], aspect: Literal["criticism",
                                                                            "example",
                                                                            "importance_and_relevance",
                                                                            "materials_and_methods",
                                                                            "praise",
                                                                            "presentation_and_reporting",
                                                                            "results_and_discussion",
                                                                            "suggestion_and_solution"]
    ):

    # Get review after thinking traces
    reviews = [review.split("</think>") for review in completions]
    reviews = [review[1].strip() if len(review) > 1 else review[0].strip() for review in reviews]

    # Tokenize each review at sentences level
    tokenized_sentences = [sent_tokenize(review) for review in reviews]
    
    # Classify sentences and calculate reward for each review
    rewards = [calc_ratio_with_aspect(sentences, aspect) for sentences in tokenized_sentences]
    
    return rewards


def reward_all_aspects_of_review(completions: List[str], **kwargs) -> List[float]:
    # Tokenize each review at sentences level
    tokenized_sentences = [sent_tokenize(review) for review in completions]
    
    # Classify sentences of each review
    classification_results = [classify_review_sentences(sentences) for sentences in tokenized_sentences]
    
    # Calculate reward
    rewards = [calc_reward(sentence_criteria_count) for sentence_criteria_count in classification_results]
    
    #rewards = [math.sqrt((25 - len(sentences))**2/64) * -1 for sentences in tokenized_sentences]

    return rewards


def reward_criticism(completions: List[str], **kwargs) -> List[float]:
    return reward_aspect(completions, "criticism")


def reward_example(completions: List[str], **kwargs) -> List[float]:
    return reward_aspect(completions, "example")


def reward_importance_and_relevance(completions: List[str], **kwargs) -> List[float]:
    return reward_aspect(completions, "importance_and_relevance")


def reward_materials_and_methods(completions: List[str], **kwargs) -> List[float]:
    return reward_aspect(completions, "materials_and_methods")


def reward_praise(completions: List[str], **kwargs) -> List[float]:
    return reward_aspect(completions, "praise")


def reward_presentation_and_reporting(completions: List[str], **kwargs) -> List[float]:
    return reward_aspect(completions, "presentation_and_reporting")


def reward_results_and_discussion(completions: List[str], **kwargs) -> List[float]:
    return reward_aspect(completions, "results_and_discussion")


def reward_suggestions_and_solution(completions: List[str], **kwargs) -> List[float]:
    return reward_aspect(completions, "suggestion_and_solution")


def reward_thinking_trace(completions: List[str], **kwargs) -> List[float]:
    rewards = [0.5 if re.match(r"^<think>.+</think>\n.+", review) else 0 for review in completions]
    return rewards


def reward_coverage(completions: List[str], **kwargs) -> List[float]:
    rewards = [0] * len(completions)
    return rewards


def reward_meteor(completions: List[str], **kwargs):
    # Get review after thinking traces
    reviews = [review.split("</think>") for review in completions]
    reviews = [review[1] if len(review) > 1 else review[0] for review in reviews]

    # Calculate meteor score compared to the full text
    rewards = [calculate_meteor_score_review_full_text(review, full_text) for review, full_text  in zip(reviews, kwargs["full_text"])]

    return rewards


model_path = "pawin205/Qwen-7B-Review-ICLR-90th-sft"

#dataset = load_dataset("pawin205/paper-review-pair", split="train")
dataset = load_dataset("pawin205/iclr-2017-2020-peer-review-with-thinking-trace", split="90thPercentile")

model = Qwen2ForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
)


#model = get_peft_model(model, peft_config)
#print(model)
#print(model.print_trainable_parameters())

reward_tokenizer = AutoTokenizer.from_pretrained(model_path)
reward_tokenizer.pad_token = tokenizer.eos_token

training_args = GRPOConfig(
        output_dir="saves/Qwen-Review-7B-ICLR-GRPO-U",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=1,
        lr_scheduler_type='cosine',
        logging_steps=1,
        bf16=True,
        num_generations=4,
        max_prompt_length=12288,
        max_completion_length=4096,
        num_train_epochs=1,
        save_steps=200,
        use_vllm=False,
        vllm_gpu_memory_utilization=0.7,
        gradient_checkpointing=True,
        )

trainer = GRPOTrainer(
    model=model,
    processing_class=reward_tokenizer,
    reward_funcs=[
        # reward_all_aspects_of_review,
        reward_criticism,
        reward_example,
        reward_importance_and_relevance,
        reward_materials_and_methods,
        reward_praise,
        reward_presentation_and_reporting,
        reward_results_and_discussion,
        reward_suggestions_and_solution,
        reward_meteor,
        #reward_thinking_trace
        ],
    args=training_args,
    train_dataset=dataset,
)
trainer.train()

trainer.save_model(training_args.output_dir)

