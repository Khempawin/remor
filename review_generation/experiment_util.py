import nltk
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from nltk.tokenize import sent_tokenize
from nltk.translate.meteor_score import meteor_score
from typing import List, TypedDict, Literal
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline, pipeline
from ollama import chat
from tqdm import tqdm

from importlib.metadata import version

version('transformers')
version('pandas')

nltk.download("punkt_tab")

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


class ReviewResult(TypedDict):
    title: str
    abstract: str
    full_text: str
    review: str

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

classifier_dict = dict()

# Load classifier models to reduce loading overhead but will consume more VRAM
for model_idx in JIF_MODELS:
    current_model = AutoModelForSequenceClassification.from_pretrained(
        f"distilbert_models/{model_idx}", num_labels=2
    )
    classifer:TextClassificationPipeline = pipeline(
        "text-classification",
        model=current_model,
        tokenizer=tokenizer
    )
    classifier_dict[model_idx] = classifer
    

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


def create_user_message(paper_content, title, abstract):
    prompt_template = "You are a member of the scientific community tasked with peer review. Review the following paper content.\n\n### Paper Content\n\n{paper_content}"
    
    if (paper_content == "ERROR"):
        paper_content = f"""Title: {title}\n\nAbstract: {abstract}"""
        
    paper_content = paper_content.replace("Title: No title found", f"Title: {title}").replace("Abstract: No abstract found", f"Abstract: {abstract}")
    
    return prompt_template.format(paper_content=paper_content)


def classify_review_sentences_legacy(sentences: List[str]) -> ReviewSentenceCriteriaCount:
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
        current_model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert_models/" + model_idx, num_labels = 2)
        # run classification pipeline
        classifier = pipeline("text-classification",
                                model = current_model,
                                tokenizer = tokenizer)
        results = classifier(sentences, **tokenizer_kwarg)
        positive_criteria_count = len(list(filter(lambda x: x["label"] == "LABEL_1", results)))
        criteria_count[model_idx] = positive_criteria_count
    criteria_count["total"] = len(sentences)
    
    return criteria_count


def reward_function(row):
    reward = 0
    for metric in JIF_MODELS:
        reward += row[f"{metric}"]
    
    return reward


def calculate_reward_u(row) -> float:
    metrics_names = [
        "criticism",
        "example",
        "importance_and_relevance",
        "materials_and_methods",
        "praise",
        "presentation_and_reporting",
        "results_and_discussion",
        "suggestion_and_solution",
        "meteor_score"
    ]
    score = 0
    for metric in metrics_names:
        score += row[metric]
    return score


def calculate_reward_h(row) -> float:
    metrics_weights = [
        ("criticism", 0.01),
        ("example", 0.01),
        ("importance_and_relevance", 0.11),
        ("materials_and_methods", 0.01),
        ("praise", 0.01),
        ("presentation_and_reporting", 0.01),
        ("results_and_discussion", 0.01),
        ("suggestion_and_solution", 0.16),
        ("meteor_score", 8.67),
    ]
    score = 0
    for metric, weight in metrics_weights:
        score += row[metric] * weight
    return score


def length_penalty(length: int):
    mean_of_sentence_count = 13
    sentence_count_std = 7.96
    return ((length - mean_of_sentence_count)**2)/(sentence_count_std**2)


def reward_function_length_adjusted(row):
    mean_of_sentence_count = 13
    sentence_count_std = 7.96
    reward = 0
    for metric in JIF_MODELS:
        reward += row[f"{metric}"]
    length_reward = ((row["total"] - mean_of_sentence_count)**2)/(sentence_count_std**2)
    reward -= length_reward
    return reward


def calculate_meteor_score(review: str, full_text: str) -> float:
    # Tokenize your inputs
    candidate = nltk.word_tokenize(review)

    reference_list = [nltk.word_tokenize(full_text)]

    # Calculate METEOR score
    return meteor_score(reference_list, candidate)


llm_api_base = "http://localhost:11434"

def generate_review(row, model_name: str) -> str:
    response = chat(model_name, row["conversations"][:1])
    return response["message"]["content"]


def generate_review_from_dataframe(paper_df: pd.DataFrame, model_name: str, save_file_path="review_save.pickle", save_to_file=True) -> List[ReviewResult]:
    assert not (save_to_file == True and save_file_path == ""), "Save file path must be provided to save the results"
    
    generated_reviews_list = list()
    rows = [v for i, v in paper_df.drop_duplicates(["title"]).iterrows()]
    for row in tqdm(rows):
        res = ReviewResult(
            title=row["title"],
            abstract=row["abstract"],
            review=generate_review(row, model_name),
            full_text=row["full_text"]
        )
        generated_reviews_list.append(res)
    
    if(save_to_file == True):
        with open(save_file_path, "wb") as f:
            pickle.dump(generated_reviews_list, f)
    
    return generated_reviews_list


def prepare_reviews_for_analysis(reviews: List[ReviewResult]) -> pd.DataFrame:
    tqdm.pandas()
    # Tokenize each review into a list of sentences
    sentence_tokenized_reviews = [{
        "title": review["title"],
        "abstract": review["abstract"],
        "full_text": review["full_text"],
        "review": review["review"],
        "review_sentences": sent_tokenize(review["review"])
    } for review in reviews]
    
    # Create a dataframe from the data
    review_df = pd.DataFrame(sentence_tokenized_reviews)
    
    # Classify each review sentences
    review_df["criteria_count"] = review_df.progress_apply(lambda row: classify_review_sentences(row["review_sentences"]), axis=1)
    
    # Expand classification count to columns
    for criteria in JIF_MODELS + ["total"]:
        review_df[criteria] = review_df.apply(lambda row: row["criteria_count"][criteria], axis=1)
    
    # Normalize result
    for criteria in JIF_MODELS:
        review_df[criteria] = review_df.apply(lambda row: row["criteria_count"][criteria] / row["total"] if row["total"] != 0 else 0, axis=1)
        
    # Determine if criteria is present in review based on classification count
    for criteria in JIF_MODELS:
        review_df[f"has_{criteria}"] = review_df.apply(lambda row: (row["criteria_count"][criteria] != 0) * 1, axis=1)
        
    # Calculate ReMe (METEOR Score)
    review_df["meteor_score"] = review_df.apply(lambda row: calculate_meteor_score(row["review"], row["full_text"]), axis=1)
        
    review_df["reward_value"] = review_df.apply(lambda row: reward_function(row), axis=1)
    review_df["reward_value_length_adjusted"] = review_df.apply(lambda row: reward_function_length_adjusted(row), axis=1)
    review_df["length_penalty"] = review_df.apply(lambda row: length_penalty(row["total"]), axis=1)
    review_df["reward_u"] = review_df.apply(lambda row: calculate_reward_u(row), axis=1)
    review_df["reward_h"] = review_df.apply(lambda row: calculate_reward_h(row), axis=1)
    
    return review_df


def calculate_std_error(data) -> float:
    return np.std(data, ddof=1) / np.sqrt(np.size(data))


def plot_metric_distributions(data: pd.DataFrame):
    fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(10, 20))

    for i, metric in enumerate(JIF_MODELS):
        row = int(i / 2)
        col = i % 2
        sns.kdeplot(data[metric], fill=True, ax=ax[row][col])
        ax[row][col].set_xlabel(f"{metric} MEAN={np.mean(data[metric]):.3f}, SEM={calculate_std_error(data[metric]):.3f}\nSTD={np.std(data[metric]):.3f}")
        ax[row][col].set_ylabel("density")
        
    return


def plot_reward_value_distribution(data: pd.DataFrame, model_name: str):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

    ax[0].hist(data["reward_value"], bins=40)
    ax[0].set_title(f"Distribution of Reward Score of Generated reviews from {model_name}")
    ax[0].set_xlabel("Reward Score")
    ax[0].set_ylabel("Count")

    ax[1].hist(data["reward_value_length_adjusted"], bins=40)
    ax[1].set_title(f"Distribution of Reward Score of Generated reviews from {model_name} (length penalized)")
    ax[1].set_xlabel("Reward Score")
    ax[1].set_ylabel("Count")

    print(data["total"].mean(), data["total"].std())
    return


def plot_reward_value_density(data: pd.DataFrame):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

    metric = "reward_value"
    sns.kdeplot(data[metric], fill=True, ax=ax)
    ax.set_xlabel(metric)
    ax.set_xlabel(f"{metric} MEAN={np.mean(data[metric]):.3f}, SEM={calculate_std_error(data[metric]):.3f}\nSTD={np.std(data[metric]):.3f}")
    ax.set_ylabel("density")
    
    return


def plot_review_length_distribution(data: pd.DataFrame, model_name: str):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

    ax.hist(data["total"], bins=40)
    ax.set_title(f"Distribution of Length of Generated reviews from {model_name}")
    ax.set_xlabel("Length")
    ax.set_ylabel("Count")
    
    return