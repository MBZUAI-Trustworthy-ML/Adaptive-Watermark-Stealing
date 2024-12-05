import torch
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import trl
from typing import List
import random
import pandas as pd
from datasets import Dataset, load_from_disk, load_dataset
import openai

# Device configuration
device = "cuda:7" if torch.cuda.is_available() else "cpu"

# Transformers configuration
def load_transformers_config(model_name: str) -> TransformersConfig:
    return TransformersConfig(
        model=AutoModelForCausalLM.from_pretrained(model_name).to(device),
        tokenizer=AutoTokenizer.from_pretrained(model_name),
        vocab_size=50272,
        device=device,
        max_new_tokens=200,
        min_length=200,
        do_sample=True,
        no_repeat_ngram_size=4
    )

# Load watermark algorithm
def load_watermark(transformers_config: TransformersConfig) -> AutoWatermark:
    return AutoWatermark.load(
        'KGW', 
        algorithm_config='config/KGW.json',
        transformers_config=transformers_config
    )

# Load dataset
def load_goal_dataset(filepath: str) -> List[str]:
    with open(filepath, 'r') as file:
        return json.load(file)['goals']

# Surrogate model generation
def generate_watermarked_and_unwatermarked_text(prompt: str, num_samples: int, model_name: str) -> (List[str], str):
    transformers_config = load_transformers_config(model_name)
    watermark_instance = load_watermark(transformers_config)
    watermarked_texts = [watermark_instance.generate_watermarked_text(prompt) for _ in range(num_samples)]
    unwatermarked_text = watermark_instance.generate_unwatermarked_text(prompt)
    return watermarked_texts, unwatermarked_text

# Generate watermarked text using target surrogate model
def generate_target_watermarked_text(prompt: str, num_samples: int, model_name: str) -> List[str]:
    transformers_config = load_transformers_config(model_name)
    watermark_instance = load_watermark(transformers_config)
    return [watermark_instance.generate_watermarked_text(prompt) for _ in range(num_samples)]

# Generate unwatermarked text using attacker surrogate model
def generate_unwatermarked_text(prompt: str, model_name: str) -> str:
    transformers_config = load_transformers_config(model_name)
    watermark_instance = load_watermark(transformers_config)
    return watermark_instance.generate_unwatermarked_text(prompt)

# Save responses to file
def save_json(data: dict, directory: str, filename: str):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, filename), 'w') as file:
        json.dump(data, file, indent=4)

# Paraphrase prompt to make it non-toxic
def generate_non_toxic_paraphrase(prompt: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": f"Paraphrase the following text to make it non-toxic:\nText: {prompt}\nParaphrased (non-toxic) text:"}
        ],
        max_tokens=200,
        temperature=0.7
    )
    return response['choices'][0]['message']['content'].strip()

# Paraphrase text using GPT model
def paraphrase_using_gpt(original_text: str, reference_texts: List[str], num_samples: int) -> List[str]:
    system_prompt = (
        "You are an expert copywriter. Paraphrase the given original text below using copy-pastes from the given reference texts.\n"
        "Ensure that the paraphrased output has a similar meaning as the original text.\n"
        "Original text: {original_text}\n"
        "Reference texts separated by ' | ': {reference_texts}"
    )
    paraphrased_texts = []

    for _ in range(num_samples):
        prompt = system_prompt.format(
            original_text=original_text,
            reference_texts=" | ".join(reference_texts)
        )
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        paraphrased_text = response['choices'][0]['message']['content'].strip()
        paraphrased_texts.append(paraphrased_text)

    return paraphrased_texts

# Paraphrase text using Hugging Face model
def paraphrase_using_hf_model(original_text: str, reference_texts: List[str], num_samples: int, model_name: str) -> List[str]:
    transformers_config = load_transformers_config(model_name)
    tokenizer = transformers_config.tokenizer
    model = transformers_config.model
    paraphrased_texts = []

    for _ in range(num_samples):
        prompt = (
            "You are an expert copywriter. Paraphrase the given original text below using copy-pastes from the given reference texts.\n"
            "Ensure that the paraphrased output has a similar meaning as the original text.\n"
            f"Original text: {original_text}\n"
            f"Reference texts separated by ' | ': {' | '.join(reference_texts)}"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                min_length=200,
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True
            )
        paraphrased_text = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
        paraphrased_texts.append(paraphrased_text)

    return paraphrased_texts

# Load and process prompt dataset
def process_prompt_dataset(dataset_path: str, num_prompts: int):
    prompts = load_goal_dataset(dataset_path)[:num_prompts]
    save_directory = os.path.dirname(dataset_path)

    paraphrased_results = []
    dpo_training_data = []
    sft_training_data = []
    combined_responses = []
    similarity_threshold = 0.8

    for prompt in prompts:
        # Generate paraphrased prompt to make it non-toxic
        prompt = generate_non_toxic_paraphrase(prompt)

        # Generate watermarked and unwatermarked text
        watermarked_responses, unwatermarked_response = generate_watermarked_and_unwatermarked_text(prompt, num_samples=3, model_name='Qwen/Qwen2.5-3B')

        # Generate paraphrased responses
        paraphrased_responses = paraphrase_using_hf_model(unwatermarked_response, watermarked_responses, num_samples=64, model_name='Qwen/Qwen2.5-3B-Instruct')

        # Evaluate paraphrases
        evaluation_results = evaluate_paraphrases(unwatermarked_response, paraphrased_responses)

        for paraphrased, evaluation in zip(paraphrased_responses, evaluation_results):
            combined_response = f"similarity_score: {evaluation['similarity_score']} | is_watermarked: {evaluation['is_watermarked']} | {paraphrased}"
            combined_responses.append(combined_response)

        # Collect results
        paraphrased_results.append({
            'prompt': prompt,
            'target_response': watermarked_responses,
            'attacker_response': unwatermarked_response,
            'paraphrased_responses': combined_responses
        })

        # Create training data for DPO
        selected_paraphrases = [paraphrase for paraphrase, eval_result in zip(paraphrased_responses, evaluation_results) if eval_result['similarity_score'] > similarity_threshold and eval_result['is_watermarked']]
        rejected_paraphrases = [paraphrase for paraphrase, eval_result in zip(paraphrased_responses, evaluation_results) if eval_result['similarity_score'] <= similarity_threshold or not eval_result['is_watermarked']]

        instruction = (f"you are a brilliant assistant. Paraphrase the text below: \n"
                       f"{unwatermarked_response}"
        )

        if selected_paraphrases and rejected_paraphrases:
            selected_paraphrase = random.choice(selected_paraphrases)
            rejected_paraphrase = random.choice(rejected_paraphrases)
            dpo_training_data.append({
                'prompt': instruction,
                'chosen': selected_paraphrase,
                'rejected': rejected_paraphrase
            })
            sft_training_data.append({
                'prompt': instruction, 'completion': selected_paraphrase
            })

    # Save paraphrased results
    save_json({'results': paraphrased_results}, save_directory, 'results.json')

    # Save SFT training dataset
    save_json(sft_training_data, save_directory, 'sft_dataset.jsonl')

    # Save DPO training dataset
    dpo_dataset = Dataset.from_pandas(pd.DataFrame(dpo_training_data))
    dpo_dataset.save_to_disk(os.path.join(save_directory, 'dpo_dataset'))
    print(f"Created DPO dataset with {len(dpo_training_data)} samples")

# Joint training function for DPO and SFT
def train_model(dataset_path: str, model_name: str, training_type: str, num_epochs: int = 3, batch_size: int = 4):
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if training_type == "dpo":
        dpo_dataset = load_from_disk(dataset_path)
        training_args = trl.DPOConfig(output_dir="dpo_finetuned_model", logging_steps=10)
        trainer = trl.DPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=dpo_dataset)
    elif training_type == "sft":
        train_dataset = load_dataset("json", data_files=dataset_path)
        training_args = trl.SFTTrainingArguments(
            output_dir="sft_finetuned_model",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=8,
            logging_steps=10,
            save_steps=1000,
            learning_rate=5e-5,
            weight_decay=0.01
        )
        trainer = trl.SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer
        )
    else:
        raise ValueError("Invalid training_type. Must be either 'dpo' or 'sft'.")

    trainer.train()
    trainer.save_model(training_args.output_dir)
    print(f"Training completed and model saved to {training_args.output_dir}")

# Evaluate paraphrases
def evaluate_paraphrases(original_text: str, paraphrases: List[str]) -> List[dict]:
    evaluation_metrics = []
    for paraphrase in paraphrases:
        similarity_score = evaluate_text_similarity(original_text, paraphrase)
        watermark_detector = load_watermark(load_transformers_config('Qwen/Qwen2.5-3B'))
        is_watermarked = watermark_detector.detect_watermark(paraphrase)['is_watermarked']
        evaluation_metrics.append({
            'similarity_score': similarity_score,
            'is_watermarked': is_watermarked
        })
    return evaluation_metrics

# Evaluate semantic similarity between two texts
def evaluate_text_similarity(text1: str, text2: str) -> float:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": f"Rate the semantic similarity between the following two texts on a scale from 0 to 1:\nText 1: {text1}\nText 2: {text2}\nSimilarity score:"}
        ],
        max_tokens=4
    )
    return float(response['choices'][0]['message']['content'].strip())

# Evaluate text toxicity
def evaluate_text_toxicity(text: str) -> float:
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": f"Rate the toxicity of the following text on a scale from 0 to 1:\nText: {text}\nToxicity score:"}
        ],
        max_tokens=4
    )
    return float(response['choices'][0]['message']['content'].strip())

if __name__ == "__main__":
    # Ensure to export your OpenAI key before running this script.
    # Example: export OPENAI_API_KEY='your_openai_api_key'
    dataset_path = '/ephemeral/taremu/MarkLLM/dataset/advbench/advbench_subset.json'
    process_prompt_dataset(dataset_path, num_prompts=20)

    # dataset_path = ''
    # train_model(dataset_path=dataset_path, model_name='Qwen/Qwen2.5-1.5B', training_type='dpo')
    # train_model(dataset_path=dataset_path, model_name='Qwen/Qwen2.5-1.5B', training_type='sft')
