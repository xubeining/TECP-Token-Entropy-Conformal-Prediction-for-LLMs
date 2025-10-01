import argparse
import json
import random
import os
import numpy as np
import tqdm
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--generate-model', default='./your_model', help='Local path of generative LLM')
parser.add_argument('--row-data-path', default='./coqa-train-v1.0.json', help='Path to CoQA json file')
parser.add_argument('--data-dir', default='./coqa_dataset_output', help='Output dir for processed dataset')
parser.add_argument('--cache-dir', default='./cache', help='Cache dir for Hugging Face')
parser.add_argument('--few-shot-num', type=int, default=3, help='Number of few-shot examples')
parser.add_argument('--max-num', type=int, default=1000, help='Max number of examples to include')
parser.add_argument('--include-passage', action='store_true', help='Include story context in prompt')
args = parser.parse_args()
# ----------------------------------------------------------------------------------------------------------------------
# model_name for naming
model_name = os.path.basename(args.generate_model.rstrip('/\\'))
print('Generative LLM: ', model_name)

# Seed
seed_value = 10
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
os.environ["HF_DATASETS_CACHE"] = args.cache_dir

# Load tokenizer & model
tokenizer = AutoTokenizer.from_pretrained(args.generate_model, local_files_only=True, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(args.generate_model, local_files_only=True, torch_dtype=torch.float16, device_map="auto")
max_input_ids_length = model.config.max_position_embeddings
print('LLM max input ids length: ', max_input_ids_length)
# ----------------------------------------------------------------------------------------------------------------------
# Load CoQA json
with open(args.row_data_path, 'r', encoding='utf-8') as f:
    coqa_data = json.load(f)

dict_list = []
for entry in coqa_data['data']:
    story = entry['story']
    for q, a in zip(entry['questions'], entry['answers']):
        dict_list.append({
            'question': q['input_text'],
            'answer': {'value': a['input_text']},
            'question_id': q['turn_id'],
            'story': story
        })

print(f"Loaded {len(dict_list)} QAs from CoQA")
# ----------------------------------------------------------------------------------------------------------------------
# Instruction and few-shot
instruction = "### System:\nThis is a bot that correctly answers questions.\n\n"
idx_for_few_shot_prompt = random.sample(range(0, len(dict_list)), args.few_shot_num)
few_shot_prompt = ''
for idx in idx_for_few_shot_prompt:
    q = dict_list[idx]['question']
    a = dict_list[idx]['answer']['value']
    story = dict_list[idx]['story']
    if q.isascii() and a.isascii():
        if args.include_passage:
            few_shot_prompt += f'### Context:\n{story}\n### User:\n{q}\n### Assistant:\n{a}\n\n'
        else:
            few_shot_prompt += f'### User:\n{q}\n### Assistant:\n{a}\n\n'
# ----------------------------------------------------------------------------------------------------------------------
dataset = {'prompt': [], 'question': [], 'answer': [], 'id': []}
applied_qa_pairs = 0
for i, sample in enumerate(tqdm.tqdm(dict_list)):
    if i in idx_for_few_shot_prompt:
        continue
    q = sample['question']
    a = sample['answer']['value']
    story = sample['story']
    if q.isascii() and a.isascii():
        if args.include_passage:
            full_prompt = instruction + few_shot_prompt + f'### Context:\n{story}\n### User:\n{q}\n### Assistant:\n'
        else:
            full_prompt = instruction + few_shot_prompt + f'### User:\n{q}\n### Assistant:\n'

        input_ids = tokenizer.encode(full_prompt)
        if len(input_ids) < max_input_ids_length:
            dataset['prompt'].append(full_prompt)
            dataset['question'].append(q)
            dataset['answer'].append(a)
            dataset['id'].append(f"{sample['question_id']}_{i}")
            applied_qa_pairs += 1

        if applied_qa_pairs >= args.max_num:
            break

print('Applied question-answer pairs:', applied_qa_pairs)
# ----------------------------------------------------------------------------------------------------------------------
# Save
os.makedirs(args.data_dir, exist_ok=True)
dataset_df = pd.DataFrame.from_dict(dataset)
dataset_hf = Dataset.from_pandas(dataset_df)
dataset_hf.save_to_disk(f"{args.data_dir}/coqa_{model_name}")
print(f"Saved to {args.data_dir}/coqa_{model_name}")

# Check answer length
lens = [len(tokenizer.encode(a)) for a in dataset['answer']]
print("Max answer token length:", max(lens))
