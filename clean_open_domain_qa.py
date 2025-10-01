import argparse
import pathlib
import pickle
import tqdm
import os
import random
import json

import accelerate
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='../datasets', help='save parsed dataset')
parser.add_argument('--cache-dir', default='../cache', help='cache model from hugging face')
parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
parser.add_argument('--record-dir', default='../records', help='save experimental records')
parser.add_argument('--generate-model', default=r'D:\projects\cache_model\Qwen2.5-14B-Instruct', help='local path of generative llm')
parser.add_argument('--dataset', default='coqa')
parser.add_argument('--max-length-of-generation', type=int, default=128)
parser.add_argument('--sample', type=bool, default=True, help='sampled or the most likely')
parser.add_argument('--num-beams', type=int, default=5, help='for the most likely generation')
parser.add_argument('--num-generations-per-prompt', type=int, default=10, help='for sampling')
parser.add_argument('--top-p', type=float, default=0.9, help='for sampling')
parser.add_argument('--temperature', type=float, default=1.0, help='for sampling')
args = parser.parse_args()

model_name = args.generate_model.split('\\')[-1] if '\\' in args.generate_model else args.generate_model
if args.dataset in ['coqa', 'triviaqa']:
    args.max_length_of_generation = 36
if args.sample:
    run_name = os.path.join(args.record_dir, args.dataset, model_name,
                            'num_generations-' + str(args.num_generations_per_prompt),
                            'temperature-' + str(args.temperature),
                            'max_len_of_generation-' + str(args.max_length_of_generation))
else:
    run_name = os.path.join(args.record_dir, args.dataset, model_name,
                            'num_beams-' + str(args.num_beams),
                            'max_len_of_generation-' + str(args.max_length_of_generation))

seed_value = 10
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
os.environ["HF_DATASETS_CACHE"] = args.cache_dir
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

with open(f'{run_name}/generations.pkl', 'rb') as record_file:
    generations = pickle.load(record_file)

def filter(text):
    text = text.strip()
    strings_to_filter_on = ['\n', '.', '#']
    while '.' in text:
        point_idx = text.index('.')
        if point_idx != len(text) - 1 and text[point_idx - 1].isdigit() and text[point_idx + 1].isdigit():
            text = text.replace('.', '(dp)', 1)
        else:
            break
    for string in strings_to_filter_on:
        if string in text:
            text = text.split(string)[0]
    if '(dp)' in text:
        text = text.replace('(dp)', '.')
    new_text = ''
    if len(text) != 0:
        for ch in text:
            if ch.isascii():
                new_text += ch
    if len(new_text) == 0:
        new_text = 'error'
    return new_text

error_sampled_generations = 0
cleaned_generations = []

for generation in tqdm.tqdm(generations):
    cleaned_gens = []
    for sampled_generation in generation['sampled_generated_texts']:
        cleaned_gen = filter(sampled_generation)
        cleaned_gens.append(cleaned_gen)
        if cleaned_gen == 'error':
            error_sampled_generations += 1
        print(cleaned_gen)

    cleaned_generation = {
        'id': generation.get('id'),
        'prompt': generation.get('prompt'),
        'question': generation.get('question'),
        'answer': generation.get('answer'),
        'sampled_generated_texts': cleaned_gens
    }

    if 'options' in generation:
        cleaned_generation['options'] = generation['options']
    if 'category' in generation:
        cleaned_generation['category'] = generation['category']

    cleaned_generations.append(cleaned_generation)

print('error sampled generations: ', error_sampled_generations)

with open(f'{run_name}/cleaned_generations.pkl', 'wb') as record_file:
    pickle.dump(cleaned_generations, record_file)
print('Record saved to ', f'{run_name}/cleaned_generations.pkl')
