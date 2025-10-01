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
parser.add_argument('--split-ratio', type=float, default=0.5, help='for splitting calibration and test set')
parser.add_argument('--correctness-threshold', type=float, default=0.7, help='for correctness evaluation')
parser.add_argument('--alpha', type=float, default=0.2, help='risk level')
parser.add_argument('--seed', type=int, default=10, help='random seed for reproducibility')
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

seed_value = args.seed
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
os.environ["HF_DATASETS_CACHE"] = args.cache_dir
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

with open(f'{run_name}/cleaned_generations.pkl', 'rb') as record_file:
    generations = pickle.load(record_file)
with open(f'{run_name}/similarity_scores.pkl', 'rb') as record_file:
    similarity_scores = pickle.load(record_file)

likelihoods_path = f'{run_name}/generations_likelihoods.pkl'
if not os.path.exists(likelihoods_path):
    raise FileNotFoundError(f"Missing {likelihoods_path}")
with open(likelihoods_path, 'rb') as f:
    likelihoods = pickle.load(f)

def _norm_id(x):
    return x[0] if isinstance(x, list) else x
likelihoods_by_id = {_norm_id(e['id']): e for e in likelihoods}
similarity_dict, similarity_for_correctness = similarity_scores

applied_generations = []
for generation in generations:
    id = generation['id']
    similarity_for_correctness_list = similarity_for_correctness[id]
    if max(similarity_for_correctness_list) >= args.correctness_threshold:
        applied_generations.append(generation)

total_num = len(applied_generations)
num_cal = int(total_num * args.split_ratio)
calibration_set = random.sample(applied_generations, num_cal)
test_set = [generation for generation in applied_generations if generation not in calibration_set]
print('Applied num: ', total_num)
print('Calibration num: ', len(calibration_set))
print('Test num: ', len(test_set))

def compute_entropies(likeli_entry):
    return [float(torch.tensor(ent).sum()) for ent in likeli_entry['token_wise_entropy']]

nonconformity_scores = []
for cal_data in tqdm.tqdm(calibration_set):
    cal_id = cal_data['id']
    entropies = compute_entropies(likelihoods_by_id[cal_id])
    nonconformity_scores.extend(entropies)

N = len(nonconformity_scores)
q_level = np.ceil((N + 1) * (1 - args.alpha)) / N
q_hat = np.quantile(nonconformity_scores, q_level, method='higher')

miscoverage_num = 0
total_set_size = 0

for test_data in tqdm.tqdm(test_set):
    test_id = test_data['id']
    entropies = compute_entropies(likelihoods_by_id[test_id])
    prediction_set = [i for i, ent in enumerate(entropies) if ent <= q_hat]
    total_set_size += len(prediction_set)

    correct_list = np.array(similarity_for_correctness[test_id])
    if not np.any(correct_list[prediction_set] >= args.correctness_threshold):
        miscoverage_num += 1

miscoverage_rate = miscoverage_num / len(test_set)
print(f"APSS: {total_set_size / len(test_set)}")
print(f"EMR: {miscoverage_rate}")
