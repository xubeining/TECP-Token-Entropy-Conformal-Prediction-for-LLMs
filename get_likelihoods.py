#!/usr/bin/env python
import argparse
import os
import pickle
import random
import numpy as np
import torch
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--evaluation-model', type=str, default='huggyllama/llama-7b')
parser.add_argument('--run-name', type=str, default='huggyllama/llama-7b/coqa')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import config
output_dir = config.output_dir
os.environ["HF_DATASETS_CACHE"] = config.hf_datasets_cache

seed_value = 10
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)

run_name = args.run_name

with open(f'{output_dir}/{run_name}/cleaned_generations.pkl', 'rb') as infile:
    sequences = pickle.load(infile)

with open(f'{output_dir}/{run_name}/similarity_scores.pkl', 'rb') as infile:
    sim_list = pickle.load(infile)
    similarity_dict, similarity_for_correctness = sim_list

semantic_clusters = None
sc_path = f'{output_dir}/{run_name}/semantic_clusters.pkl'
if os.path.exists(sc_path):
    with open(sc_path, 'rb') as f:
        semantic_clusters = pickle.load(f)

if 'opt-30b' in args.evaluation_model or 'llama-13b' in args.evaluation_model or 'vicuna' in args.evaluation_model:
    model = AutoModelForCausalLM.from_pretrained(args.evaluation_model, torch_dtype=torch.float16, device_map='auto')
else:
    model = AutoModelForCausalLM.from_pretrained(args.evaluation_model, torch_dtype=torch.float16).to(device)

tokenizer = AutoTokenizer.from_pretrained(args.evaluation_model, use_fast=False)

if 'opt' in args.evaluation_model:
    pad_token_id = tokenizer.pad_token_id
elif 'Llama' in args.evaluation_model or 'vicuna' in args.evaluation_model:
    pad_token_id = 1
elif 'llama' in args.evaluation_model or 'vicuna' in args.evaluation_model:
    pad_token_id = 1
elif 'Qwen' in args.evaluation_model or 'vicuna' in args.evaluation_model:
    pad_token_id = 1
else:
    raise NotImplementedError(f"Unsupported model: {args.evaluation_model}")

def get_token_wise_entropies(generation, logits, labels, vocab_size):
    shifted_logits = logits[..., :-1, :].reshape(-1, vocab_size)
    shifted_labels = labels[..., 1:].reshape(-1)
    token_wise_entropy = torch.nn.CrossEntropyLoss(reduction='none')(shifted_logits, shifted_labels)
    token_wise_entropy = token_wise_entropy[shifted_labels != -100].cpu().detach()
    generation = generation[labels != -100]
    assert token_wise_entropy.size(0) == generation.size(0)
    return token_wise_entropy

def get_neg_loglikelihoods(model, sequences):
    with torch.no_grad():
        result = []
        for sample in tqdm.tqdm(sequences):
            result_dict = {}
            prompt = sample['prompt']

            if 'cleaned_generations' in sample:
                generations = sample['cleaned_generations'].to(device)
            elif 'generations' in sample:
                generations = sample['generations'].to(device)
            elif ('cleaned_generated_texts' in sample) or ('generated_texts' in sample) or ('sampled_generated_texts' in sample):
                text_key_candidates = ['cleaned_generated_texts', 'generated_texts', 'sampled_generated_texts']
                text_key = next(k for k in text_key_candidates if k in sample)
                generations = []
                for text in sample[text_key]:
                    input_ids = tokenizer.encode(sample['prompt'], add_special_tokens=False)
                    out_ids = tokenizer.encode(text, add_special_tokens=False)
                    generations.append(torch.tensor(input_ids + out_ids))
                generations = torch.nn.utils.rnn.pad_sequence(
                    generations, batch_first=True, padding_value=pad_token_id
                ).to(device)
            else:
                continue

            id_ = sample['id']
            if isinstance(id_, list):
                id_ = id_[0]

            token_wise_entropy_list = []

            for generation_index in range(generations.shape[0]):
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                prompt_tensor = torch.tensor(prompt_ids, device=device)
                prompt_tensor = prompt_tensor[prompt_tensor != pad_token_id]

                generation = generations[generation_index]
                generation = generation[generation != pad_token_id]

                target_ids = generation.clone()
                target_ids[:len(prompt_tensor)] = -100

                model_output = model(torch.reshape(generation, (1, -1)), labels=target_ids)
                token_wise_entropy = get_token_wise_entropies(
                    generation, model_output.logits, target_ids, model.config.vocab_size
                )
                token_wise_entropy_list.append(token_wise_entropy)

            result_dict['prompt'] = prompt
            result_dict['id'] = id_
            result_dict['token_wise_entropy'] = token_wise_entropy_list

            if semantic_clusters is not None and id_ in semantic_clusters:
                ss = semantic_clusters[id_]['semantic_set_ids']
                result_dict['semantic_set_ids'] = torch.tensor(ss, device=device)
            else:
                result_dict['semantic_set_ids'] = torch.arange(len(token_wise_entropy_list), device=device)

            result.append(result_dict)

        return result

likelihoods = get_neg_loglikelihoods(model, sequences)

with open(f'{output_dir}/{run_name}/generations_likelihoods.pkl', 'wb') as outfile:
    pickle.dump(likelihoods, outfile)
