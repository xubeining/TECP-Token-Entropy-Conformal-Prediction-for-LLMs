## TECP: Token-Entropy Conformal Prediction for LLMs

### [Paper](https://arxiv.org/abs/2509.00461)

This repo contains the official PyTorch implementation of TECP.

> [Beining Xu](https://xubeining.github.io/)
> <br>Shenzhen MSU-BIT University<br>

## abstract
Uncertainty quantification (UQ) for open-ended language generation remains a critical yet underexplored challenge, particularly in settings where token-level probabilities are available via API access. In this paper, we **introduce Token-Entropy Conformal Prediction (TECP)**. The framework leverages logit-based, reference-free token entropy as an uncertainty measure and integrates it into a split conformal prediction (CP) pipeline to construct prediction sets with formal coverage guarantees. Unlike approaches that rely on semantic-consistency heuristics alone, TECP directly estimates episodic uncertainty from the token-entropy structure of sampled generations and calibrates uncertainty thresholds via CP quantiles to ensure provable error control. Empirical evaluations across six large language models and two benchmarks (CoQA and TriviaQA) show that TECP consistently achieves reliable coverage and compact prediction sets, outperforming prior self-UQ methods. Our results provide a principled and efficient solution for trustworthy generation in white-box, log-probability–accessible LLM settings.

## Setup
First, download and set up the repo:

```bash
git clone https://github.com/xubeining/TECP-Token-Entropy-Conformal-Prediction-for-LLMs.git
cd TECP-Token-Entropy-Conformal-Prediction-for-LLMs
```

## Data
You should run:

```bash
unzip row_data.zip
```

Finally, you should have the following structure:

```
nwm/data
├── <triviaqa>
│   ├── validation-00000-of-00001.parquet
│   ├── train-00000-of-00001.parquet
│   └── test-00000-of-00001.parquet
├── <coqa>
│   ├── coqa-train-v1.0.json
│   ├── coqa-train-small.json
│   ├── coqa-dev-small.json
│   └── coqa-dev-v1.0.json
└──
```  


## Requirements:
```bash
conda create -n conu_env python=3.11 -y
conda activate conu_env
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126
pip install -r requirements.txt
```

## Run for coqa
```bash
python coqa_prase.py   --generate-model "Llama-3.2-1B"   --row-data-path "./row_data/coqa/coqa-train-v1.0.json"   --data-dir "./processed_datasets"   --cache-dir "./hf_cache"   --few-shot-num 3   --max-num 1000

python generate.py   --generate-model "Llama-3.2-1B"   --dataset "coqa"   --data-dir "./processed_datasets"   --record-dir "./records"   --cache-dir "./hf_cache"   --num-generations-per-prompt 10   --temperature 1.0


python clean_open_domain_qa.py   --generate-model "Llama-3.2-1B"    --dataset "coqa"   --record-dir "./records"   --cache-dir "./hf_cache"

python open_domain_qa_similarity.py   --generate-model "Llama-3.2-1B"   --dataset "coqa"   --record-dir "./records"   --cache-dir "./hf_cache"   --similarity-model "./models/stsb-roberta-large"

CUDA_VISIBLE_DEVICES=0 python get_likelihoods.py   --evaluation-model Llama-3.2-1B  --run-name coqa/Llama-3.2-1B/num_generations-10/temperature-1.0/max_len_of_generation-36


python conformal_uncertainty_criterion.py   --generate-model "Llama-3.2-1B"   --dataset "coqa"   --record-dir ./records   --cache-dir ./hf_cache   --split-ratio 0.5   --correctness-threshold 0.7   --alpha 0.1
```
## Run for TriviaQA


## Result of mywork
<p align="center">
  <img src="https://github.com/xubeining/TECP-Token-Entropy-Conformal-Prediction-for-LLMs/blob/main/photo/TECP_on_TriviaQA.png" width="45%"/>
  <img src="https://github.com/xubeining/TECP-Token-Entropy-Conformal-Prediction-for-LLMs/blob/main/photo/TECP_on_CoQA.png" width="45%"/>
</p>

## Result of baseline
<p align="center">
  <img src="https://github.com/xubeining/TECP-Token-Entropy-Conformal-Prediction-for-LLMs/blob/main/photo/Conu_on_TriviaQA.png" width="45%"/>
  <img src="https://github.com/xubeining/TECP-Token-Entropy-Conformal-Prediction-for-LLMs/blob/main/photo/Conu_on_CoQA.png" width="45%"/>
</p>



