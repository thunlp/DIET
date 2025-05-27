# The Overthinker's DIET: Cutting Token Calories with DIfficulty-AwarE Training

This repository contains the official implementation for the paper: "[The Overthinker's DIET: Cutting Token Calories with DIfficulty-AwarE Training](https://arxiv.org/pdf/2505.19217)".

## Abstract
Recent large language models (LLMs) exhibit impressive reasoning but often _overthink_, generating excessively long responses that hinder efficiency. We introduce **<ins>DI</ins>fficulty-Awar<ins>E</ins> <ins>T</ins>raining (DIET)**, a framework that systematically cuts these "token calories" by integrating on-the-fly problem difficulty into the reinforcement learning (RL) process. **DIET** dynamically adapts token compression strategies by modulating token penalty strength and conditioning target lengths on estimated task difficulty, to optimize the performance-efficiency trade-off. We also theoretically analyze the pitfalls of naive reward weighting in group-normalized RL algorithms like GRPO, and propose _Advantage Weighting_ technique, which enables stable and effective implementation of these difficulty-aware objectives.
Experimental results demonstrate that **DIET** significantly reduces token counts while simultaneously _improving_ reasoning performance. Beyond raw token reduction, we show two crucial benefits largely overlooked by prior work: (1) **DIET** enhances the natural positive correlation between response length and problem difficulty, ensuring verbosity is appropriately allocated, unlike many existing compression methods that disrupt this relationship. (2) Critically, **DIET** leads to superior **inference scaling**; by maintaining high per-sample quality with fewer tokens, it enables better aggregate performance (e.g., via majority voting) under fixed computational budgets, an area where other methods falter. Our analyses provide a principled and effective framework for developing more efficient, practical, and high-performing LLMs.

## Installation

1.  **Create Conda Environment:**
    ```bash
    conda create -n diet python==3.11
    conda activate diet
    ```

2.  **Install `verl` library and dependencies:**
    The `verl` directory contains core code for our reinforcement learning framework.
    ```bash
    git clone git@github.com:thunlp/DIET.git
    cd diet

    cd verl
    pip install -e .  # Installs 'verl' in editable mode
    cd ..
    pip install -r requirements.txt
    ```

## Data and Model Setup

Follow these steps to download the necessary dataset and base model.

### 1. Download Dataset (DeepScaleR)
The DeepScaleR dataset will be downloaded to `verl/dataset/deepscaler_dataset/`.
```bash
huggingface-cli download --repo-type dataset --resume-download agentica-org/DeepScaleR-Preview-Dataset --local-dir verl/dataset/deepscaler_dataset --local-dir-use-symlinks False
```
### 2. Download Base Model (R1-Distilled Qwen 1.5B)
The base model will be downloaded to `checkpoints/r1-distilled-qwen-1.5b/`.
```bash
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --local-dir checkpoints/r1-distilled-qwen-1.5b --local-dir-use-symlinks False
```
### 3. Data Preprocessing
This step preprocesses the DeepScaleR dataset.
```bash
cd verl
python examples/data_preprocess/deepscaler.py \
    --input_path dataset/deepscaler_dataset/ 
cd ..
```
*Note: The `--input_path` is relative to the `verl` directory after `cd verl`.*

## Training

### Training with DIET
To train our main DIET model:
```bash
cd verl
bash bash_scripts/deepscaler_r1_distilled_qwen_1.5b/diet.sh
```
