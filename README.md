[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/your-username/your-model-name)



# Gemma3-Reasoning-GSM8K
A learning experiment: Fine-tuning Gemma with GRPO using Unsloth for faster, efficient reasoning alignment on limited hardware.
# Gemma 3 1B Reasoning with GRPO 🧠

This project demonstrates fine-tuning the **Google Gemma 3 1B Instruct** model using **Group Relative Policy Optimization (GRPO)** to improve mathematical reasoning capabilities. This is the same reinforcement learning logic used by models like DeepSeek-R1.


## 📌 Problem Statement
Small Language Models (SLMs) like Gemma 3 1B are highly efficient but often struggle with complex, multi-step reasoning. Traditional fine-tuning (SFT) teaches them what to say, but not how to think.
The goal of this project was to implement Group Relative Policy Optimization (GRPO):the reinforcement learning algorithm popularized by DeepSeek-v3/R1—to "unlock" reasoning capabilities in a 1B model. I aimed to train the model to follow a structured Chain-of-Thought (CoT) process to solve mathematical word problems using the GSM8K dataset.

## 🚀 Overview
The goal of this project was to train a lightweight model (1B parameters) to follow a structured "Chain of Thought" (CoT) reasoning process before providing a final answer to math problems.

- **Base Model:** `google/gemma-3-1b-it`
- **Dataset:** [GSM8K](https://huggingface.co/datasets/openai/gsm8k) (Grade School Math)
- **Method:** GRPO (Reinforcement Learning)
- **Framework:** [Unsloth](https://github.com/unslothai/unsloth) (Fast 4-bit LoRA training)

## 🛠️ Hardware & Environment
- **GPU:** NVIDIA Tesla T4 (16GB VRAM)
- **Runtime:** Google Colab / Linux
- **Precision:** Fine-tuned in 4-bit quantization to fit within T4 memory constraints.

## Methodologies
1. Group Relative Policy Optimization (GRPO)
GRPO is a Reinforcement Learning (RL) algorithm that eliminates the need for a separate "Critic" model (unlike PPO). Instead, it generates a group of outputs for each prompt and uses their relative scores to calculate the advantage.
The GRPO Loss Function:
The objective function minimized during training is: <img width="869" height="91" alt="image" src="https://github.com/user-attachments/assets/6f836ee9-6ca9-45bf-b593-e590b29b1ea7" />
<img width="625" height="109" alt="image" src="https://github.com/user-attachments/assets/5a44b60c-d623-49c1-bbe2-15dcfdecf2b7" />


2. Reward Functions :
I guided the model's behavior using multiple verifiable reward functions:
Correctness Reward: Rewards the model if the extracted answer matches the ground truth.
Format Reward: Rewards the model for correctly using XML tags: <thought> for reasoning and <answer> for the result.
Integer Reward: Extra points for providing a clean numeric answer for math problems.
3. Training Framework: Unsloth
Used Unsloth for 2x faster training and 4-bit LoRA (Low-Rank Adaptation). This allowed the 1B model and the GRPO generation phase to fit within the 16GB VRAM of a Tesla T4.

## 📈 Results Comparison
After fine-tuning, the model shifted from direct answering to a structured reasoning approach.

| Feature | Base Gemma 3 1B | Fine-Tuned (GRPO) |
| :--- | :--- | :--- |
| **Strategy** | Direct Response | Step-by-Step Reasoning |
| **Formatting** | Plain Text | Structured XML Tags |
| **Reasoning** | Brief / Implicit | Explicit `<thought>` trace |

### Example Comparison:
**Question:** Janet's ducks lay 16 eggs per day. She eats three for breakfast and bakes with four. How many are left?

**Base Model Output:** 
> Janet has 9 eggs left.

**Fine-Tuned Model Output:**
> <thought>
> 1. Janet starts with 16 eggs.
> 2. She eats 3: 16 - 3 = 13.
> 3. She bakes with 4: 13 - 4 = 9.
> The answer is 9.
> </thought>
> <answer>9</answer>

## 📈 Training Progress

| Step | Training Loss | Reward  | Completions / Mean Length |
|------|---------------|---------|---------------------------|
| 1    | 0.0000        | -0.4688 | 323.17                    |
| 50   | 0.0002        | 0.3070  | 86.33                     |
| 100  | 0.0003        | 1.3958  | 85.67                     |
| 150  | 0.0003        | 1.0000  | 65.00                     |
| 200  | 0.0002        | 1.2708  | 202.50                    |
| 250  | 0.0003        | 1.0000  | 56.33                     |


This graph shows how **reward increases** and **reasoning length stabilizes** during GRPO fine-tuning of Gemma 3 1B. Early training produces long, low-reward outputs, while later steps converge toward **higher reward with more concise reasoning**, indicating improved alignment and efficiency.

<img width="600" height="347" alt="image" src="https://github.com/user-attachments/assets/8f68bc5e-bf4e-4043-bbcd-ae51863b03e0" />

Take a look at the training loss vs steps graph

<img width="500" height="310" alt="image" src="https://github.com/user-attachments/assets/cb6464b6-232b-4922-a080-ed95364d5403" />


Training loss remains low and stable across 250 steps, with minor spikes reflecting exploration during GRPO fine-tuning of Gemma 3 1B.

## ⚠️ Technical Note for Tesla T4 Users
Gemma 3 models are designed for `bfloat16`. On older hardware like the **Tesla T4**, training requires using `float32` compute dtype and disabling vLLM fast inference to avoid numerical instability (NaN losses). 
## 💡 What I Learned
1. Hardware Architecture Matters: I learned that older GPUs like the Tesla T4 cannot handle bfloat16. This taught me how to debug "Numerical Instability" errors and the importance of matching the compute_dtype to the hardware.
2. RL vs. SFT: Standard fine-tuning (SFT) is like teaching a student to memorize answers. GRPO is like teaching a student to "show their work" through rewards, which leads to much better generalization on math tasks.
3. The Complexity of GRPO: Implementing GRPO requires balancing multiple rewards. If the "Format Reward" is too high, the model might use tags but give the wrong answer. If "Correctness" is too high, it might skip the reasoning. Tuning these weights is the "secret sauce" of RL.
4. Efficiency is Possible: You don't need a massive GPU cluster to experiment with DeepSeek-style reasoning. With libraries like Unsloth, you can perform advanced RL on a free-tier T4 GPU.

## 🏗️ How to Use
1. Clone the repo: `git clone https://github.com/tanicodesallday/Gemma-3-1B-GRPO-Reasoning.git`
2. Install dependencies: `!pip install unsloth` and `!pip install --upgrade pillow`
3. Load the model via Hugging Face:
```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained("tanicodesallday/gemma3-1b-grpo")
