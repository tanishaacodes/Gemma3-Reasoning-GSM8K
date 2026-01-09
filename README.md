# Gemma3-Reasoning-GSM8K
A learning experiment: Fine-tuning Gemma with GRPO using Unsloth for faster, efficient reasoning alignment on limited hardware.
# Gemma 3 1B Reasoning with GRPO 🧠

This project demonstrates fine-tuning the **Google Gemma 3 1B Instruct** model using **Group Relative Policy Optimization (GRPO)** to improve mathematical reasoning capabilities. This is the same reinforcement learning logic used by models like DeepSeek-R1.

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

## 📊 Training Methodology (GRPO)
Unlike standard PPO, **GRPO** removes the need for a separate Critic model, saving significant VRAM. The model generates a group of responses (e.g., 4-8) for each prompt and is rewarded based on:
1.  **Correctness:** Does the final answer match the ground truth?
2.  **Format:** Does the model use `<thought>` and `<answer>` XML tags?
3.  **Numerical Stability:** Encouraging integer outputs for math problems.

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

## ⚠️ Technical Note for Tesla T4 Users
Gemma 3 models are designed for `bfloat16`. On older hardware like the **Tesla T4**, training requires using `float32` compute dtype and disabling vLLM fast inference to avoid numerical instability (NaN losses). 

## 🏗️ How to Use
1. Clone the repo: `git clone https://github.com/tanicodesallday/Gemma-3-1B-GRPO-Reasoning.git`
2. Install dependencies: `!pip install unsloth` and `!pip install --upgrade pillow`
3. Load the model via Hugging Face:
```python
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained("tanicodesallday/gemma3-1b-grpo")
