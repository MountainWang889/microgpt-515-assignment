# MicroGPT-515 Assignment

## Overview
This project is based on a minimal implementation of a GPT-style language model.  
The objective is to enhance the baseline architecture by incorporating several modern techniques used in large language models, including:

- GELU activation
- LoRA (Low-Rank Adaptation)
- RoPE (Rotary Position Embedding)
- Mixture of Experts (MoE)

These modifications aim to improve model expressiveness, efficiency, and scalability while maintaining a simple implementation.

---

## 1. GELU Activation

### Motivation
The original implementation uses ReLU activation, which is simple but introduces non-smooth behavior.

### Modification
ReLU was replaced with GELU in the MLP block:
x = [xi.gelu() for xi in x]
### Insight
GELU provides smoother activation and better gradient flow, which is widely adopted in modern transformer architectures such as GPT and BERT.

---

## 2. LoRA (Low-Rank Adaptation)

### Motivation
Fine-tuning large models can be expensive. LoRA reduces the number of trainable parameters by introducing low-rank updates.
q = Wq x + Bq Aq x
v = Wv x + Bv Av x
Implemented using:
def lora_linear(...)
### Insight
Instead of modifying the original weight matrix directly, LoRA adds a low-rank decomposition, enabling efficient parameter updates.

---

## 3. RoPE (Rotary Position Embedding)

### Motivation
Absolute positional embeddings have limited ability to encode relative position information.

### Modification
RoPE was applied to query and key vectors:
q = apply_rope(q, pos_id)
k = apply_rope(k, pos_id)

### Insight
RoPE injects position information into the attention mechanism itself by rotating vectors, allowing the model to better capture relative distances between tokens.

---

## 4. Mixture of Experts (MoE)

### Motivation
Increasing model capacity using dense layers is computationally expensive.

### Modification
The MLP block was replaced with multiple experts, with a gating mechanism selecting one expert per token:
expert_id = argmax(gate(x))
x = expertexpert_id

### Insight
MoE enables scaling model capacity without proportionally increasing computation, as only a subset of experts is activated.

---
---

## Results

The modified model successfully trains and generates outputs without errors:
step 1000 / 1000 | loss ...
--- inference ---
sample: ...
All enhancements are integrated into the model and function correctly.

---

## Conclusion

This project demonstrates how modern techniques can be incorporated into a minimal GPT implementation:

- GELU improves activation smoothness
- LoRA enables efficient parameter adaptation
- RoPE enhances positional encoding
- MoE increases model capacity efficiently

Together, these modifications reflect key design ideas used in state-of-the-art large language models.
