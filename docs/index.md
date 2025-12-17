---
# This is the YAML Front Matter block.
# For a simple page like this, it can be left completely empty.
# It tells Jekyll "this is a page to build," but the content comes after.
---

---

# Replicating "Textbooks Are All You Need"

A Case Study on Data Quality in Fine-Tuning Small Language Models

**Author:** Krasper707

**GitHub Repository:** [Link](https://github.com/Krasper707/phi-1.5-data-quality-replication)

**Date:** December 2025

---

### Abstract

This project presents an empirical replication of the core hypothesis from Microsoft Research's paper, "Textbooks Are All You Need." We demonstrate that the quality of data, rather than its sheer quantity, is a primary driver of performance when fine-tuning Small Language Models (SLMs). By fine-tuning the 1.5 billion parameter `phi-1.5` model on two distinct datasets—a small, curated set of 1,000 "textbook-quality" instruction pairs versus a larger, uncurated set of 5,000 raw code snippets—we show a significant performance delta. The model trained on high-quality data successfully learned to follow instructions and perform reasoning tasks, while the model exposed to more, but lower-quality, data exhibited pattern mimicry without true instruction-following capabilities.

---

### 1. Introduction and Motivation

The proliferation of Large Language Models (LLMs) has been fueled by scaling laws, suggesting that bigger models and larger datasets lead to better performance. However, recent work, particularly from Microsoft Research with their Phi model series, challenges this paradigm. Their paper, "Textbooks Are All You Need" (Touvron et al., 2023), posits that training on "textbook-quality" data can yield surprisingly powerful models at a much smaller scale.

This project was motivated by the desire to empirically validate this claim in a resource-constrained environment. Can the "quality over quantity" principle be observed not just in pre-training, but also in a more accessible fine-tuning context? This study aims to answer that question.

### 2. Methodology

The experiment was designed to isolate data quality as the sole independent variable.

- **Base Model:** `microsoft/phi-1.5`, a 1.5 billion parameter transformer model.
- **Fine-Tuning Technique:** We employed **QLoRA** (Quantized Low-Rank Adaptation) to make the fine-tuning process computationally feasible on a single T4 GPU. This involves 4-bit quantization of the base model and training only a small set of adapter weights.
- **Dataset 1 (High-Quality):** A subset of 1,000 samples from the `CodeAlpaca-20k` dataset. This dataset is structured with clear `[Instruction] -> [Code Output]` pairs, mimicking a textbook problem-and-solution format.
- **Dataset 2 (Low-Quality):** A subset of 5,000 samples from the `codeparrot-ds-train` dataset, representing a raw scrape of Python files from GitHub. This data lacks instructional formatting.
- **Training:** Both models were fine-tuned from the same base checkpoint for an identical number of training steps to ensure equal computational effort. The training was conducted using a manual PyTorch loop with `GradScaler` to handle mixed-precision training robustly.

### 3. Results and Analysis

A standardized evaluation suite of 10 prompts was used to compare the performance of the "Textbook-Tuned" model against the "Messy-Tuned" model. The full results can be found in the project repository. Below is a representative example:

**Prompt:** _"Create a Python function that checks if a string is a palindrome."_

| Model              | Response                                                                                                  | Analysis of Behavior                                                                                                                                                                                                            |
| ------------------ | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Textbook-Tuned** | `def is_palindrome(string): return string == string[::-1]`                                                | **Success.** The model correctly interpreted the instruction and provided a concise, correct, and Pythonic function. It demonstrated true task completion.                                                                      |
| **Messy-Tuned**    | `A palindrome is a word, phrase, number, or other sequence of characters that reads the same backward...` | **Failure.** The model failed to understand the intent. It pattern-matched the keyword "palindrome" to a textual definition it likely saw in a Markdown file or code comment, rather than performing the requested coding task. |

This pattern was consistent across the evaluation suite. The Messy-Tuned model frequently hallucinated context (e.g., creating "Exercise 2" in its response), generated unrelated code, or failed to adhere to constraints, demonstrating a critical failure in instruction-following.

### 4. Discussion

The results strongly support our initial hypothesis. The high-quality dataset, despite its small size, successfully imparted the ability to **map natural language instructions to executable code**. The model learned the _semantic link_ between a request and a solution.

Conversely, the larger, low-quality dataset primarily taught the model the _statistical structure of a code file_. The model learned to generate code that _looks like_ code from its training data (including comments, test cases, and tutorial-like formatting), but it failed to connect this generation to the user's specific instruction. This is a clear demonstration that for specialized tasks, data curation is paramount.

### 5. Limitations and Future Work

While this study provides strong evidence for the "quality over quantity" hypothesis, it has several limitations that offer avenues for future research:

- **Limited Evaluation Scope:** Our evaluation was based on a small set of 10 prompts. A more rigorous analysis would use a standardized benchmark like HumanEval to provide a more robust quantitative score.
- **Simple Success Metric:** The success metric used for our chart was a simple heuristic. A more nuanced evaluation would involve unit testing the generated code for functional correctness.
- **Scale:** This experiment was conducted at the 1.5B parameter scale. Future work could explore whether these findings hold true for larger models (e.g., 7B or 13B), or if there is a model scale at which raw data quantity begins to overcome deficits in quality.

Future work should aim to address these limitations by incorporating automated, execution-based evaluation and expanding the experiment across different model scales.

### 6. Conclusion

This case study successfully replicates the core findings of the "Textbooks Are All You Need" paper within a fine-tuning context. We have shown that a small amount of high-quality, "textbook-style" data is significantly more effective for teaching a model to perform specific tasks than a much larger corpus of uncurated data. This has important implications for the future of AI, suggesting that a focus on data-centric AI and careful curation may be a more efficient path to capable models than a brute-force scaling approach.

---

### References

1.  Touvron, H., et al. (2023). _Textbooks Are All You Need_. arXiv preprint arXiv:2306.11644.
