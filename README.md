# BERT-PM-Classification

This repository contains the code accompanying our paper:

**Classification of Product Manager Types to Understand the Job Market**

[[Paper]](Classification%20of%20Product%20Manager%20Types%20to%20Understand%20the%20Job%20Market.pdf)
 [[Slides]](figure/Poster%20Presentation%20(Corredor,%20Liu,%20Asanova).pdf)


---

![Results](figure/Confidence_Interval.png)

## Overview

In this study, we explored various textual feature engineering techniques for classifying job descriptions into role-specific categories. We specifically compared:

- **TF-IDF (Term Frequency-Inverse Document Frequency)**: A statistical measure weighting words based on their frequency within a document and rarity across the dataset.
- **BERT Embeddings**: Contextualized embeddings generated from the pre-trained BERT model for a richer representation of textual data.
- **Chain-of-Thought (CoT) Prompting**: Utilizing GPT-4 to generate intermediate reasoning steps for benchmarking and comparison.

### Repository Contents

- **Extract Token.ipynb**: Jupyter notebook for data preprocessing, including downloading and tokenizing raw data.
- **final_code.ipynb**: Jupyter notebook containing the core model code, including training, evaluation, and result visualization.
- **ConfidenceInterval.png**: Image showing model results and confidence intervals.

---

## Dependencies

Below are the primary dependencies required to run the code. A detailed list can be found in the `requirements.txt` file:

```bash
re
pandas==1.5.3
numpy==1.23.5
matplotlib==3.7.1
seaborn==0.12.2
torch==2.0.1
tqdm==4.65.0
scikit-learn==1.3.0
transformers==4.31.0
