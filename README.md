# ReFilter
ReFilter is a hybrid retrieval framework designed to identify functionally similar mobile applications by combining embedding-based retrieval with LLM-based contextual filtering. This repository contains all code, data processing scripts, prompts, evaluation materials, and instructions required to reproduce the results reported in our paper.

Traditional app-similarity approaches rely solely on embeddings, which capture semantic similarity but often miss functional context, resulting in retrieved apps that are textually related but not actually comparable. ReFilter addresses this gap by:

* Retrieving candidate apps using scalable embedding-based similarity search

* Filtering those candidates using an LLM to verify functional similarity using contextual reasoning

This two-step design enables both efficiency and precision, achieving 92% precision in identifying truly comparable mobile apps.
The replication package supports full reproducibility so researchers and practitioners can explore app ecosystems, conduct competitor analysis, and study market dynamics with more accurate app-similarity signals. It includes:

* refilter_method: Includes scripts for preprocessing, candidate retrieval, and LLM filtering.
* embedding_baselines: Scripts for embedding baseline.
* data: includes the golden truth.


## Step 1: Preprocessing

Our preprocessing consists of two stages: (1) applying the baseline procedure introduced in prior work, and (2) applying our extensions for improved text normalization.

### 1. Baseline Preprocessing (Prior Work)

We follow the preprocessing procedure introduced by Wei et al. [1], which includes:

- Language filtering (retaining only English descriptions),
- Removal of noisy sentences containing URLs, emails, or boilerplate phrases (e.g., privacy policy, subscription information),
- Removal of emojis.

The original implementation is publicly available in the authors’ replication package:

**Repository:** https://github.com/Jl-wei/feature-inspiration  
**File used:** `preprocessing.py`  
(See the original project for full details and instructions.)

To reproduce our results, first run their preprocessing script on the raw app data to obtain the baseline-cleaned descriptions.

### 2. ReFilter Extended Preprocessing (Our Additions)

After generating the baseline-preprocessed file, we apply our own extensions to improve text consistency and prepare descriptions for embedding-based retrieval. Specifically, we:

- Normalize whitespace,
- Convert all text to lowercase,
- Remove HTML tags and markup,
- Strip metadata-like content that does not describe app functionality,
- Remove apps with descriptions containing fewer than **200 characters**, following the same threshold used in Wei et al. (ASE 2024).

Our preprocessing script implementing these extensions is provided in refilter_method/preprocessing.py

## Step 2: Candidate Retrieval

We evaluate several embedding models in the paper and identify Linq as the most effective for retrieving functionally similar apps.

Generate embeddings
Each app description is converted into a vector representation using the Linq embedding model
(script: candidate_embedding_LINQ.py).

Compute similarity rankings
For each target app, we compute cosine similarity between its embedding and all other apps, then produce a ranked list sorted by similarity
(script: candidate_ranking.py).

This ranking forms the candidate pool for the next stage.

## Step 3: LLM Filtering
We apply 2-shot prompting with GPT-5 to filter the candidate list and produce the final set of functionally similar apps
(script: LLM_filtering.py).

The model receives the target app, top-K candidates, and two decision examples, and returns a structured judgment for each candidate.

Reference: 

[1] Jialiang Wei, Anne-Lise Courbis, Thomas Lambolais, Binbin Xu, Pierre Louis Bernard, Gerard Dray, and Walid Maalej. 2024. Getting Inspiration for Feature Elicitation: App Store- vs. LLM-based Approach. In Proceedings of the 39th IEEE/ACM International Conference on Automated Software Engineering (Sacramento, CA, USA) (ASE ’24). Association for Computing Machinery, New York, NY, USA, 857–869. https://doi.org/10.1145/3691620.3695591


