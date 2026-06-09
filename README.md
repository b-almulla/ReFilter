# ReFilter
ReFilter: a hybrid framework that first Retrieves semantically related candidate apps using embeddings and then applies LLM-based contextual Filtering to identify true functionally similar apps with higher precision. This design balances efficiency and accuracy, achieving an F1-score of 90% for retrieving similar apps.

This repository contains all code, data processing scripts, prompts, evaluation materials, and instructions required to reproduce the results reported in our paper.

Traditional app-similarity approaches rely solely on embeddings, which capture semantic similarity but often miss functional context, resulting in retrieved apps that are textually related but not actually comparable. ReFilter addresses this gap by:

* Retrieving candidate apps using scalable embedding-based similarity search

* Filtering those candidates using an LLM to verify functional similarity using contextual reasoning

The replication package supports full reproducibility so researchers and practitioners can explore app ecosystems, conduct competitor analysis, and study market dynamics with more accurate app-similarity signals. It includes:

* refilter_method: Includes scripts for preprocessing, candidate retrieval, and LLM filtering.
* baselines: Scripts for embedding baseline.
* data: includes the golden truth.

## Our Approach (`Refilter_method` folder)

### Step 1: Preprocessing

Our preprocessing consists of two stages: (1) applying the baseline procedure introduced in prior work, and (2) applying our extensions for improved text normalization.

1. **Baseline Preprocessing** (Prior Work)

We follow the preprocessing procedure introduced by Wei et al. [1], which includes:

- Language filtering (retaining only English descriptions),
- Removal of noisy sentences containing URLs, emails, or boilerplate phrases (e.g., privacy policy, subscription information),
- Removal of emojis.

The original implementation is publicly available in the authors’ replication package:

**Repository:** https://github.com/Jl-wei/feature-inspiration  
**File used:** `preprocessing.py`  
(See the original project for full details and instructions.)

To reproduce our results, first run their preprocessing script on the raw app data to obtain the baseline-cleaned descriptions.

2. **ReFilter Extended Preprocessing** (Our Additions)

After generating the baseline-preprocessed file, we apply our own extensions to improve text consistency and prepare descriptions for embedding-based retrieval. Specifically, we:

- Normalize whitespace,
- Convert all text to lowercase,
- Remove HTML tags and markup,
- Strip metadata-like content that does not describe app functionality,
- Remove apps with descriptions containing fewer than **200 characters**, following the same threshold used in Wei et al. (ASE 2024).

Our preprocessing script implementing these extensions is provided in `refilter_method/step_1_preprocessing/preprocessing.py`

### Step 2: Candidate Retrieval

We evaluate several embedding models in the paper and identify Linq as the most effective for retrieving functionally similar apps.

Generate embeddings
Each app description is converted into a vector representation using the Linq embedding model
(script: `refilter_method/step_2_candidate_retrieval/candidate_embedding_LINQ.py`).

Compute similarity rankings
For each target app, we compute cosine similarity between its embedding and all other apps, then produce a ranked list sorted by similarity
(script: `refilter_method/step_2_candidate_retrieval/candidate_ranking.py`).

This ranking forms the candidate pool for the next stage.

### Step 3: LLM Filtering
We apply 2-shot prompting with GPT-5 to filter the candidate list and produce the final set of functionally similar apps
(script: `refilter_method/step_3_LLM_filtering/LLM_filtering.py`).

The model receives the target app, top-K candidates, and two decision examples, and returns a structured judgment for each candidate.

## Implemented baselines (`baselines` folder)
The `baseline` folder contains the scripts used to generate the five baseline embedding sets that we compare against the LINQ model in our experiments. Each script produces one embedding per app using a different model or pooling strategy.

Included baselines:

* `BGE_avg.py`: Generates embeddings using the BGE model, applying average pooling over all token embeddings.

* `BGE_first.py`: Generates embeddings using the BGE model but uses only the first token (CLS-style) representation.

* `VoyageAI.py`: Produces a single embedding for each app using Voyage Multimodal, incorporating both the app description and all available screenshots.

* `SigLIP2/`: This folder contains scripts for the SigLIP2 model. Each app’s text and images are embedded individually, then combined using two different pooling strategies to produce one final embedding per app:

1) Simple average pooling
2) Top-k pooling (highest-similarity elements)

These baselines allow direct comparison with LINQ across diverse embedding architectures and pooling methods.

## Dataset (`data` folder)

* `data/validation_set.csv`: Contains the manually validated ground truth for the top 30 candidates generated by each embedding baseline, for 10 randomly selected target apps. This set is used to identify the best-performing embedding model across the six baselines. The LINQ results from this set were then used to refine the prompt for the LLM filtering stage.
* `data/test_set.csv`: Contains a manually validated ground truth set for the top 30 candidates of a new set of 10 target apps, selected using LINQ.
This dataset is used to evaluate the refined LLM filtering prompts and ensure that improvements do not overfit the validation set.
* Full dataset with all apps and their metadata, and LINQ embeddings of app descriptions is available in [this link.](https://doi.org/10.6084/m9.figshare.30816728)
 
Reference: 

[1] Jialiang Wei, Anne-Lise Courbis, Thomas Lambolais, Binbin Xu, Pierre Louis Bernard, Gerard Dray, and Walid Maalej. 2024. Getting Inspiration for Feature Elicitation: App Store- vs. LLM-based Approach. In Proceedings of the 39th IEEE/ACM International Conference on Automated Software Engineering (Sacramento, CA, USA) (ASE ’24). Association for Computing Machinery, New York, NY, USA, 857–869. https://doi.org/10.1145/3691620.3695591


