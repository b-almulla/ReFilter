# ReFilter
ReFilter is a hybrid retrieval framework designed to identify functionally similar mobile applications by combining embedding-based retrieval with LLM-based contextual filtering. This repository contains all code, data processing scripts, prompts, evaluation materials, and instructions required to reproduce the results reported in our paper.

Traditional app-similarity approaches rely solely on embeddings, which capture semantic similarity but often miss functional context, resulting in retrieved apps that are textually related but not actually comparable. ReFilter addresses this gap by:

* Retrieving candidate apps using scalable embedding-based similarity search

* Filtering those candidates using an LLM to verify functional similarity using contextual reasoning

This two-step design enables both efficiency and precision, achieving 92% precision in identifying truly comparable mobile apps.
The replication package supports full reproducibility so researchers and practitioners can explore app ecosystems, conduct competitor analysis, and study market dynamics with more accurate app-similarity signals. It includes:

* embedding_baselines: Scripts for embedding baseline.
* data: includes the golden truth.
* refilter_method: Inlcudes scripts for preprocessing, candidate retrieval, and LLM filtering.
