# Finding App Screens from Words, and Words from Screens

Cross-modal retrieval on RICO-Screen2Words. COMP 1030 Multimodal Data Analytics, Project 2.2 (Group 21 proposal,
individual implementation by Mariya).

Given a screenshot of an Android app screen, the system retrieves its human-written summary (screen to text, *i2t*);
given a summary, it retrieves the matching screen (text to screen, *t2i*). Frozen pretrained encoders embed every
screenshot and caption once, and small trainable heads compare the two modalities, so the whole study runs on a free
Google Colab T4 GPU.

## Results (test set: 1,707 screens, 8,447 captions; mean of 3 seeds)

| Method | i2t R@1 | i2t R@10 | t2i R@1 | t2i R@10 |
|---|---|---|---|---|
| Random | 0.1 | 0.6 | 0.0 | 0.5 |
| Ridge regression (baseline) | 2.1 | 10.2 | 1.1 | 7.7 |
| Early fusion | 2.6 | 15.5 | 2.7 | 14.2 |
| Contrastive dual encoder | 2.9 | 16.0 | 2.6 | 14.1 |
| Cross-attention re-rank | 3.3 | 16.3 | 2.8 | 14.7 |
| **Hybrid (contrastive top-50, early-fusion re-rank)** | 2.9 | **17.2** | 2.8 | **15.3** |
| *CLIP ViT-B/32 zero-shot (reference)* | *13.4* | *35.0* | *9.2* | *27.7* |

- **Success criterion met.** The hybrid beats ridge on Recall@10 by +7.0 points (95% CI 5.3 to 8.5) for screen to text
  and +7.6 points (6.5 to 8.6) for text to screen. The proposal required at least +5 in both directions.
- **Ranking cost per query** against a gallery of 1 million screens: contrastive 3.4 ms, hybrid 3.5 ms, early fusion
  33.6 ms, full cross-attention about 17 s (extrapolated).
- **Bias:** text-to-screen R@10 ranges from 5.9% (Dating) to 28.0% (Food & Drink). Category-balanced sampling narrowed
  the gap from 21.5 to 16.0 points but cost 1.4 points overall.

All figures, tables and numbers are in [`results/`](results/). The headline numbers are in
[`results/report_fact_sheet.txt`](results/report_fact_sheet.txt).

## Methods

| Method | Type | Notes |
|---|---|---|
| Ridge regression | linear baseline | text features mapped to image space, cosine ranking |
| Early fusion | joint representation | per-modality projections, concatenation, MLP match score |
| Contrastive dual encoder | coordinated representation | CLIP-style projections, symmetric multi-positive InfoNCE |
| Cross-attention | intermediate (attention) fusion | caption tokens attend to 21 screen-region tokens; re-ranks the contrastive top 20 |
| Hybrid | deployment design | contrastive shortlist of 50, re-ranked by early fusion |
| CLIP | reference only | zero-shot, and CLIP features with our contrastive head |

- **Feature extractors** (frozen): ResNet-50, EfficientNet-B0 and ViT-B/16 for screens, and BERT-base (mean or [CLS]
  pooling) for captions. ViT-B/16 with mean pooling was selected on validation data.
- **Data quality and preprocessing:**
  - checks for missing values, duplicate and reused captions, malformed categories, corrupt images and personal
    information;
  - captions deduplicated within each screen;
  - images resized to 224×224 without cropping.
- **Leakage-aware split:** an app-level 80/10/10 split, then the validation and test screens with a near-duplicate in
  training (cosine > 0.95) are removed.
- **Evaluation:**
  - Recall@1/5/10, MRR and median rank in both directions, in strict and lenient protocols;
  - cluster-bootstrap confidence intervals and paired tests;
  - 1K-gallery folds;
  - an overfitting check, ablations, latency from 1k to 1M candidates, per-category bias with a mitigation
    experiment, and failure analysis.

## Repository layout

```
notebooks/Project_2_2_Implementation.ipynb   main notebook, with the outputs of the full run
notebooks/project_2_2.py                     the same notebook as a plain Python script
results/figures/                             figures from the full run (PNG)
results/tables/                              all result tables (CSV), including the configuration (t15_config.csv)
results/report_fact_sheet.txt                every number used in the report
tests/smoke_data.py                          synthetic data and tiny models for an offline test run
tests/build_notebook.py                      rebuilds the .ipynb from the .py script
requirements.txt
```

## How to reproduce

1. Open `notebooks/Project_2_2_Implementation.ipynb` in Google Colab (File > Open notebook > GitHub, then paste this
   repository's URL).
2. Runtime > Change runtime type > **T4 GPU**.
3. Runtime > **Run all**, and allow Google Drive access when asked. Features are cached in
   `MyDrive/comp1030_p22_cache` (about 5 GB), so a re-run after a disconnect skips feature extraction.
4. The full run takes about 1.5 hours on a free T4. Settings live in the `CFG` dictionary in the first code section.
   To shorten the run, set `RUN_ABLATIONS=False` (saves about 10 minutes) or `RUN_CROSS_ATTN=False` (saves about 25
   minutes).
5. Outputs are written to `outputs/` and downloaded as `p22_outputs.zip`.

Seeds are fixed: 42 for data operations and 0, 1, 2 for model training. The tested environment was Python 3.13,
PyTorch 2.14, Transformers 5.22 and Datasets 4.8 on a Tesla T4.

**Offline test** (no internet or GPU needed; checks that every cell runs, on synthetic data):

```bash
pip install -r requirements.txt
SMOKE=1 PYTHONPATH=tests python notebooks/project_2_2.py
```

## Data, licence and privacy

- **Data:** RICO-Screen2Words from Hugging Face (`rootsautomation/RICO-Screen2Words`; mirror `bevaya/RICO-Screen2Words`).
- **Licence:** the annotations are CC BY 4.0. The screenshots come from the RICO research dataset (Deka et al., 2017)
  and remain the property of the app owners.
- **Screenshots in this repository:** the stand-alone `results/` folder contains no screenshots. The notebook's saved
  outputs include a few example screenshots (failure analysis figure), shown for research illustration only.
- **Privacy:** no personal-information exclusions were applied in this run (see report, Section 7).

## References

- Baltrušaitis, T., Ahuja, C., & Morency, L.-P. (2019). Multimodal machine learning: A survey and taxonomy. *IEEE TPAMI*.
- Deka, B., et al. (2017). Rico: A mobile app dataset for building data-driven design applications. *UIST*.
- Wang, B., et al. (2021). Screen2Words: Automatic mobile UI summarization with multimodal learning. *UIST*.
- Devlin, J., et al. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL*.
- Dosovitskiy, A., et al. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. *ICLR*.
- He, K., et al. (2016). Deep residual learning for image recognition. *CVPR*.
- Tan, M., & Le, Q. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. *ICML*.
- Radford, A., et al. (2021). Learning transferable visual models from natural language supervision. *ICML*.
- van den Oord, A., Li, Y., & Vinyals, O. (2018). Representation learning with contrastive predictive coding. *arXiv:1807.03748*.
- Miech, A., et al. (2020). End-to-end learning of visual representations from uncurated instructional videos. *CVPR*.
- Li, J., et al. (2021). Align before fuse: Vision and language representation learning with momentum distillation. *NeurIPS*.
- Faghri, F., et al. (2018). VSE++: Improving visual-semantic embeddings with hard negatives. *BMVC*.
- Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR*.
- Efron, B., & Tibshirani, R. (1993). *An introduction to the bootstrap*. Chapman & Hall.
- Mitchell, M., et al. (2019). Model cards for model reporting. *FAT\**.
