# What does [DS@GT](https://github.com/dsgt-kaggle-clef/plantclef-2025) do?

It appears that they do not do any training.

- GENAI:
  - Uses [GBIF](https://www.gbif.org/search?q=Lactuca%20virosa%20L.) to query countries each species appears in.
  - Use gemini models to filter plants that are not from Pyrenean/Mediterranean region.
- plantclef:
  - Use luigi and spark and typer to split pipeline into independent parts
  - Preprocessing
    - They also have a top N image frequency filter
  - retrieval
    - Pretrained vit_base_patch14_reg4_dinov2.lvd142m embeds
    - Use knn between test and train to make predictions
  - masking
    - Does object detection and segmentation on types of plant parts (sand, leaf, flower, plant, ...) using grounding dino and SAM
  - ensemble
    - Combine multiple predictions via union, intersection, or a Jaccard similarity‑based ensemble
  - embedding
    - Use vit_base_patch14_reg4_dinov2.lvd142m and extract predictions and embeddings
  - detection
    - Use grounding dino to do bounding box detection
    - Only do prediction on the bounding boxes
  - classification
    - Option to use only species from Pyrenean/Mediterranean region
    - Naive Baseline - Use k most common species
    - vit_base_patch14_reg4_dinov2.lvd142m
    - Submission can be top k predictions
    - Submission can be top k predictions per test image class (The test images are one of 5 types. The type is encoded in the file name)
