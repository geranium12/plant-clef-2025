# TODO

#### Hanna
- [x] Continue trainining the model on the LUCAS no labels dataset
- [ ] Train 5 heads of the pre-trained kaggle model (only-classifier-then-all) on augmented data + non-plant datasets (submit it)
- [ ] Train the whole model (unfreeze the backbone) on augmented data + non-plant datasets (submit it)
- [ ] Train 5 heads ... with a threshold for low-resource classes (submit it)
- [ ] Train the whole model ... with a threshold for low-resouce classes (submit it)
- [ ] Implement FAISS knn on ALL different backbones
- [ ] Experiment with top_k and threshold when you submit the solutions
- [ ] Look at the Atlantic code and find how they used SAM (try to make it run for our case)

#### Robin

#### Tomo
- [ ] Create a model (Random Forest, basic grey pixels calculation, CNN, PCA, etc.) that classifies plant/non-plant images

## Later TODO

## Done
- [x] Set up hydra
- [x] Set up WandB
- [x] Set up pipeline
- [x] Extract the family -> genus -> species tree
- [x] Extract the organ -> species tree
- [x] Find datasets on stones, sand, ground, material objects (household objects) - no plants
- [x] Implement weighted loss
- [x] Add one augmentation that is a pipe of many random ones
- [x] Implement evaluation metric (F1 score)
- [x] Add 4 heads (organ, genus, family, plant/no plant) to the pre-trained kaggle model (only-classifier-then-all)
- [x] Add train to the pipeline
- [x] Add non-plant datasets (download + to the code)
- [x] Implement data augmentation (tiling, color change, rotation, ...)
- [x] Resplit the training data
- [x] Combine low-resource data classes
- [x] Figure out what DS@GT does for this challenge
- [x] Submit a model (no train) to test our pipeline
- [x] Add eval to the pipeline
- [x] Add accelerate
- [x] Add checkpointing during training
- [x] Add lr scheduler
- [x] Integrate the LUCAS no labels dataset
- [x] Implement multi-scale tiling for inference
- [x] Implement "top genus" filter for inference
- [x] Implement Bayesian Model Averaging or find a similar voting/pooling technique for inference
- [x] Use trees for eval on multiplication of possible options

