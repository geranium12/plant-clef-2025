# TODO

#### Hanna
- [x] Add eval to the pipeline
- [x] Train 5 heads of the pre-trained kaggle model (only-classifier-then-all) on augmented data + non-plant datasets
- [ ] Train the whole model (unfreeze the backbone) on augmented data + non-plant datasets
- [x] Add accelerate
- [x] Add checkpointing during training
- [x] Add lr scheduler

#### Robin
- [ ] Integrate the LUCAS no labels dataset
- [x] Combine low-resource data classes
- [ ] Figure out what DS@GT does for this challenge
- [x] Submit a model (no train) to test our pipeline

#### Tomo
- [ ] Use trees for eval on multiplication of possible options
- [ ] Implement multi-scale tiling for inference

## Later TODO
- [ ] Continue trainining the model on the LUCAS no labels dataset
- [ ] Finetune ViT Large on plant data

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
