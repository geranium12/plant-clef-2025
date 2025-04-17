# TODO

#### Hanna
- [ ] Add eval to the pipeline
- [ ] Train 4 heads of the pre-trained kaggle model (only-classifier-then-all) on augmented data + non-plant datasets

#### Robin
- [ ] Integrate the LUCAS no labels dataset
- [ ] Combine low-resource data classes

#### Tomo
- [ ] Use trees from metadata to train 4 classifier heads and eval on multiplication of possible options
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
