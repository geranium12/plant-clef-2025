# TODO

#### Hanna
- [x] Implement evaluation metric (F1 score)
- [ ] Add 4 heads (organ, genus, family, plant/no plant) to the pre-trained kaggle model (only-classifier-then-all)
- [ ] Add train to the training pipeline

#### Robin
- [x] Implement data augmentation (tiling, color change, rotation, ...)
- [ ] Resplit the training data
- [ ] Integrate the LUCAS no labels dataset
- [ ] Add non-plant datasets (download + to the code)

#### Tomo

## Later TODO
- [ ] Train 4 heads of the pre-trained kaggle model (only-classifier-then-all) on augmented data + non-plant datasets
- [ ] Continue trainining the model on the LUCAS no labels dataset
- [ ] Implement multi-scale tiling for inference
- [ ] Finetune ViT Large on plant data
- [ ] Use trees from metadata to train 4 classifier heads and eval on multiplication of possible options


## Done
- [x] Set up hydra
- [x] Set up WandB
- [x] Set up pipeline
- [x] Extract the family -> genus -> species tree
- [x] Extract the organ -> species tree
- [x] Find datasets on stones, sand, ground, material objects (household objects) - no plants
