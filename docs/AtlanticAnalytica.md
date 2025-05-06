# [AtlanticAnalytica](https://github.com/stevefoy/AtlanticAnalytica)

- They use SAM the following way:
  - Somehow (code not available), they generate a rock mask using SAM.
  - If the mask is too much mask (80%) they don't use it.
  - Otherwise, for each tile, if it is mostly masked, they ignore that tile.
  - So basically they use SAM as a plant, not a plant classifier.
