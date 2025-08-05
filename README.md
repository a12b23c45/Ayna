#  Polygon Colorization using Conditional UNet

This project implements a **UNet model from scratch in PyTorch** to colorize polygon images based on a given color name (as a one-hot or RGB vector). The model learns to fill polygon regions in an RGBA image conditioned on a specific color label.

---
## File Structure while implementing in collab

-- Drive/Mydrive/dataset/Training....
--drive/ Mydrive/dataset/Validation/...

---
##  Architecture

- Custom-built **UNet from scratch**
- Uses **FiLM layers** (Feature-wise Linear Modulation) to condition every layer on color input
- **GroupNorm + SiLU** activations
- Takes both **RGBA polygon image** and **color name input** (one-hot or RGB)
---


## Hyperparameters
- Hyperparameter	Value 
- Image Size	192×192
- Batch Size	32
- Learning Rate	1e-4
- Optimizer	Adam
- Loss Function	0.7 * L1 + 0.3 * MSE
- Conditioning Input	One-hot colour name or RGB (dim = 9)
- Epochs	80 (No overfitting or Underfitting)
---
## Training Dynamics:
The model was trained using both L1 loss and MSE loss to ensure sharp yet consistent outputs.
- •	Training Loss decreased steadily.
- •	PSNR improved during early epochs, then plateaued.
---
## Key Learnings:
- 1)	Fi LM conditioning is very effective when used across all Convolution Blocks.
- 2)	Group Norm + Si LU works well with small batch sizes.
- 3)	Learned that U Net performs better than CNN.
- 4)	Tracking via Weights & Biases was critical for debugging but graphs helped to check if model was trained correctly.
---
