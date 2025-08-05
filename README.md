#  Polygon Colorization using Conditional UNet

This project implements a **UNet model from scratch in PyTorch** to colorize polygon images based on a given color name (as a one-hot or RGB vector). The model learns to fill polygon regions in an RGBA image conditioned on a specific color label.

---

##  Architecture

- Custom-built **UNet from scratch**
- Uses **FiLM layers** (Feature-wise Linear Modulation) to condition every layer on color input
- **GroupNorm + SiLU** activations
- Takes both **RGBA polygon image** and **color name input** (one-hot or RGB)

```python
class UNet(nn.Module):
    def __init__(self, in_ch=4, out_ch=4, color_embedding_dim=128, color_input_dim=9):
        ...
