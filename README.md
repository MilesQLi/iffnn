# Interpretable Feedforward Neural Network (IFFNN) Library

This library provides an implementation of the Interpretable Feedforward Neural Network (IFFNN) architecture, based on the work by Li et al. ("On the Effectiveness of Interpretable Feedforward Neural Network").

IFFNNs aim to combine the predictive power of neural networks with the interpretability of simpler models like logistic/softmax regression. They achieve this by dynamically computing feature weights based on the input sample itself and then using these weights to calculate the final output. The contribution of each feature `x_i` to the prediction can be directly inspected as `W(x)_i * x_i` (binary) or `W(x)_{j,i} * x_i` (multi-class).

---

## Features

*   Easy-to-use `IFFNN` class compatible with PyTorch workflows.
*   Handles both binary (`num_classes=1`) and multi-class (`num_classes > 1`) classification.
*   Automatic default hidden layer configuration if not specified.
*   Feature name handling for clearer explanations.
*   Integrated `train_model` method for convenient training with validation and best model saving.
*   `predict_proba` and `predict` methods for inference.
*   `explain` method to generate feature contributions for model predictions, with options for controlling output.
*   Selectable activation functions ('relu', 'tanh').
*   Automatic device selection ('auto', 'cuda', 'cpu').

---

## Installation

You can install the latest release directly from PyPI:

```bash
pip install iffnn
```

Or specify a version (e.g., version 0.2.0):

```bash
pip install iffnn==0.2.0
```

If you want the latest development version:

```bash
pip install git+https://github.com/milesqli/iffnn.git
```

---

## Quick Usage Example

This example assumes you have your data prepared as PyTorch `DataLoader` objects: `train_loader`, `valid_loader`, `test_loader`. It also assumes you know `input_size` (number of features), `n_classes` (number of classes), and optionally have lists for `feature_names` and `class_names`.

```python
import torch
from iffnn import IFFNN # Make sure iffnn is installed

# --- Assumed Inputs ---
# train_loader, valid_loader, test_loader = ... (Your PyTorch DataLoaders)
# input_size = ... # Number of features
# n_classes = ... # Number of classes (use 1 for binary)
# feature_names = [...] # Optional list of feature names
# class_names = [...] # Optional list of class names

# --- 1. Initialize IFFNN Model ---
# Example: Multi-class with specific class names
model = IFFNN(
    input_size=input_size,
    num_classes=n_classes,
    feature_names=feature_names,  # Optional
    class_names=class_names,      # Optional
    hidden_sizes=None,            # Use default hidden layers
    device='auto'                 # Use CUDA if available
)

# Example: Binary (num_classes=1) with specific class names
# binary_class_names = ["Negative", "Positive"] # Must be length 2
# model = IFFNN(input_size=input_size, num_classes=1, class_names=binary_class_names)

print(model)

# --- 2. Train the Model ---
history = model.train_model(
    train_loader=train_loader,
    valid_loader=valid_loader,
    num_epochs=30, # Adjust as needed
    save_path='best_iffnn_model.pth' # Optional: saves the best model based on validation accuracy
)

# --- 3. Evaluate (Optional) ---
# model.load_state_dict(torch.load('best_iffnn_model.pth')) # Load best model if saved
model.eval()
model.evaluate_model(test_loader)

# --- 4. Get Explanations ---
print("\n--- Explaining a sample batch from the test set ---")
# Get a sample batch (replace with your actual test_loader)
# x_sample_batch, _ = next(iter(test_loader))

# Explain the batch
explanations = model.explain(
    x_sample_batch,   # Your batch of input features
    top_n=5,          # Show top 5 features per class
    print_output=True # Print explanations to console
)

# The explanation output now includes predicted probabilities for the sample.
# The returned 'explanations' list also contains this info programmatically.
# first_sample_probs = explanations[0]['predicted_probabilities']
# print(f"Probabilities for first sample: {first_sample_probs}")

```
It will display the explanations like below:

```
=====

--- Sample 0 (Predicted Probs: 'Type A': 0.15%, 'Type B': 99.70%, 'Type C': 0.15%) ---
=====
  --- Top Contributions towards Class 'Type A' with Probability 0.15% ---
    Rank 1: Feature 'feat_3' (value=-1.2775) -> Contribution=1.8417
    Rank 2: Feature 'feat_5' (value=0.9910) -> Contribution=-1.4437
    Rank 3: Feature 'feat_1' (value=0.7186) -> Contribution=-1.2168
    Rank 4: Feature 'feat_8' (value=-0.7216) -> Contribution=-0.8652
    Rank 5: Feature 'feat_6' (value=-0.8176) -> Contribution=-0.6254
  --- Top Contributions towards Class 'Type B' with Probability 99.70% ---
    Rank 1: Feature 'feat_5' (value=0.9910) -> Contribution=1.9779
    Rank 2: Feature 'feat_13' (value=0.9488) -> Contribution=1.0702
    Rank 3: Feature 'feat_7' (value=0.7568) -> Contribution=0.9619
    Rank 4: Feature 'feat_1' (value=0.7186) -> Contribution=0.5660
    Rank 5: Feature 'feat_8' (value=-0.7216) -> Contribution=0.5482
  --- Top Contributions towards Class 'Type C' with Probability 0.15% ---
    Rank 1: Feature 'feat_3' (value=-1.2775) -> Contribution=-1.6741
    Rank 2: Feature 'feat_12' (value=0.4865) -> Contribution=-0.6309
    Rank 3: Feature 'feat_13' (value=0.9488) -> Contribution=-0.4875
    Rank 4: Feature 'feat_17' (value=-0.9104) -> Contribution=-0.3930
    Rank 5: Feature 'feat_15' (value=0.5226) -> Contribution=0.3745
=====

--- Sample 1 (Predicted Probs: 'Type A': 95.82%, 'Type B': 3.45%, 'Type C': 0.73%) ---
=====
  --- Top Contributions towards Class 'Type A' with Probability 95.82% ---
    Rank 1: Feature 'feat_4' (value=-1.2076) -> Contribution=1.7098
    Rank 2: Feature 'feat_5' (value=0.7826) -> Contribution=-1.0723
    Rank 3: Feature 'feat_6' (value=1.3129) -> Contribution=0.9099
    Rank 4: Feature 'feat_3' (value=-0.6692) -> Contribution=0.8650
    Rank 5: Feature 'feat_9' (value=-1.1353) -> Contribution=0.8623
  --- Top Contributions towards Class 'Type B' with Probability 3.45% ---
    Rank 1: Feature 'feat_4' (value=-1.2076) -> Contribution=-1.4260
    Rank 2: Feature 'feat_5' (value=0.7826) -> Contribution=1.3524
    Rank 3: Feature 'feat_7' (value=-0.9743) -> Contribution=-1.2001
    Rank 4: Feature 'feat_0' (value=-0.6086) -> Contribution=0.6032
    Rank 5: Feature 'feat_14' (value=0.7466) -> Contribution=0.5671
  --- Top Contributions towards Class 'Type C' with Probability 0.73% ---
    Rank 1: Feature 'feat_14' (value=0.7466) -> Contribution=-0.8471
    Rank 2: Feature 'feat_3' (value=-0.6692) -> Contribution=-0.7801
    Rank 3: Feature 'feat_4' (value=-1.2076) -> Contribution=-0.4313
    Rank 4: Feature 'feat_19' (value=-1.2111) -> Contribution=0.3709
    Rank 5: Feature 'feat_12' (value=-0.2503) -> Contribution=0.2781

```


**For a complete, runnable example including data preparation using scikit-learn, please see `examples/simple_usage.py`.**

---

## Citation

If you use this repository or the pre-trained model, please cite our work:

```bibtex
@inproceedings{li2022effectiveness,
  title={On the effectiveness of interpretable feedforward neural network},
  author={Li, Miles Q and Fung, Benjamin CM and Abusitta, Adel},
  booktitle={2022 International Joint Conference on Neural Networks (IJCNN)},
  pages={1--8},
  year={2022},
  organization={IEEE}
}
```

---

## License

Apache 2.0

---


## Disclaimer:

The software is provided as-is with no warranty or support. We do not take any responsibility for any damage, loss of income, or any problems you might experience from using our software. If you have questions, you are encouraged to consult the paper and the source code. If you find our software useful, please cite our paper above.

---

Copyright 2025 (Miles) Qi Li. All rights reserved.
