# Colour-Invariant Saree Design Recognition

Pretrained **ResNet34 + batch-hard triplet loss + strong colour augmentation** (PyTorch).
Everything (data cleaning → training → evaluation → inference) is in one notebook:
**[saree_color_invariant_reid.ipynb](saree_color_invariant_reid.ipynb)**, which is saved with its outputs.

```
Image → colour augmentation → ResNet34 (ImageNet) → GeM → 256-D → BN → L2
Training : anchor (design A, colour 1) / positive (A, colour 2) / negative (B, SAME colour 1) → triplet loss
Inference: query embedding → cosine similarity → gallery → Top-K designs   |   pair → cosine ≥ τ → same design
```

## Approach note (489 chars)
We use a pretrained ImageNet ResNet34 as a shared embedding encoder. Images are resized and color-augmented using brightness, saturation, hue and grayscale transformations. A 256-D normalized embedding is learned with Triplet Loss using same-design different-colour positives and different-design negatives. PK batches deliberately share colourways across designs to create hard same-colour negatives. Retrieval uses cosine similarity, with validation-selected thresholds for verification.

## Results
Test set: 189 designs that never appear in training (25 Roboflow + 164 DeepLure). Each query is in a palette that differs from every gallery palette of its design, and the k-means recolouring operator is never used in training. The threshold τ is picked on validation.

| Method | R@1 | R@5 | mAP | ROC-AUC | AUC same-palette | EER |
|---|---|---|---|---|---|---|
| RGB colour histogram | 0.011 | 0.048 | 0.026 | 0.421 | 0.000 | 0.541 |
| ImageNet ResNet34 (no training) | 0.262 | 0.482 | 0.234 | 0.686 | 0.515 | 0.353 |
| **Ours** | **0.476** | **0.743** | **0.408** | **0.900** | **0.853** | **0.194** |

* R@1 per source: Roboflow 0.98, DeepLure 0.40 (out-of-domain).
* Efficiency: 21.4 M parameters, 7.3 GFLOPs @224, 256-D embedding (512 B fp16).
* This run used the CPU budget (15 epochs, P=16). On a GPU the notebook switches to 30 epochs, P=32 automatically.

## Run it
1. Kaggle: add the dataset `div456/indian-saree-patterns`, and add the DeepLure corpus as a **private** dataset. Turn the GPU on and choose Run All. Data paths are detected automatically.
2. Locally: `pip install -r requirements.txt`, then set `DATA_DIR` in step 02 to your data folder.

## Data handling and disclosures
* No colourway labels exist, so a **design is one source photo**. Roboflow augmentation copies are merged, and dHash near-duplicates (Hamming ≤ 6) are merged too. 87 duplicate groups that straddled the original splits were moved to train, so val and test are unseen.
* Pretrained weights: torchvision `ResNet34_Weights.IMAGENET1K_V1`. Data: Kaggle *Indian Saree Patterns* (Roboflow, MIT) and the DeepLure corpus. No external data.
* The DeepLure corpus is proprietary. It is **not** included in this repository, and figures that show its images were removed from the saved outputs.
