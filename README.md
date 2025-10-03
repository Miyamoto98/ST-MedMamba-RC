# Spatiotemporal-Enhanced MedSAM2-Mamba Framework: Contextual Learning for Multi-Modal MRI-Based Rectal Cancer Staging

## Overview

This project aims to build a **Spatiotemporal-Enhanced MedSAM2-Mamba Framework** for efficient and accurate rectal cancer staging using multi-modal MRI through contextual learning. We leverage advanced deep learning models like MedSAM2 and Mamba, deeply optimizing and applying them to this specific medical task.

*Note: In the context of this project, "Spatiotemporal" is a compound concept. "Spatio" refers to the features within each 2D slice, while "temporal" is an analogous term, following industry convention, that refers to the sequential relationship of slices along the depth axis (Z-axis). Our model treats this depth sequence as a time series to capture 3D context.*

## Core Methodology

Our model is designed with a multi-stage architecture to process and fuse multi-modal 3D MRI data, ultimately outputting a precise staging prediction for rectal cancer.

### 1. Base Feature Extraction

-   **Input**: The model takes paired 3D MRI data from T2-weighted (T2W) and Diffusion-weighted Imaging (DWI) modalities.
-   **2D Encoder**: We utilize the powerful **MedSAM2 Image Encoder** as the base feature extractor. The 3D MRI volume is processed slice by slice, with each 2D slice fed into the encoder to extract its deep spatial features. This step lays the foundation for subsequent spatiotemporal analysis and modal fusion.

### 2. From Image to Sequence: Feature Transformation and Construction

To enable subsequent modules like MemoryBlock and Mamba to process the data, we need to convert the spatial features extracted by the 2D encoder into a sequence. This process does not reconstruct a 3D image from feature maps; instead, it builds a **1D feature sequence** that represents the entire 3D volume.

The process is as follows:

1.  **Input Image Dimensions**: A 3D MRI volume has dimensions of `(B, C, D, H, W)`, where `B` is the batch size, `C` is the number of channels, `D` is the slice depth, and `H` and `W` are the height and width of the slices.

2.  **Slice-wise Feature Extraction**: The 2D Image Encoder processes each of the `D` slices independently. For each 2D slice, the encoder outputs a feature map of dimensions `(B, C_feat, H_feat, W_feat)`, where `C_feat` is the number of feature channels (e.g., 256), and `H_feat` and `W_feat` are the reduced dimensions of the feature map.

3.  **Flatten and Concatenate**: This is the crucial step. We **do not** stack these feature maps back into a 3D volume. Instead, we **flatten** the spatial dimensions (`H_feat`, `W_feat`) of each feature map into a vector. Then, we **concatenate** the flattened vectors from all `D` slices along the sequence dimension.

4.  **Final Sequence Format**: The final output of this process is a tensor with dimensions `(B, L, C_feat)`.
    -   `B`: Batch size, which remains unchanged.
    -   `L`: The total sequence length, equal to `D * H_feat * W_feat`. It represents the collection of all spatial positions within the entire 3D volume.
    -   `C_feat`: The feature dimension for each position.

This `(B, L, C_feat)` formatted feature sequence is the input data processed by the subsequent `MemoryBlock` module. It successfully transforms the 3D spatial information of the original image into a 1D sequential structure suitable for sequence models like Transformers or Mamba.

#### Memory Optimization: Slice Batching

Given that processing all slices of an entire 3D volume at once would incur significant memory overhead, the model implements a **Slice Batching** optimization strategy internally. During the feature extraction phase, instead of computing features for all slices at once, the model divides them into smaller mini-batches (controlled by the `slice_batch_size` parameter). Through this "divide and conquer" approach, the model significantly reduces peak memory usage during computation, making it feasible to process large-scale 3D images on hardware with limited resources. The `slice_batch_size` parameter can be configured during training and inference to balance speed and memory consumption.

### 3. Spatiotemporal Context Enhancement (`MemoryBlock`)

-   The feature sequence extracted from the 2D encoder is fed into our custom-designed **MemoryBlock** module.
-   This module captures contextual dependencies between slices in the sequence via **spatiotemporal self-attention**, effectively integrating spatial and temporal (sequential) information within the 3D volume.
-   Simultaneously, by performing cross-attention with a learnable **prototype memory**, the MemoryBlock enhances the semantic representation of the features, making them more sensitive to pathological characteristics related to rectal cancer.

### 4. Innovative Multi-Modal Fusion (`BGAMF`)

This is the core innovation of our methodology. We designed and implemented a **Bidirectional Gated & Alignment-aware Mamba Fusion (BGAMF)** framework, which replaces traditional fusion methods like simple concatenation or attention. BGAMF deeply fuses information from T2W and DWI modalities through a refined three-step process:

1.  **Alignment-aware Feature Mapping**: Before fusion, we first align the features of the two modalities using a **cross-attention module**. T2W features are adjusted using DWI as context, and vice versa. This step resolves potential spatial or semantic discrepancies between the multi-modal data, ensuring that the subsequent fusion is performed on highly correlated and aligned features.

2.  **Bidirectional Mamba Modeling**: The aligned feature sequence of each modality is processed by a **Bidirectional Mamba (Bi-Mamba)** block. Through forward and backward scans, the model efficiently captures global long-range dependencies within each sequence at linear complexity, achieving a comprehensive contextual understanding of the entire 3D volume.

3.  **Gated Fusion Mechanism**: The two modality features, after being processed by Bi-Mamba, are concatenated and fed into a **gating unit** (a linear layer followed by a Sigmoid activation function). This unit dynamically learns a weight (gate) to adaptively control the contribution of T2W and DWI information at each feature position, following the formula: `f_fusion = gate * h_t2w + (1 - gate) * h_dwi`. This mechanism allows the model to intelligently select and prioritize the more informative modality at specific spatial locations, leading to a more refined and effective feature fusion.

### 5. Classification Prediction

-   The unified feature sequence, after being deeply fused by the BGAMF module, is further refined by a final **post-fusion Bi-Mamba block**.
-   Finally, the refined feature sequence is passed to a **Classification Head** (Global Average Pooling layer + Fully Connected layer) to output the final rectal cancer staging prediction.