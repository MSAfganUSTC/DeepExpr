# DeepExpr: Facial Expression and Pose Generation via Self-Supervised Disentangled Embeddings Fusion in Text-to-Image Diffusion Models

---

## 📜 Abstract

Text-to-image diffusion models have shown great potential in personalized image synthesis. However, they fall short in providing a dedicated framework for human-centric tasks. Specifically, while they can successfully render a person in new textual contexts or backgrounds, they lack precise control over critical human attributes such as facial expression and head pose.

Recent efforts have attempted to bridge this gap by injecting externally learned embeddings into pre-trained text-to-image models. However, these approaches often fall short in ensuring accurate expression control and pose alignment, primarily due to the limited semantic capacity of the external embeddings. Furthermore, the direct injection of these embeddings tends to degrade facial fidelity and compromise identity preservation.

To overcome these limitations, we propose **DeepExpr**, a framework that enables precise control over facial expression and head pose while preserving identity across diverse visual contexts. DeepExpr comprises two key components:

1. **Semantic and Identity Disentanglement (SID)** — A self-supervised module leveraging frame-to-frame supervision to disentangle identity and expression representations.
2. **Multi-Embedding Fusion (MEF)** — A mechanism to integrate these embeddings into pre-trained diffusion models without degrading fidelity.

---

## 🧠 Framework Overview

<p align="center">
  <img src="Images/DeepExpr_Framework.png" alt="DeepExpr Framework" width="800"/>
</p>

> **Figure:** Overview of the DeepExpr framework showing disentangled fusion and expression control from reference input.

---
## 🧾 Framework Comparison Table
<p align="center"> <img src="Images/Framwork_Comparison.png" alt="Evaluation Table: DeepExpr vs Existing Methods" width="700"/> </p>
**Figure:** Evaluation of recent facial synthesis methods based on their ability to preserve identity, control expression and pose, and integrate contextual information. DeepExpr offers comprehensive control and fidelity compared to existing methods across all attributes.
---
## 🧪 Module-wise Results and Comparisons

### 🧩 SID Module: Identity and Semantic Disentanglement

<p align="center">
  <img src="Images/SID_Comparison.png" alt="SID Comparison" width="800"/>
</p>

> **Figure:** Results from the SID module demonstrating identity preservation across expressions.

---

### 🧬 SID Evaluation: Cross-Renactment with Static and Video Driving

<p align="center">
  <img src="Images/SID_Table_Cross_renechmnet_bothvideoand images.png" alt="Cross-Renactment Evaluation" width="800"/>
</p>

> **Figure:** SID performance comparison on static image and video-based cross-expression reenactment tasks.

---

### 🌀 Inference Pipeline

<p align="center">
  <img src="Images/DeepExpr_inference.png" alt="DeepExpr Inference Pipeline" width="800"/>
</p>

> **Figure:** End-to-end inference flow using source image and expression/pose.

---

## ⚔️ DeepExpr vs Personalized and Expression-Centric Models

### DeepExpr vs Personalized Diffusion Models

<p align="center">
  <img src="Images/DeepExprvsPersonalized.png" alt="Comparison with Personalized Diffusion Models" width="800"/>
</p>

> **Figure:** Identity-preserving facial expression and pose control with contextual background integration in personalized text-to-image diffusion models.

### DeepExpr vs Expression Editing Methods

<p align="center">
  <img src="Images/DeepExprvsExpressio.png" alt="Comparison with Expression Editing Methods" width="800"/>
</p>

> **Figure:** Identity-preserving Expression accuracy and accurate pose alignment comparison with existing expression-control techniques.

---
## 📈 Comparison with Existing Personalized and Facial Expression Models

<p align="center">
  <img src="Images/DeeExpr_Table_personalized and FaceExpression methods .png" alt="Comparison Table with Baselines" width="800"/>
</p>

> **Figure:** Quantitative comparison of DeepExpr with baseline personalized and facial expression generation methods.

---

## 🔍 Reproducibility and Results

DeepExpr consistently outperforms prior models across:
- Identity preservation
- Expression accuracy
- Pose alignment
- Context integration

It works effectively on:
- Personalized image generation
- Emotion behavior simulation
- Forensic reconstruction using minimal supervision.

---
# Supplementary Results for DeepExpr

## 1. Additional Qualitative Examples

Below are several examples demonstrating the expression and pose control of DeepExpr on diverse identities.
## Visual Results

### Example 1: Identity-Preserving Pose and Expression Variation Results for Male Subjects

![Figure 3 - Male Results](images/png/Figure_3_More_results_Male_page1.png)  
_Figure 1: Identity-preserving pose and expression variation results for male subjects._

---

### Example 2: Identity-Preserving Pose and Expression Variation Results for Female Subjects

![Figure 4 - Female Results](images/png/Figure_4_More_results_Female_page1.png)  
_Figure 2: Identity-preserving pose and expression variation results for female subjects._

---

### Example 3: Identity-Preserving Pose and Expression Variation Results for Young Boys and Girls Subjects

![Figure 5 - Kids Results](images/png/Figure_5_More_results_Kids_page1.png)  
_Figure 3: Pose and expression variation results for young boys and girls from diverse ethnic groups._

---

### Example 4: Identity-Preserving Pose and Expression Variation Results for Multiple Ethnic Groups

![Figure 6 - Ethnic Results](images/png/Figure_6_More_results_Ethinic_page1.png)  
_Figure 4: Pose and expression variation results across multiple ethnic groups._

---

### Example 5: Identity-Preserving Pose and Expression Variation Results for Cross-Identity and Reference Inputs

![Figure 7 - Cross Identity Results](images/png/Figure_7_More_results_cross_page1.png)  
_Figure 5: Results showing cross-identity and reference input variations._

---

### Example 6: Identity-Preserving Pose and Expression Variation for Extreme Orientations for both Reference and Identity Inputs

![Figure 8 - Extreme Orientation Results](images/png/Figure_8_More_results_extream_page1.png)  
_Figure 6: Results for extreme pose and expression orientations on both reference and identity inputs._

---

## 3. Ablation Study: Multi-Embedding Fusion Impact

| Fusion Strategy          | Identity Preservation (%) | Expression Accuracy (%) | Pose Alignment Error (%) | Face FID (%) |
|--------------------------|---------------------------|------------------------|---------------------------|--------------|
| Direct Injection         |            85.3           |           78.5         |             50            |      70     |
| **DeepExpr MEF (Ours)**  |           **90**          |         **84.4**       |            **86**         |      85     |

*Table 2: Ablation results demonstrating the effectiveness of DeepExpr's multi-embedding fusion.*

---
## 4. Dataset Details and Preprocessing

- We used three distinct datasets for training our Semantic and Identity Disentanglement (SID) encoding module:  
  **HDTF** [57], **VoxCeleb** [58], and **VFHQ** [59].  
- Original videos were carefully selected and processed consistently.  
- Preprocessing included filtering out blurred faces and extreme head angles to ensure quality training data.
- 
- The results presented across multiple figures demonstrate DeepExpr’s strong performance and versatility across diverse categories, ethnicities, age groups, and extreme head orientations. It successfully preserves identity while generating realistic and varied facial expressions with accurate pose alignment.  
- Identity images from the **CelebA-HQ** [28] dataset and reference images from **AffectNet** [29] were used to generate images with different expressions.

### Preprocessing Instructions

- Preprocessing scripts are available in the `/preprocess` folder.  
- To get face crop suggestions for videos, use:  
  ```bash
  python crop-video.py --inp path/to/video.mp4


## 5. Practical Applications

- Emotion recognition datasets integration.
- Forensic facial reconstruction support.
- Facial behavior simulation for AI training.

---

# Contact & Support

For questions, issues, or collaboration requests:  
**[Muhammad Sher Afgan]** — [msafgan@mail.ustc.edu.cn]  
GitHub: [https://github.com/yourusername/DeepExpr](https://github.com/yourusername/DeepExpr)  

---
