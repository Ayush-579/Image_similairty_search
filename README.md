# 📌 Google Lens Alternative - Image Similarity Search  
### **A Comparative Study of Multiple Approaches for Image Similarity Search**  

---

## 1. Project Overview  
This project develops an alternative to Google Lens by implementing **image similarity search** using multiple deep learning-based approaches. Models including Autoencoders, ResNet50, VGG16, and Vision Transformers (ViT) are trained and evaluated on the **FashionMNIST dataset** to measure precision, recall, F1-score, and computational efficiency.  

---

## 2. Objectives  
- Implement and compare various neural networks for **image similarity search**.  
- Fine-tune models on the **FashionMNIST dataset** and analyze their performance.  
- Evaluate results based on **precision, recall, F1-score, and LRAP (Label Ranking Average Precision)**.  
- Provide a scalable and computationally efficient solution for real-time image similarity search.  

---

## 3. Approaches Implemented  
1. **Autoencoder**  
2. **ResNet50**  
3. **VGG16**  
4. **Vision Transformer (ViT)**  

Each model was trained and evaluated independently to determine the most efficient and accurate solution for image similarity tasks.  

---

## 4. Dataset  
**Dataset:** FashionMNIST  
- **Training:** 60,000 images  
- **Testing:** 10,000 images  
- **Classes:** 10 (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)  
- **Image Size:** 28x28 Grayscale  

---

## 5. Results  

| **Model**          | **Precision** | **Recall**  | **F1-Score** | **LRAP**  | **Accuracy** |  
|--------------------|---------------|-------------|--------------|-----------|--------------|  
| **Autoencoder**    | 0.7501        | 0.0150      | 0.0294       | 0.8623    | 1.0000       |  
| **ResNet50**       | 0.7606        | 0.0152      | 0.0298       | 0.8637    | 1.0000       |  
| **VGG16**          | 0.7491        | 0.0150      | 0.0294       | 0.8555    | 1.0000       |  
| **ViT**            | 0.7906        | 0.0158      | 0.0310       | 0.8804    | 1.0000       |  

---

## 6. Key Insights  
- **ViT (Vision Transformer)** outperforms other models in **precision, recall, and LRAP**, demonstrating its superiority in capturing complex features.  
- **ResNet50** follows closely behind ViT in performance.  
- All models achieved **100% accuracy**, reflecting the simplicity of the dataset; however, low recall suggests room for improvement through data augmentation and larger datasets.  
- **Autoencoder** and **VGG16** showed competitive performance but lag slightly behind ResNet50 and ViT.  

---

## 7. Project Structure  
Google Lens Alternative - Image Similarity Search/ │
```bash
├── data/ # Dataset (FashionMNIST - Automatically Downloaded)
├── models/ # Saved Models (Autoencoder, ResNet50, VGG16, ViT)
│ ├── autoencoder.pth
│ ├── resnet50.pth
│ ├── vgg16.pth
│ └── vit.pth
├── notebooks/ # Jupyter Notebooks for EDA and Training
│ └── Image_Similarity.ipynb
├── src/ # Source Code
│ ├── Autoencoder.py
│ ├── Resnet50.py
│ ├── VGG16.py
│ ├── Vit_b.py
├── results/ # Evaluation Reports
│ └── evaluation_report.txt
├── Requirements.txt # Project Dependencies
└── README.md # Documentation
```

---

## 8. How to Run the Project  
### **Step 1: Clone the Repository**  
```bash
git clone git@github.com:Ayush-579/Image_similarity_search.git
cd Image_similarity_search
```

---

### **Step 3: Install Dependencies**
```bash
pip install -r requirements.txt
```
---
###**Step 4: Train and Save Models**
```bash
python __main__.py --save_models
```
---
## 11. Future Improvements
- Data Augmentation – Improve recall and overall performance.
- Larger Datasets – Train on datasets like ImageNet or CIFAR-100.
- Web Interface – Develop a simple web app for image uploads and similarity search in real time.
---
## 12. References  
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)  
- [Vision Transformer (ViT) by Dosovitskiy et al.](https://arxiv.org/abs/2010.11929)  
- [ResNet Research Paper](https://arxiv.org/abs/1512.03385)  
- [FashionMNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [Medium Articles](https://medium.com/@meetdheerajreddy/fashion-mnist-analysis-classifying-fashion-with-deep-learning-0ba793ba5234) 

