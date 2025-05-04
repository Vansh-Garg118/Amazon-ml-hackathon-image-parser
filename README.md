# Amazon-ml-hackathon-image-parser

An end-to-end pipeline to extract numeric product attributes (weight, volume, dimensions, etc.) directly from images. Built for the Amazon ML Challenge 2024, this project combines OCR, NLP, and deep learning features to predict entity values with high accuracy.

---

## 🚀 Features

* **OCR Extraction**: Uses EasyOCR to read raw text from product images.
* **Multimodal Feature Fusion**: Combines TF‑IDF, Word2Vec text embeddings and ResNet50 image features.
* **Dual-Model Architecture**:

  * **Value Regressor** (RandomForestRegressor) for predicting numeric amounts.
  * **Unit Classifier** (RandomForestClassifier) for selecting valid units.
* **Regex Post-Processing**: Enforces allowed units and formats predictions (`"x unit"`).


---

## 📦 Technologies & Tools

* **Python** & **Jupyter Notebook**
* **OCR**: EasyOCR
* **NLP**: scikit-learn (TF-IDF), gensim (Word2Vec)
* **Vision**: TensorFlow/Keras (ResNet50)
* **Modeling**: scikit-learn (RandomForestClassifier, RandomForestRegressor)
* **Utilities**: pandas, NumPy, regex, Git


---

## 📁 Dataset

The dataset is publicly available on Kaggle:

[Amazon ML Challenge 2024](https://www.kaggle.com/datasets/abhishekgautam12/amazon-ml-challenge-2024)

---


## 🔍 Methodology

1. **Data Loading & Preprocessing**: Load `train.csv` and `test.csv`, sample data, and download images via `src/utils.download_images()`.
2. **OCR & Text Extraction**: Apply EasyOCR to each image to extract raw text and append it to the dataset.
3. **Text Cleaning & Feature Extraction**: Clean the OCR output (remove punctuation/whitespace), vectorize text using TF‑IDF, and generate Word2Vec embeddings.
4. **Image Feature Extraction**: Extract deep visual features from images using a pre-trained ResNet50 model.
5. **Feature Fusion**: Concatenate text-based and image-based features into a unified feature set for each sample.
6. **Label Engineering**: Use regex to parse numeric values and units from `entity_value`, filter to allowed units (from `constants.py`), and encode units for modeling.
7. **Modeling**:

   * Train a **RandomForestRegressor** for numeric value prediction.
   * Train a **RandomForestClassifier** for unit classification (ensuring valid unit outputs).
8. **Post-Processing & Validation**: Format predictions as `"x unit"`, apply rounding and unit normalization (e.g., `g` → `gram`), and validate the final CSV with `sanity.py` for challenge compliance.

---


## 📊 Results

* **Private Test Accuracy**: 0.72
* **Hackathon Rank**: 62 / 4000+

---


##
