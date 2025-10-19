# Fake Arabic News Detection

This project focuses on detecting **fake vs. real Arabic news articles** using Natural Language Processing (NLP) and Machine Learning techniques.  
It is implemented in a single Jupyter Notebook: **`Arabic_fake_news_detection.ipynb`**, and uses the **AFND (Arabic Fake News Dataset)** from Kaggle as the main data source.

---

## Project Overview

The notebook performs all key steps for fake news detection:

1. **Data loading** from the AFND dataset.  
2. **Arabic text preprocessing** (diacritics removal, normalization, tokenization, stopword filtering).  
3. **Feature extraction** using **TF-IDF** and an experimental custom weighting method (ATCF-IDF).  
4. **Model training & evaluation** using classic ML algorithms:
   - Logistic Regression  
   - Linear Support Vector Machine (SVM)  
   - Multinomial Naive Bayes  
5. **Performance evaluation** using accuracy, precision, recall, F1-score, and confusion matrix visualization.

---

## Dataset

- **Name:** [AFND – Arabic Fake News Dataset](https://www.kaggle.com/datasets/mostafamohamed/afnd-arabic-fake-news-dataset)
- **Description:** A collection of Arabic news articles labeled as *Fake* or *Real*.
- **Columns:**  
  - `text` → news article text  
  - `label` → 0 (Real) or 1 (Fake)

Download the dataset from Kaggle and place the CSV file inside a folder named **`data/`** in the same directory as the notebook.

---

## Repository Structure
├── Arabic_fake_news_detection.ipynb # Main notebook (preprocessing, training, evaluation)

├── data/
│ 
└── AFND.csv # Place your downloaded dataset here

└── README.md

---

## Installation

Tested on **Python 3.10+**.  
Install the required libraries before running the notebook:

```bash
pip install -U pandas numpy scikit-learn matplotlib seaborn nltk camel-tools tqdm bs4
```
## Outputs

The notebook prints and visualizes:

- Dataset balance  
- TF-IDF vector shape  
- Accuracy, precision, recall, and F1-score  
- Confusion matrix heatmap  
- Optional comparison between **TF-IDF** and **ATCF-IDF**

---

## 🔧 Customization

- **Switch models:** change the classifier (e.g., `LinearSVC()`, `LogisticRegression()`, `MultinomialNB()`).  
- **Tweak TF-IDF:** modify parameters like `max_features`, `ngram_range`, or `min_df`.  
- **Add more preprocessing:** adjust normalization, stemming, or stopword lists for Arabic text.

---

## Technologies Used

- **Python**  
- **Pandas**, **NumPy**  
- **Scikit-learn** → ML models and metrics  
- **NLTK** & **Camel Tools** → Arabic NLP preprocessing  
- **Matplotlib**, **Seaborn** → data visualization

---

## Acknowledgments

- [Kaggle AFND Dataset](https://www.kaggle.com/datasets/mostafamohamed/afnd-arabic-fake-news-dataset)  
- [Camel Tools](https://github.com/CAMeL-Lab/camel_tools) for Arabic NLP processing  


