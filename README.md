### Fake News Detection Solution

#### Introduction
This document details the methodology and implementation used to predict labels (fake, biased, or true) for a French-language fake news dataset related to climate change. The dataset includes a labeled `train.csv` file and an unlabeled `test.csv` file where predictions are needed.

---

#### Dataset Description

1. **Train Dataset**:
   - **Columns**: `Text`, `Label`
   - **Text**: Contains the content of news articles.
   - **Label**: Manually annotated with one of three classes: `fake`, `biased`, or `true`.

2. **Test Dataset**:
   - **Columns**: `Text`
   - The `Label` column is generated by the model based on predictions.

---

#### Prerequisites

Before proceeding, install the required Python packages using:

```bash
pip install -r requirements.txt
```

---

#### Implementation Steps

##### 1. **Text Preprocessing**
To prepare the dataset for modeling, the text is preprocessed as follows:
   - Convert all text to lowercase.
   - Remove special characters, digits, and extra whitespace.
   - Eliminate French stopwords using the NLTK library.

##### 2. **Feature Extraction**
A Term Frequency-Inverse Document Frequency (TF-IDF) vectorizer is applied to convert preprocessed text into numerical representations. The vectorizer is configured to retain the 5,000 most frequent terms to ensure efficient processing.

##### 3. **Model Selection and Training**
A Logistic Regression model is selected for its effectiveness in text classification tasks. The following parameters are used:
   - **Hyperparameters**: `max_iter=1000`, `random_state=42`
   - Training is performed using the TF-IDF features and corresponding labels from the training data.

##### 4. **Prediction**
The trained model is utilized to predict labels for the test dataset based on its TF-IDF-transformed text data.

##### 5. **Output**
The predictions are added as a `Label` column in the test dataset and saved to a new CSV file, `test_predictions.csv`.

---

#### Key Libraries
- **Pandas**: For data manipulation and analysis.
- **Scikit-learn**: For feature extraction and machine learning models.
- **NLTK**: For text preprocessing, including French stopwords.

---

#### Results
The output file, `test_predictions.csv`, contains the original `Text` column from the test dataset along with the predicted `Label` column indicating whether the article is `fake`, `biased`, or `true`.

---

#### Code Repository
Access the full implementation on GitHub: [GitHub Repository](https://github.com/nickcatalin/SII_Homework).
