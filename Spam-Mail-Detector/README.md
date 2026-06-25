# 📧 Spam Mail Detection using Machine Learning

## 📌 Project Overview

This project uses Machine Learning to classify SMS messages as either Spam or Ham (non-spam). The model learns patterns from previously labeled messages and predicts whether a new message is spam.

The project uses Natural Language Processing (NLP) techniques along with the Multinomial Naive Bayes algorithm.

---

## 🎯 Objective

To develop a machine learning model that can automatically identify spam messages and distinguish them from legitimate messages.

---

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF Vectorizer
- Multinomial Naive Bayes

---

## 📂 Dataset

The project uses the SMS Spam Collection Dataset.

Dataset columns:

- v1 → Message Category (Spam/Ham)
- v2 → Message Text

---

## ⚙️ Machine Learning Algorithm

### Multinomial Naive Bayes

Naive Bayes is a probabilistic classification algorithm commonly used for text classification tasks such as:

- Spam detection
- Sentiment analysis
- Document classification

---

## 🔄 Project Workflow

1. Load Dataset
2. Data Cleaning
3. Convert Labels into Numerical Values
4. Text Preprocessing
5. TF-IDF Feature Extraction
6. Train-Test Split
7. Train Naive Bayes Model
8. Predict Messages
9. Evaluate Accuracy

---

## 📊 Model Performance

- Accuracy: 96%

Evaluation Metrics:

- Accuracy Score
- Confusion Matrix

Example Output:

```
Accuracy: 0.96

Message Prediction: SPAM
```

---

## 🚀 Installation

Install the required libraries:

```bash
pip install pandas scikit-learn numpy
```

---

## ▶️ Run the Project

```bash
python spam_detector.py
```

---

## 📥 Example Prediction

Input:

```
Congratulations! You won ₹10,000.
```

Output:

```
SPAM
```

Input:

```
Meeting is scheduled at 4 PM.
```

Output:

```
HAM
```

---

## 🧠 Skills Gained

- Text Preprocessing
- Natural Language Processing
- Feature Extraction
- Machine Learning Classification
- Model Evaluation

---

## 🔮 Future Improvements

- Support Vector Machine (SVM)
- Deep Learning Models
- Email Spam Detection
- Web Application Integration

---

## 👨‍💻 Author

SNIGDHA P

BSc Artificial Intelligence and Machine Learning

QSkill AI & ML Internship 2026
