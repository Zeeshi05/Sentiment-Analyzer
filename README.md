# 🛍️Sentiment Analysis AI Project

## 🚀 Overview

In the world of e-commerce, customer reviews hold immense power. Understanding the sentiment behind reviews helps businesses improve products and customers make informed choices.  
**This project leverages machine learning to classify Amazon product reviews as Positive or Negative—instantly and interactively!**

---

## 🎯 Project Highlights

- **Two Powerful ML Models:** Logistic Regression with TF-IDF and Bernoulli Naive Bayes with Count Vectorizer.  
- **Interactive Gradio App:** Try both models live in your browser—no coding required!  
- **Data Preprocessing:** Includes tokenization, stopword removal, and lemmatization with NLTK.  
- **Handling Imbalanced Data:** Uses Random Over Sampling for balanced training.  
- **Visual Insights:** WordCloud visualizations for positive and negative reviews.  
- **Clean & Reproducible:** Well-structured code with clear instructions.  

---

## 📊 Why Sentiment Analysis Matters?

> “Customer feedback is the heartbeat of any product’s success. Understanding sentiment helps businesses adapt and thrive.”  
> — *Project Motivation*

Sentiment analysis can help:

- Improve product quality  
- Enhance customer satisfaction  
- Automate review monitoring  
- Drive data-informed business decisions  

---

## 🛠️ Features

- **Text Preprocessing:** Cleans and normalizes review text for better model performance.  
- **Multiple Vectorization Techniques:** TF-IDF and Count Vectorizer for flexible feature extraction.  
- **Model Training & Evaluation:** Includes classification reports with precision, recall, and F1-score.  
- **Imbalanced Data Handling:** Balances classes using RandomOverSampler.  
- **Gradio Web Interface:** Input reviews and select your preferred model for real-time sentiment prediction.  

---

## 📦 Dataset

- **Source:** `Amazon_Reviews.csv` (must be placed in project root)  
- **Content:** Amazon product reviews with ratings and review text  
- **Labels:** Positive (rating ≥ 4), Negative (rating ≤ 2), neutral reviews excluded  

---

## 💻 Quickstart

1. **Clone the Repo**

   ```bash
   git clone https://github.com/Zeeshi05/Sentiment-Analyzer
   ```

2. **Install Requirements**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Dataset**

   * Place `Amazon_Reviews.csv` in the project root directory.

4. **Run the Notebook**

   * You can run this notebook using collab 

5. **Launch the Gradio App**

   * Run the Gradio interface cell.  
   * Enter any review text and select the model to get instant sentiment predictions.

---

## 🧠 Model Performance

| Model                   | Accuracy | Precision | Recall   | F1-Score |
| ----------------------- | -------- | --------- | -------- | -------- |
| Bernoulli Naive Bayes   | 94%      | High      | High     | High     |
| Logistic Regression     | 82%      | Moderate  | Moderate | Moderate |

*Bernoulli Naive Bayes generally performs better in this project.*

---

## 🏗️ Project Structure

```
amazon-review-sentiment/
│
├── Amazon_Reviews.csv             # Dataset file (not included)
├── Untitled6.ipynb                # Main notebook with preprocessing, modeling, and Gradio interface
├── sentiment_pipeline.pkl         # Saved Logistic Regression model pipeline
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── images/                       # WordCloud images and other visuals (optional)
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to fork the repo and submit a pull request.

---

## 📢 Credits

* **Dataset:** Amazon product reviews (Kaggle)  
* **Libraries:** NLTK, Scikit-learn, Imbalanced-learn, Gradio, WordCloud  
* **Developed by:** Muhammad Zeeshan

---

**Empower your business with insightful sentiment analysis!**

---

*Star ⭐ this repo if you find it useful!*

---

**Notes:**

- Replace `https://github.com/yourusername/amazon-review-sentiment.git` with your actual repo URL.  
- Update model performance metrics with your real results.  
- Add images to the `/images` folder if you want to display WordClouds or screenshots.  
- Customize author name and any other details as needed.  
- If you want, I can also help you generate a `requirements.txt` file or add badges!
