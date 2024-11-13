# Fake-News-Detection

This project aims to detect fake news using machine learning techniques. It utilizes a dataset of news articles classified as fake or real, applies text preprocessing, and trains multiple models to predict the authenticity of news articles.

## 📋 Project Explanation and Key Features
This project utilizes machine learning techniques to classify news articles. The key features include:

1. **Data Cleaning and Preprocessing**  
   The text data is cleaned by removing URLs, special characters, punctuation, and numbers.

2. **Text Vectorization**  
   The cleaned text is transformed into numerical format using **TF-IDF Vectorization**, capturing important text features.

3. **Model Training**  
   Three machine learning models are used to classify the articles:
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier

4. **Manual Testing**  
   Users can input their own news articles, and the system predicts whether they are fake or real.

5. **Accuracy Evaluation**  
   The models are evaluated using metrics like accuracy, precision, recall, and F1-score.

---

## 🛠️ Technology Stack
- **Language:** Python
- **Libraries:**
  - **Scikit-learn:** For machine learning models and evaluation
  - **Pandas, NumPy:** For data processing and analysis
  - **TfidfVectorizer:** For text vectorization
  - **Matplotlib, Seaborn:** For data visualization

---

## 🧑‍💻 Team and Role
This project was completed individually. My responsibilities included:
- Data collection and preparation
- Data preprocessing and text cleaning
- Model training and evaluation
- Implementing a user interface for manual testing

---

## 💻 Why I Chose This Tech Stack
- **Python** was selected for its strong support for data processing and machine learning.
- **Scikit-learn** was chosen for its efficient implementation of various machine learning algorithms.
- **TfidfVectorizer** was used for effective text feature extraction, allowing the model to understand the text data better.

---

## 🚧 Challenges Faced
1. **Data Preprocessing Issues:**  
   The dataset contained noisy text (e.g., URLs, special characters). I developed custom functions to clean the data effectively.

2. **Model Overfitting:**  
   Decision Trees were prone to overfitting. I mitigated this issue using hyperparameter tuning and simpler models like Logistic Regression and Random Forest, which provided better generalization.

3. **Manual Testing Challenges:**  
   Handling unstructured user input required refining the text-cleaning process to ensure accurate predictions.

---

## 🔄 End-to-End Project Flow
1. **Data Collection:** Load datasets containing fake and true news articles.
2. **Preprocessing:** Clean the text data by removing unnecessary elements (e.g., URLs, punctuation).
3. **Text Vectorization:** Use **TF-IDF Vectorizer** to convert text into numerical features.
4. **Model Training:** Train models like Logistic Regression, Decision Tree, and Random Forest.
5. **Evaluation:** Assess the models using accuracy and classification reports.
6. **Manual Testing:** Allow users to input their own articles and classify them as fake or real.

---

## 🔍 Future Improvements
1. Integrate real-time data from live news sources for immediate detection.
2. Explore deep learning models like **BERT** for improved accuracy.
3. Add **multi-language support** for broader applicability.
4. Include **sentiment analysis** to assess the tone of the news (e.g., biased, neutral).

---
