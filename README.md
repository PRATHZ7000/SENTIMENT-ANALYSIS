# SENTIMENT-ANALYSIS

*COMPANY* : CODTECH IT SOLUTIONS

*NAME* : PRATHAMESH MURKUTE

*INTERN ID* : CT06DF2317

*DOMAIN* : DATA ANALYTICS

*DURATION* : 6 WEEKS

*MENTOR* : NEELA SANTOSH

## README: Sentiment Analysis Using Walmart Dataset on Google Colab
# Project Title:
Sentiment Analysis on College Dataset (Walmart Related Project)

# Overview:
This project performs sentiment analysis using textual data from a college dataset, sourced from Kaggle. The analysis is conducted entirely in Google Colab, leveraging Python libraries for preprocessing, modeling, and evaluation. The primary aim of this task was to classify the college type (Private or Public) based on textual features extracted from the dataset. Though the dataset's name suggests a focus on Walmart, the loaded dataset is related to colleges and institutions.

# Tools & Environment:
Platform: Google Colab

Language: Python 3

# Data Source: Kaggle (Walmart dataset - specifically "College.csv")

Libraries Used:

pandas, numpy for data handling

nltk for text preprocessing

sklearn for machine learning modeling and evaluation

matplotlib, seaborn for visualization

# Dataset Description:
The dataset used was College.csv, downloaded from Kaggle. It contains various attributes of colleges, including a column labeled Unnamed: 0 that was treated as textual input and Private as the target label indicating whether the college is private or not.

# Workflow:
Data Loading:

The dataset was read into a DataFrame using pandas.read_csv().

Basic checks for null values were conducted, and rows with missing textual data were considered (though the script notes that the cleaning line for dropping missing text was commented out).

# Preprocessing:

Text data was cleaned by removing non-alphabetic characters and converting all text to lowercase.

Tokenization was performed, followed by stop word removal using nltk and stemming via the Porter Stemmer.

The clean tokens were rejoined into processed text, stored in a new column clean_text.

Feature Engineering:

TF-IDF Vectorization was applied to convert textual data into numerical vectors.

A limit of 5,000 features was set for the vectorizer.

# Label Encoding:

The Private column was label-encoded using LabelEncoder from sklearn.

Model Training and Testing:

The dataset was split into training and test sets using an 80/20 split.

A Multinomial Naive Bayes classifier was trained on the training data.

Predictions were made on the test set and evaluated using a classification report and confusion matrix.

# Visualization:

A count plot using seaborn showed the distribution of college types (Private vs. Public).

# Results:
The MultinomialNB classifier achieved results summarized in a classification report, including precision, recall, F1-score, and accuracy metrics for each class. The project provides a baseline implementation of text classification using basic NLP and machine learning techniques.

# Conclusion:
This project demonstrates the application of sentiment analysis using TF-IDF and Naive Bayes for binary classification. Despite being titled as a Walmart project, it utilized a college dataset. Nonetheless, the core techniques are adaptable for similar real-world sentiment or text classification tasks.

# Future Work:
Improve preprocessing by adding lemmatization.

Try advanced models like Logistic Regression or SVM.

Use word embeddings for richer text representation.

Apply on actual Walmart review datasets for more relevant sentiment insight.

# Technologies Used
•	Python 3.10+
•	Google Colab (for development and execution)
•	Libraries:
o	pandas
o	numpy
o	matplotlib (if visualization is included)

# How to Run
1.	Clone the repository:
git clone https://github.com/yourusername/your-repo-name.git
Open the .py script in Google Colab or Jupyter Notebook.
Upload the TripAdvisor dataset (tripadvisor_hotel_reviews.csv) if not already available.
Run all cells to view the analysis and results.

# Author
Name: Prathamesh Murkute
Task: SENTIMENT ANALYSIS - Task 4
Platform: Google Colab

# License
This project is licensed under the MIT License.
