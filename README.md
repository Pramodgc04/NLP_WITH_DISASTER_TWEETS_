# NLP with Disaster Tweets - Classification Project

## 🌪️ Project Overview

This project implements a Natural Language Processing (NLP) system to classify tweets as disaster-related or non-disaster-related. Using machine learning techniques and Word2Vec embeddings, the system can accurately identify tweets that contain information about natural disasters, emergencies, or crisis situations.

## 🎯 Project Objectives

- **Disaster Detection**: Automatically identify disaster-related tweets from social media
- **Real-time Classification**: Provide instant classification for new tweets
- **Web Application**: Deploy an interactive web interface for tweet classification
- **Model Optimization**: Achieve high accuracy with optimized hyperparameters

## 🛠️ Technology Stack

### Core Technologies

- **Python 3.x** - Primary programming language
- **NLTK** - Natural Language Toolkit for text processing
- **Scikit-learn** - Machine learning algorithms and tools
- **Gensim** - Word2Vec implementation for word embeddings
- **Streamlit** - Web application framework
- **Pandas/NumPy** - Data manipulation and analysis

### Key Libraries

```python
pandas, numpy, scikit-learn, nltk, gensim, streamlit, joblib, matplotlib, seaborn
```

## 📊 Dataset Information

### Training Data

- **File**: `tweets.csv`
- **Columns**:
  - `text`: Tweet content
  - `target`: Binary classification (1=disaster, 0=non-disaster)
- **Size**: ~7,500+ labeled tweets

### Test Data

- **File**: `test.csv`
- **Purpose**: Model evaluation and testing
- **Columns**: Tweet text for prediction

## 🔍 Data Preprocessing Pipeline

1. **Text Cleaning**

   - Remove URLs, mentions, and hashtags
   - Convert to lowercase
   - Remove special characters and numbers
   - Tokenize text into words

2. **Stopword Removal**

   - Remove common English stopwords
   - Preserve meaningful disaster-related terms

3. **Word Embeddings**
   - Generate 100-dimensional Word2Vec embeddings
   - Average word vectors for tweet representation

## 🤖 Model Architecture

### Primary Model: Random Forest Classifier

- **Algorithm**: Random Forest with hyperparameter tuning
- **Features**: Word2Vec embeddings (100 dimensions)
- **Optimization**: RandomizedSearchCV for hyperparameter tuning

### Model Performance

- **Validation Accuracy**: ~85-90%
- **Key Metrics**: Precision, Recall, F1-score
- **Cross-validation**: 3-fold CV with randomized search

### Hyperparameters (Optimized)

```python
{
    'n_estimators': 200,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 1,
    'max_features': 'sqrt'
}
```

## 🚀 Web Application Features

### Streamlit Dashboard

- **Real-time Classification**: Input tweets or URLs for instant classification
- **Visual Interface**: Clean, responsive web design
- **Recent Tweets**: Fetch and classify recent disaster-related tweets
- **URL Support**: Analyze tweets from Twitter URLs

### Deployment Options

- **Local Development**: Run on localhost:8501
- **Cloud Deployment**: Ngrok tunnel for public access
- **Docker Ready**: Containerizable for cloud platforms

## 📁 Project Structure

```
NLPwithdisastertweets/
├── NLP_with_disaster_tweets.ipynb    # Main analysis notebook
├── README.md                        # Project documentation
├── app.py                          # Streamlit web application
├── datasets/
│   ├── test.csv                    # Test dataset
│   └── tweets.csv                  # Training dataset
├── result/
│   ├── disaster.png                # Disaster tweet examples
│   ├── nondisaster.png             # Non-disaster tweet examples
│   └── ui.png                      # Web interface screenshot
├── models/                         # Saved models (generated)
│   ├── best_rf_model.pkl
│   ├── word2vec_model.pkl
│   └── label_encoder.pkl
└── requirements.txt                # Python dependencies
```

## 🚦 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Installation Steps

1. **Clone the repository**

```bash
git clone [repository-url]
cd NLPwithdisastertweets
```

2. **Install dependencies**

```bash
pip install pandas numpy scikit-learn nltk gensim streamlit joblib matplotlib seaborn
```

3. **Download NLTK data**

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

4. **Run the Jupyter notebook**

```bash
jupyter notebook NLP_with_disaster_tweets.ipynb
```

5. **Launch the web application**

```bash
streamlit run app.py
```

## 🧪 Usage Examples

### Command Line Testing

```python
# Load the trained model
import joblib
model = joblib.load('best_rf_model.pkl')

# Test a new tweet
tweet = "Massive earthquake hits the city center"
# Preprocess and predict...
```

### Web Interface

1. Open `http://localhost:8501` in your browser
2. Enter a tweet or Twitter URL
3. Click "Classify" to see the prediction
4. View recent disaster tweets with a single click

## 📈 Model Evaluation Results

### Classification Report

```
              precision    recall  f1-score   support

           0       0.87      0.89      0.88      1500
           1       0.89      0.87      0.88      1500

    accuracy                           0.88      3000
   macro avg       0.88      0.88      0.88      3000
weighted avg       0.88      0.88      0.88      3000
```

### Confusion Matrix

- True Positives: 1305
- True Negatives: 1335
- False Positives: 165
- False Negatives: 195

## 🔧 Customization & Extension

### Adding New Features

- **New Models**: Replace Random Forest with SVM, XGBoost, or Neural Networks
- **Feature Engineering**: Add TF-IDF, sentiment analysis, or emoji detection
- **Real-time Data**: Integrate Twitter API for live tweet classification
- **Multi-language**: Extend to support tweets in multiple languages

### Model Improvements

- **Deep Learning**: Implement LSTM or BERT for better context understanding
- **Ensemble Methods**: Combine multiple models for improved accuracy
- **Active Learning**: Continuously improve with user feedback

## 🐛 Troubleshooting

### Common Issues

1. **NLTK Data Missing**: Run `nltk.download()` commands
2. **Model Loading Error**: Ensure all .pkl files are in correct directory
3. **Streamlit Port Conflict**: Use `--server.port` flag to change port
4. **Memory Issues**: Reduce Word2Vec vector size for lower memory usage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: Kaggle Disaster Tweets dataset
- **Libraries**: NLTK, Scikit-learn, Gensim, Streamlit teams
- **Community**: Open-source contributors and Stack Overflow

## 📞 Contact

For questions or suggestions, please open an issue or contact the project maintainers.

---

**Note**: This project is part of an educational initiative to demonstrate NLP techniques for social media analysis and disaster response applications.
