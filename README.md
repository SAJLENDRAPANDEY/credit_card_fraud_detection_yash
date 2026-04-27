# 💳 Credit Card Fraud Detection Suite

An advanced machine learning application for detecting fraudulent credit card transactions using XGBoost and Streamlit.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 Overview

This project provides a complete machine learning pipeline for credit card fraud detection with:

- **Advanced ML Model**: XGBoost classifier with 100 estimators
- **Data Balancing**: SMOTE oversampling for imbalanced dataset
- **Feature Scaling**: StandardScaler normalization
- **Interactive Dashboard**: Streamlit web application
- **Real-time Predictions**: Batch processing of transactions
- **Comprehensive Analytics**: Visualizations and risk assessment

### Key Metrics
- **Accuracy**: ~99%+
- **Precision**: High false positive minimization
- **Recall**: Optimized fraud detection
- **ROC-AUC**: Excellent discrimination ability

---

## 📦 Project Structure

```
fraud-detection/
│
├── app.py                    # Streamlit application
├── train_model.py            # Model training script
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
│
├── xgb_model.pkl            # Trained XGBoost model
├── xgb_scaler.pkl           # Feature scaler
├── feature_columns.pkl      # Feature column names
│
├── model_evaluation.png     # Training visualizations
├── dataset.csv              # Training data (optional)
└── sample_data.csv          # Sample prediction data
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 2GB RAM minimum

### Installation

1. **Clone or Download the Project**
```bash
git clone https://github.com/SAJLENDRAPANDEY/fraud-detection.git
cd fraud-detection
```

2. **Create Virtual Environment (Recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

---

## 📊 Features

### 1. **🔍 Detection Tab**
- Upload CSV file with transaction data
- Real-time fraud detection
- Adjustable probability threshold
- Instant results and statistics
- High-risk transaction identification

### 2. **📈 Analytics Tab**
- Feature importance analysis
- Model performance insights
- Data preprocessing information
- Best practices and tips

### 3. **ℹ️ Model Info Tab**
- Algorithm specifications
- Hyperparameters
- Feature descriptions
- Performance metrics
- Data handling techniques

### 4. **👤 About Tab**
- Developer information
- Contact details
- Project links (GitHub, LinkedIn)
- Technology stack
- How it works explanation

---

## 🤖 Model Details

### Algorithm: XGBoost Classifier

```
Configuration:
├── Estimators: 100
├── Max Depth: 6
├── Learning Rate: 0.1
├── Random State: 42
└── Eval Metric: Logloss
```

### Data Processing Pipeline

```
Raw Data
    ↓
Train-Test Split (80-20)
    ↓
SMOTE Balancing (Training Data)
    ↓
StandardScaler Normalization
    ↓
Model Training
    ↓
Probability Predictions
    ↓
Threshold Classification (0.7)
```

### Input Features

- **Time**: Transaction timestamp
- **Amount**: Transaction amount
- **V1-V28**: PCA-transformed features (anonymized)

Total: 30 features

---

## 📥 Data Format

### Input CSV Structure

```csv
Time,V1,V2,V3,...,V28,Amount
0,0.123,-0.456,0.789,...,0.234,149.62
1,-0.234,0.567,-0.123,...,-0.456,2.69
...
```

### Output CSV Structure

```csv
Time,V1,V2,...,Amount,Fraud_Probability,Is_Fraud
0,0.123,-0.456,...,149.62,0.9823,1
1,-0.234,0.567,...,2.69,0.0234,0
...
```

**Column Descriptions:**
- `Fraud_Probability`: ML model confidence (0-1)
- `Is_Fraud`: Predicted class (0=Legitimate, 1=Fraud)

---

## 🎓 Training Model from Scratch

### Step 1: Prepare Your Data
Place your training data as `dataset.csv` in the project directory.

Required columns: All 30 features + 'Class' column

### Step 2: Run Training Script
```bash
python train_model.py
```

### Expected Output
```
CREDIT CARD FRAUD DETECTION - MODEL TRAINING
================================================================================

📥 Loading dataset...
✅ Dataset loaded: 284807 rows, 31 columns

📊 Data Overview:
   Missing values: 0
   ...
   Fraud rate: 0.17%

🔧 Preparing features and target...
...

✅ Training Completed Successfully!

Model Performance:
- Accuracy:  0.9993
- Precision: 0.9854
- Recall:    0.8191
- F1-Score:  0.8969
- ROC-AUC:   0.9931
```

### Generated Files
- `xgb_model.pkl` - Trained model
- `xgb_scaler.pkl` - Feature scaler
- `feature_columns.pkl` - Feature list
- `model_evaluation.png` - Visualizations

---

## 🎨 User Interface

### Color Scheme
- **Primary**: #0066FF (Blue)
- **Secondary**: #FF3366 (Red/Pink)
- **Success**: #00D084 (Green)
- **Warning**: #FFA500 (Orange)
- **Dark**: #0F1419 (Background)

### Responsive Design
- Desktop optimized
- Mobile friendly
- Dark theme for eye comfort
- Smooth animations

---

## 📊 Predictions & Analysis

### Interpretation Guide

| Fraud Probability | Risk Level | Action |
|---|---|---|
| 0.0 - 0.3 | ✅ Low | Approve |
| 0.3 - 0.6 | ⚠️ Medium | Review |
| 0.6 - 0.8 | 🔴 High | Block & Verify |
| 0.8 - 1.0 | 🚨 Critical | Block Immediately |

### Key Metrics Explained

- **Accuracy**: Overall correctness of predictions
- **Precision**: True positives / All positive predictions
- **Recall**: True positives / All actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve (0.5-1.0)

---

## 🛠️ Customization

### Change Fraud Threshold

In `app.py`, modify the THRESHOLD variable:
```python
THRESHOLD = 0.7  # Change to 0.6, 0.8, etc.
```

### Adjust Model Parameters

Edit `train_model.py`:
```python
model = XGBClassifier(
    n_estimators=150,      # Increase trees
    max_depth=8,           # Increase depth
    learning_rate=0.05,    # Lower learning rate
    random_state=42
)
```

### Custom Styling

Modify the CSS in `app.py` under the `<style>` section to change:
- Colors
- Fonts
- Spacing
- Animations

---

## 🚀 Deployment Options

### Option 1: Streamlit Cloud (Easiest)

1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Auto-deploys on push

```bash
git push origin main
```

### Option 2: Docker Containerization

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

Build and run:
```bash
docker build -t fraud-detection .
docker run -p 8501:8501 fraud-detection
```

### Option 3: Hugging Face Spaces

1. Create new Space
2. Upload files
3. Configure `app.py` as main file
4. Deploy automatically

---

## 📈 Performance Monitoring

### Key Metrics to Track

```
Daily Metrics:
├── Total Transactions Processed
├── Fraud Detection Rate
├── False Positive Rate
├── Model Confidence Score
└── System Uptime
```

### Model Retraining

Retrain monthly or when:
- Accuracy drops below 99%
- Data distribution changes significantly
- New fraud patterns emerge
- Quarterly performance review

---

## 🔐 Security Considerations

- ✅ Data anonymized (PCA transformed)
- ✅ No sensitive information stored
- ✅ Secure model serialization
- ✅ Input validation
- ✅ Error handling

### Best Practices
1. Keep model files secure
2. Validate input data
3. Monitor predictions
4. Regular backups
5. Access control

---

## 📧 Contact & Support

**Developer**: Yash Sajlendra Pandey

- 📧 Email: [sajlendrapandey2022@gmail.com](mailto:sajlendrapandey2022@gmail.com)
- 🔗 LinkedIn: [Sajlendra Pandey](https://www.linkedin.com/in/sajlendra-pandey-37378627b/)
- 🐙 GitHub: [@SAJLENDRAPANDEY](https://github.com/SAJLENDRAPANDEY)

### Report Issues
1. Check existing issues on GitHub
2. Provide detailed error messages
3. Include sample data if possible
4. Mention Python and package versions

---

## 📚 Learning Resources

### XGBoost
- [Official Documentation](https://xgboost.readthedocs.io/)
- [Parameters Guide](https://xgboost.readthedocs.io/en/latest/parameter.html)

### SMOTE
- [Imbalanced Learn Docs](https://imbalanced-learn.org/)
- [SMOTE Explanation](https://imbalanced-learn.org/stable/generated/imblearn.over_sampling.SMOTE.html)

### Streamlit
- [Streamlit Docs](https://docs.streamlit.io/)
- [Component Gallery](https://streamlit.io/components)

### Machine Learning
- [Scikit-learn Guide](https://scikit-learn.org/)
- [ML Best Practices](https://developers.google.com/machine-learning/guides)

---

## 🎯 Future Enhancements

- [ ] Multi-model ensemble (XGBoost + LightGBM + CatBoost)
- [ ] Real-time data streaming
- [ ] Database integration
- [ ] REST API endpoint
- [ ] Mobile application
- [ ] Advanced SHAP explanations
- [ ] A/B testing framework
- [ ] Performance dashboard
- [ ] Automated retraining pipeline
- [ ] Fraud pattern analysis

---

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- XGBoost Team for excellent gradient boosting framework
- Streamlit for intuitive web framework
- Imbalanced-Learn for SMOTE implementation

---

## 📊 Version History

### v1.0 (Current)
- ✅ XGBoost model training
- ✅ SMOTE balancing
- ✅ Streamlit dashboard
- ✅ Real-time predictions
- ✅ Comprehensive analytics
- ✅ Export functionality

---

## 🎓 Educational Value

This project demonstrates:
- Machine learning pipeline design
- Handling imbalanced datasets
- Feature scaling and normalization
- Model evaluation and metrics
- Interactive web application development
- Data visualization techniques
- Production-ready code practices

---

## 💡 Tips for Best Results

1. **Data Quality**
   - Ensure all 30 features present
   - No missing values
   - Proper data types

2. **Threshold Tuning**
   - Start with 0.7
   - Adjust based on business needs
   - Monitor false positives

3. **Regular Monitoring**
   - Track model performance
   - Update predictions log
   - Analyze false positives

4. **Continuous Improvement**
   - Collect more data
   - Retrain periodically
   - Tune hyperparameters

---

**Made with ❤️ by Yash Sajlendra Pandey | 2024**

```
           ______                 _
          / ____/__________ _   __(_)______  __
         / /_  / ___/ __ `/ | / / / ___/ / / /
        / __/ / /  / /_/ /| |/ / / /  / /_/ /
       / /___/ /___/ _, _/ |___/ /_/   \__,_/
       \_____/\____/_/ |_|        Detection Suite
```

---

*Last Updated: 2024 | Star ⭐ this project if you found it helpful!*
