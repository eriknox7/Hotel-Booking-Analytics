# 🏨 Hotel Booking Analytics & Predictive Modeling

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![ScikitLearn](https://img.shields.io/badge/ScikitLearn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

> End-to-end analytics on 5,500 hotel bookings — association rules, classification, clustering, and ANN-based revenue prediction.

---

## 🚀 Live Demo
👉 **[Hotel Booking Analytics Dashboard](https://hotel-booking-analytics.streamlit.app/)**

---

## ⚙️ Setup

```bash
git clone https://github.com/eriknox7/Hotel-Booking-Analytics.git
cd Hotel-Booking-Analytics
pip install -r requirements.txt
streamlit run app.py
```

---

## 📋 What's Inside

| Task | Technique | Key Finding |
|------|-----------|-------------|
| Data Integration | Merge + Feature Engineering | 30,000 rows × 22 features |
| Association Rules | Apriori & FP-Growth | 60 rules, lift ~1.00–1.03 |
| Classification | Random Forest, SVM, LR, DT | Best: Random Forest (~36%) |
| Clustering | K-Means + Hierarchical | Budget segment: 70% cancellation rate |
| ANN | Feedforward Neural Network | Revenue R² = 0.9987 |
| Explainability | SHAP | `total_amount` most influential feature |

---

## 👨‍💻 Author

### Sadique Khan
