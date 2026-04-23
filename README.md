# 🏠 Housing Price Prediction - ML Project

Predict house prices using Multiple Linear Regression. 12 features → price forecast.

## 📊 Dataset
- **Source:** Housing sales data
- **Features:** 12 (sqft, bedrooms, bathrooms, grade, location, etc.)
- **Target:** AdjSalePrice

## 🛠️ Tech Stack
- Python 3.x
- Scikit-learn (Linear Regression)
- Pandas (data handling)
- Matplotlib + Seaborn (viz)
- NumPy (numerical ops)

## 🚀 Quick Start

```bash
# Clone repo
git clone <repo-url>
cd housing-price-prediction

# Install deps
pip install -r requirements.txt

# Run notebook
jupyter notebook HOUSING_PRICE_CLEANED.ipynb
```

## 📁 Project Structure
```
├── HOUSING_PRICE_CLEANED.ipynb    # Main analysis notebook
├── Housing_Price.csv              # Dataset
├── README.md                      # This file
└── requirements.txt               # Dependencies
```

## 🔍 Methodology

### 1. Data Exploration
- Load & inspect dataset
- Check missing values
- Statistical summary

### 2. EDA
- Target distribution analysis
- Correlation heatmap
- Feature relationships

### 3. Model Building
- Feature selection (12 predictors)
- Train-test split (80-20)
- Multiple Linear Regression

### 4. Evaluation
- RMSE calculation
- R² score
- Residual analysis
- Actual vs Predicted plot

### 5. Insights
- SqFtTotLiving = strongest predictor
- BldgGrade highly correlated
- TrafficNoise negatively impacts price

## 📈 Results
- **Model:** Multiple Linear Regression
- **R² Score:** ~0.XX (variance explained)
- **RMSE:** $XXX,XXX (avg prediction error)

## 🎯 Key Features Impact
1. **SqFtTotLiving** - Primary driver
2. **BldgGrade** - Quality metric
3. **Bathrooms** - Positive correlation
4. **TrafficNoise** - Negative impact

## 🔮 Future Improvements
- [ ] Feature engineering (property age, price/sqft)
- [ ] Handle outliers
- [ ] Try Ridge/Lasso regression
- [ ] Cross-validation
- [ ] Advanced models (Random Forest, XGBoost)

## 📝 Requirements
```txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
jupyter>=1.0.0
```

## 👤 Author
**Sumit Purnapatre**  
- GitHub: [@sumitp07](https://github.com/sumitp07)
- LinkedIn: [Sumit Purnapatre](https://linkedin.com/in/sumit-purnapatre)

## 📄 License
MIT License - free to use for learning/portfolio

---

⭐ Star if helpful | 🐛 Issues welcome | 🤝 PRs appreciated
