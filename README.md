
# House Prices — End-to-End ML Regression

> Predicting house sale prices using the Kaggle House Prices dataset.  

---

## The problem

Given 79 features describing residential homes in Ames, Iowa — lot size, quality ratings, neighbourhood, basement area, garage capacity, and more — predict the final sale price of each house.

**Type:** Supervised regression  
**Target:** `SalePrice` (continuous)  
**Dataset:** [Kaggle House Prices: Advanced Regression Techniques 
**Training samples:** 1,460 houses | **Test samples:** 1,459 houses | **Features:** 79

---

## My hypotheses before looking at the data

1. I expected bedroom count to be the most important feature
2. I expected most houses to sell between $150k and $300k
3. I expected neighbourhood to be a top-5 predictor
4. I expected newer houses to always sell for more

**What the data actually showed:** OverallQual (overall build quality) was the #1 correlator at r=0.79 — not square footage, not bedrooms. Quality of construction matters more than size.
---

## EDA findings

### 1. The target is right-skewed — log-transform required
```python
# Before: skewness = 1.88 (heavily right-skewed)
# After:  skewness = 0.12 (nearly normal)
y = np.log1p(train['SalePrice'])
```

### 2. The outliers that would have broken my model

In the `GrLivArea` vs `SalePrice` scatter plot, some houses stood out immediately:
- Above-ground living area: **4,676 sq ft** and **5,642 sq ft**
- Sale price: **$160,000** and **$184,750**
**Action:** Removed both rows before training.

### 3. What actually drives house prices

Top 5 features by Pearson correlation with SalePrice:

| Feature | Correlation | What it means |
|---|---|---|
| OverallQual | 0.791 | Overall material and finish quality |
| GrLivArea | 0.709 | Above-ground living area (sq ft) |
| GarageCars | 0.640 | Garage capacity (number of cars) |
| GarageArea | 0.623 | Garage size in sq ft |
| TotalBsmtSF | 0.614 | Total basement area |

**Insight:** Build quality beats raw size. A well-built smaller house sells for more than a poorly-built large one. Garage capacity appearing twice (cars + area) suggests buyers in Ames heavily value parking — likely because it's a suburban/rural area.

### 4. "Missing" data that isn't actually missing

19 columns had missing values. The temptation is to fill them all with the column mean. That would be wrong.

- `PoolQC` missing 99.5% → **means no pool exists**, not unknown pool quality. Fill: `"None"`
- `GarageType` missing 5.5% → **means no garage**. Fill: `"None"`, `GarageArea` → `0`
- `LotFrontage` missing 17.7% → **genuinely unknown**. Fill: median

Every fill decision is a hypothesis about reality. Filling "no pool" with average pool quality encodes a lie into training data.

---

## Feature engineering
### 3. What drives house prices  

| Feature | Meaning |
|---|---|
| OverallQual | Build quality |
| GrLivArea | Living area |
| GarageArea | Garage size |
| TotalBsmtSF | Basement area |
| YearBuilt | House age |

**Insight:** Quality impacts price more than size.

---

### 4. Missing data  

- Numerical → median  
- Categorical → mode / `"None"`  

**Note:** Missing values handled based on context.

---

## Feature selection  

Used key features (area, quality, year) for simplicity and performance.

---

## Encoding  

- Ordinal → quality features  
- One-hot → nominal features  

---

## Data consistency  

```python
df = df.reindex(columns=feature_columns, fill_value=0)

## Models compared

Three models, same data, same evaluation:

| Model | Performance | Notes |
|---|---|---|
| Linear Regression | Good baseline | Requires scaled features |
| Random Forest | Moderate performance | Handles non-linearity, no scaling needed |
| **XGBoost** | **Best overall** | Captures complex patterns, strong on tabular data |

## Conclusion

1. **EDA comes first.** Detecting and removing outliers prevented misleading patterns in the model.
2. **Missing ≠ random.** Filling values based on context (not blindly) improves data quality.
3. **Encoding matters.** Wrong handling of categorical features can silently break model performance.
4. **Simpler features can still work.** Carefully selected key features performed well without heavy feature engineering.
5. **Consistency is critical.** Train-time and prediction-time preprocessing must match exactly.

---

## How to run

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/house-prices-ml
cd house-prices-ml

# Install dependencies
pip install -r requirements.txt

# Run on Kaggle
# Open house_prices_day1.ipynb in a Kaggle notebook
# Dataset: House Prices - Advanced Regression Techniques (add via Kaggle UI)
```

**requirements.txt**
```
pandas>=1.5
numpy>=1.23
scikit-learn>=1.2
matplotlib>=3.6
seaborn>=0.12
```

---

## Files

```
house-prices-ml/
├── house_prices_day1.ipynb   # Full notebook — all 12 cells with explanations
├── submission.csv             # Kaggle submission file
├── requirements.txt
└── README.md
```

---
