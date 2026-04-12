
# House Prices — End-to-End ML Regression
**Day 1 of 7 | ML Learning Roadmap**

> Predicting residential house sale prices using the Kaggle House Prices dataset.  
> Built as part of a structured 7-day ML curriculum — focus on understanding every step deeply, not just running code.

---

## The problem

Given 79 features describing residential homes in Ames, Iowa — lot size, quality ratings, neighbourhood, basement area, garage capacity, and more — predict the final sale price of each house.

**Type:** Supervised regression  
**Target:** `SalePrice` (continuous, in USD)  
**Dataset:** [Kaggle House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)  
**Training samples:** 1,460 houses | **Test samples:** 1,459 houses | **Features:** 79

---

## My hypotheses before looking at the data

Before writing a single line of model code, I wrote down what I expected to find:

1. I expected bedroom count to be the most important feature
2. I expected most houses to sell between $150k and $300k
3. I expected neighbourhood to be a top-5 predictor
4. I expected newer houses to always sell for more

**What the data actually showed:** OverallQual (overall build quality) was the #1 correlator at r=0.79 — not square footage, not bedrooms. Quality of construction matters more than size. My neighbourhood hypothesis was partially right — but OverallQual dominated it.

---

## EDA findings

### 1. The target is right-skewed — log-transform required

Raw `SalePrice` ranges from $34,900 to $755,000 with a skewness of **1.88**. A handful of expensive properties pull the mean ($180k) well above the median ($163k). Linear regression assumes a normally distributed target, so I applied `log1p()` transformation — reducing skewness to **0.12** and making errors proportional to price rather than absolute.

```python
# Before: skewness = 1.88 (heavily right-skewed)
# After:  skewness = 0.12 (nearly normal)
y = np.log1p(train['SalePrice'])
```

### 2. The two outliers that would have broken my model

In the `GrLivArea` vs `SalePrice` scatter plot, two houses stood out immediately:
- Above-ground living area: **4,676 sq ft** and **5,642 sq ft**
- Sale price: **$160,000** and **$184,750**

Enormous houses selling for less than average. These are almost certainly partial sales or unusual transactions — not representative of the market. Including them would teach the model that huge houses are cheap, which is factually wrong for 99.9% of real sales.

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

I created 5 new features from domain reasoning — things a house buyer actually thinks about:

| Feature | Formula | Reasoning |
|---|---|---|
| `TotalSF` | BsmtSF + 1stFlrSF + 2ndFlrSF | Total livable area — what buyers actually care about |
| `TotalBath` | FullBath + 0.5×HalfBath + BsmtBath | One combined bathroom signal |
| `HouseAge` | YrSold − YearBuilt | Age at time of sale |
| `RemodAge` | YrSold − YearRemodAdd | How recently renovated |
| `WasRemodeled` | YearBuilt ≠ YearRemodAdd | Binary: ever renovated? |

**Result:** `TotalSF` ranked **#1 in feature importance** — above all 79 original features. Combining 3 area columns into what buyers reason about outperformed any individual area column. This is what feature engineering is for.

---

## Encoding strategy

Used two different strategies depending on feature type:

**Label encoding** for quality ratings (10 columns) — because Poor < Fair < Typical < Good < Excellent is a real ordering. Encoding as 0–5 preserves that meaning.

**One-hot encoding** for everything else — because `Neighbourhood=CollgCr` is not numerically better or worse than `Neighbourhood=Somerst`. They're just different.

Using label encoding on neighbourhood would imply a false ordering. Using one-hot on quality ratings would throw away the ordering signal. Wrong encoding = silent accuracy loss.

---

## Data leakage prevention

Split train/val **before** fitting the `StandardScaler`. The scaler was fit on training data only, then applied to val and test:

```python
scaler = StandardScaler()
X_tr_s  = scaler.fit_transform(X_tr)   # learns mean/std from train only
X_val_s = scaler.transform(X_val)      # applies train stats to val
X_test_s = scaler.transform(X_test)    # same for test
```

Fitting the scaler on all data would let test-set statistics influence training — making val scores optimistically wrong.

---

## Models compared

Three models, same data, same evaluation:

| Model | Val RMSE | CV RMSE (5-fold) | Notes |
|---|---|---|---|
| Linear Regression | 0.1306 | 0.1307 | Needs scaled features |
| **Ridge (α=10)** | **0.1245** | **0.1278** | Best overall |
| Random Forest | 0.1459 | 0.1381 | No scaling needed |

**RMSE is on log-prices** — so 0.13 means roughly 13% average error on sale price.

### Why Ridge beat Random Forest — and what that means

Random Forest is usually the go-to for tabular data. Here it lost to a regularized linear model. Why?

- The dataset has **~220 features after encoding** but only **1,166 training samples** — a relatively high-dimensional space where linear models with regularization often outperform trees
- Ridge penalises large coefficients, preventing it from overfitting to correlated features (e.g. GarageArea and GarageCars are highly correlated)
- Random Forest's CV RMSE (0.1381) is notably worse than its Val RMSE (0.1459) — suggesting variance in its predictions depending on which data it sees

**Key takeaway:** More complex model ≠ better model. Always compare against a well-tuned linear baseline before reaching for an ensemble.

### Cross-validation vs single split

For Linear Regression, Val RMSE (0.1306) ≈ CV RMSE (0.1307) — almost identical. This means the single 80/20 split was representative and the model is stable.

For Random Forest, there's a larger gap — the model's performance varies more depending on which subset it trains on. CV RMSE is the more trustworthy number.

---

## What I would do next

- Try **XGBoost / LightGBM** — gradient boosted trees that often outperform both linear models and random forests on structured data
- **Tune Ridge's alpha** — 10 was chosen heuristically; cross-validated grid search over [0.1, 1, 10, 100] would likely improve it
- **Stack the models** — blending Ridge and Random Forest predictions often beats either alone
- **More feature engineering** — total porch area, has a fireplace (binary), price-per-sqft of neighbourhood median

---

## What I learned

1. **EDA earns you the right to model.** Those 2 outlier houses would have corrupted my model's understanding of large houses. Visualising before modelling caught it.

2. **"Missing" is information.** 99.5% of PoolQC being missing tells you most houses have no pool. Filling with average pool quality is a lie. Every fill decision needs a reason.

3. **Feature engineering beats hyperparameter tuning.** TotalSF — 3 lines of code — became the single most important feature. More creative thinking about the data beats more complex models.

4. **Ridge beating Random Forest is a lesson about dimensionality.** ~220 features, 1166 samples. In this regime, regularised linear models are strong competitors. Complexity has to earn its place.

5. **Fit scalers on train only.** This is the most common subtle data leakage mistake. The fix is one word: `transform` instead of `fit_transform` on val/test.

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