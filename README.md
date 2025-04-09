# 📊 Leverage in the Cross-Section of Equity Returns

This project explores how financial leverage affects the explanatory power of asset pricing models. Inspired by Doshi et al. (2019), it investigates whether leverage distorts observed anomalies such as the value premium and size effect.

## 🎯 Objective
To analyze whether removing the effect of leverage from stock returns alters the significance of risk factors in the CAPM and Fama-French 3-Factor models.

## 📁 Data
- **CRSP and Compustat merged data** for firm-level fundamentals
- **Ken French factor datasets** for market, size, and value factors

## 🧠 Methods
- Construction of **levered and unlevered returns**
- Portfolio sorting (size × B/M grids)
- **Fama-MacBeth regressions** for factor significance
- Hypothesis testing using robust standard errors
- Visualizing how leverage affects coefficient stability and explanatory power

## 🔧 Tools & Libraries
- `pandas`, `numpy`, `statsmodels`, `scipy`, `matplotlib`

## 🧪 Key Findings
- Once leverage is removed:
  - The **value premium weakens**
  - The **size effect nearly disappears**
  - **Market beta** remains statistically insignificant
- Leverage plays a significant role in shaping asset pricing relationships
