# Predicting Customer Churn with a Responsible-AI Lens

A telecom churn model built for a hackathon, with the main question being what it costs to miss a churner and whether the model treats customer groups fairly. The model is tuned around a dollar cost function and not the default 0.5 cutoff, then checked for fairness and explained with SHAP and LIME.

**Tools:** Python (pandas, scikit-learn, imbalanced-learn, XGBoost, LightGBM, SHAP, LIME, Fairlearn)
**Type:** Solo hackathon project

---

## Questions

1. Which customers are likely to churn, and what drives it?
2. Missing a churner costs far more than offering a retention incentive to someone who would have stayed. How should the decision threshold reflect that?
3. Does the model treat groups differently (gender, senior citizens)?

## Method

- **Data:** the public IBM Telco Customer Churn sample (7,043 customers, 21 columns). Cleaned, explored and checked for anomalies, then split into train, validation and test sets.
- **Class imbalance:** handled with SMOTE.
- **Models:** compared 8 (logistic regression, decision tree, random forest, gradient boosting, XGBoost, LightGBM, KNN, SVM), ranked by recall because a missed churner is the costly error.
- **Tuning:** grid search with 5-fold cross-validation on the logistic regression model, which had the best recall among the 8 models.
- **Threshold:** tested thresholds from 0.10 to 0.90 against a cost function (a missed churner costs $500, an unnecessary retention offer costs $50). These costs are assumptions for illustration.
- **Explainability:** SHAP, partial dependence, feature importance and a surrogate model for global explanations, and LIME and Anchors for individual predictions.
- **Fairness:** selection rate and recall by gender and by senior-citizen status, using Fairlearn.

## Findings

**Dropping the threshold from 0.50 to 0.10 catches nearly all churners, at a price.** On the held-out test set (1,409 customers):

| Metric | Default threshold (0.50) | Optimized (0.10) |
|---|---|---|
| Recall | 77.5% | 97.9% |
| Precision | 51.1% | 35.9% |
| Accuracy | 74.4% | 53.0% |
| AUC-ROC | 0.844 | 0.844 |

At 0.10 the model misses 8 churners and flags 654 loyal customers. Under the assumed costs, that cuts the modeled cost from $55,850 to $36,700, a saving of $19,150. On the validation set the saving was $20,250 (35%).

**What drives churn**

- **Contract type:** month-to-month customers churn at about 42%, two-year customers at about 3%.
- **Tenure:** customers in their first year are at the highest risk, and customers with 4+ years rarely churn.
- **Internet service:** fiber optic customers churn more than DSL customers.
- **Add-on services:** customers without online security, tech support or online backup churn more.
- **Payment method:** electronic check users churn more.
- **Monthly charges:** higher charges go with higher churn.

**Fairness**

- **Gender:** similar for men and women (selection rates 71.2% and 73.7%, recall within 1 point).
- **Senior citizens:** flagged much more often (86.9% vs 69.7% selection rate). This mirrors higher actual churn among seniors, but it could mean seniors get more retention outreach, so the notebook recommends monitoring targeting and running regular fairness audits.

## Files

| File | What it is |
|---|---|
| `telecom_churn_responsible_ai.ipynb` | Full notebook with code, outputs, explanations and fairness checks |
| `requirements.txt` | Python dependencies |

## Data

The dataset is not included. Download the IBM Telco Customer Churn file (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) from the public source and place it next to the notebook. The notebook's first data cell reads a file with a specific name, so update the file path there to match your copy.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook telecom_churn_responsible_ai.ipynb
```

## Limitations

- The $500 and $50 costs are assumptions, not company figures. A different cost ratio would give a different threshold and a different saving.
- A 0.10 threshold buys recall at a large cost in precision and accuracy: more than half of loyal customers get flagged. Whether that is worth it depends on what a retention offer really costs.
- Only logistic regression was tuned, and tuning did not improve on the base model.
- The data is a public sample dataset, so the findings show the method and are not conclusions about a real telecom company.

## Author

Prathiba Reddy, MS in Business Analytics & AI, Johns Hopkins Carey Business School
[LinkedIn](https://www.linkedin.com/in/prathiba-reddy-namireddy) | [Portfolio](https://6a38b7b1cee129ca7a613700--comforting-queijadas-d1f251.netlify.app)
