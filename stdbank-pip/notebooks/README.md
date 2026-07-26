# Fraud Detection Dashboard

Streamlit interface for the PaySim fraud detection model.

## Running it

```
pip install -r requirements.txt
streamlit run app.py
```

The app loads everything it needs from the `artifacts` folder, so it runs without
the original dataset.

## Layout

```
fraud_dashboard/
    app.py                    the dashboard
    requirements.txt
    train_and_save.py         regenerates the artifacts from the raw CSV
    artifacts/
        fraud_model.joblib    trained XGBoost model
        model_metadata.json   feature order, thresholds, metrics, costs
        eda_summary.json      precomputed aggregates for the charts
```

## Pages

1. **Overview** headline findings and the honest caveat about simulated data
2. **Explore the data** distributions, fraud by type, timing, transaction size
3. **Model performance** model comparison, why accuracy misleads, threshold chosen on cost
4. **Score a transaction** live prediction with a per feature explanation of the score
5. **Recommendations** adjustable cost calculator, type priorities, actions for the bank

## Regenerating the artifacts

Point `DATA_PATH` in `train_and_save.py` at your copy of the CSV, then run it.
The notebook also writes the same artifacts in its final section.
