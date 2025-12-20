from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, X_test, y_test):
    """
    Evaluates a XGBoost Model on test_data.

    Args:
        model: Trained Model.
        X_test: Test features.
        y_test: Test labels.
    """
    proba = model.predict_proba(X_test)[:, 1]   
    threshold = 0.3
    preds = (proba >= threshold).astype(int)
    print("Classification Report:\n", classification_report(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))