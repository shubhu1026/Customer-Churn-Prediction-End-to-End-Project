from fastapi.testclient import TestClient
from main import app  # adjust if your FastAPI app is elsewhere

client = TestClient(app)

def test_predict_endpoint():
    sample_data = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 80.99,
        "TotalCharges": 389.99
    }

    response = client.post("/predict", json=sample_data)

    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
