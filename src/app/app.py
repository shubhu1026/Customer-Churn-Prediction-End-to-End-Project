import gradio as gr

def dummy_predict(*args):
    return "🌊 Dummy prediction!"

with gr.Blocks(theme=gr.themes.Citrus(), title="Telco Churn Predictor") as demo:

    gr.Markdown("## 📊 Telco Customer Churn Prediction")
    gr.Markdown("Fill in the customer details below to get a prediction.")

    with gr.Row():
        with gr.Column():
            gender = gr.Dropdown(["Male", "Female"], label="Gender", value="Male")
            partner = gr.Dropdown(["Yes", "No"], label="Partner", value="No")
            dependents = gr.Dropdown(["Yes", "No"], label="Dependents", value="No")
            tenure = gr.Number(label="Tenure (months)", value=1)
            monthly_charges = gr.Number(label="Monthly Charges", value=50)
            total_charges = gr.Number(label="Total Charges", value=50)

        with gr.Column():
            phone_service = gr.Dropdown(["Yes", "No"], label="Phone Service", value="Yes")
            multiple_lines = gr.Dropdown(["Yes", "No", "No phone service"], label="Multiple Lines", value="No")
            internet_service = gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service", value="DSL")
            online_security = gr.Dropdown(["Yes", "No", "No internet service"], label="Online Security", value="No")
            online_backup = gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup", value="No")
            device_protection = gr.Dropdown(["Yes", "No", "No internet service"], label="Device Protection", value="No")
            tech_support = gr.Dropdown(["Yes", "No", "No internet service"], label="Tech Support", value="No")
            streaming_tv = gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming TV", value="No")
            streaming_movies = gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming Movies", value="No")

    with gr.Row():
        contract = gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract", value="Month-to-month")
        paperless_billing = gr.Dropdown(["Yes", "No"], label="Paperless Billing", value="Yes")
        payment_method = gr.Dropdown(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            label="Payment Method", value="Electronic check"
        )

    output_text = gr.Textbox(label="Prediction", interactive=False, placeholder="Your prediction will appear here 🌊")
    submit_btn = gr.Button("Predict", variant="primary")

    submit_btn.click(
        fn=dummy_predict,
        inputs=[
            gender, partner, dependents, phone_service, multiple_lines,
            internet_service, online_security, online_backup, device_protection,
            tech_support, streaming_tv, streaming_movies, contract,
            paperless_billing, payment_method, tenure, monthly_charges, total_charges
        ],
        outputs=output_text
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
