from flask import Flask, request, render_template

import pandas as pd
import pickle


application = Flask(__name__)

app = application


with open('models/churn_pipeline.pkl', 'rb') as file:
    model = pickle.load(file)


@app.route("/")
def index():

    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():

    if request.method == 'POST':


        gender = request.form.get('gender')

        SeniorCitizen = int(
            request.form.get('SeniorCitizen')
        )

        Partner = request.form.get('Partner')

        Dependents = request.form.get('Dependents')

        tenure = int(
            request.form.get('tenure')
        )

        PhoneService = request.form.get('PhoneService')

        MultipleLines = request.form.get('MultipleLines')

        InternetService = request.form.get('InternetService')

        OnlineSecurity = request.form.get('OnlineSecurity')

        OnlineBackup = request.form.get('OnlineBackup')

        DeviceProtection = request.form.get('DeviceProtection')

        TechSupport = request.form.get('TechSupport')

        StreamingTV = request.form.get('StreamingTV')

        StreamingMovies = request.form.get('StreamingMovies')

        Contract = request.form.get('Contract')

        PaperlessBilling = request.form.get('PaperlessBilling')

        PaymentMethod = request.form.get('PaymentMethod')

        MonthlyCharges = float(
            request.form.get('MonthlyCharges')
        )

        TotalCharges = float(
            request.form.get('TotalCharges')
        )


        new_customer = pd.DataFrame(
            [[
                gender,
                SeniorCitizen,
                Partner,
                Dependents,
                tenure,
                PhoneService,
                MultipleLines,
                InternetService,
                OnlineSecurity,
                OnlineBackup,
                DeviceProtection,
                TechSupport,
                StreamingTV,
                StreamingMovies,
                Contract,
                PaperlessBilling,
                PaymentMethod,
                MonthlyCharges,
                TotalCharges
            ]],
            columns=[
                'gender',
                'SeniorCitizen',
                'Partner',
                'Dependents',
                'tenure',
                'PhoneService',
                'MultipleLines',
                'InternetService',
                'OnlineSecurity',
                'OnlineBackup',
                'DeviceProtection',
                'TechSupport',
                'StreamingTV',
                'StreamingMovies',
                'Contract',
                'PaperlessBilling',
                'PaymentMethod',
                'MonthlyCharges',
                'TotalCharges'
            ]
        ) 
        prediction = model.predict(
            new_customer
        )[0]

        probability = model.predict_proba(
            new_customer
        )[0][1]


        if prediction == 1:

            result = (
                f"Churn | "
                f"Probability: {probability * 100:.2f}%"
            )

        else:

            result = (
                f"No Churn | "
                f"Probability: {probability * 100:.2f}%"
            )


        return render_template(
            'predict.html',
            result=result
        )


    else:

        return render_template('predict.html')

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )