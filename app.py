from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load trained model and encoder
model = pickle.load(open('disease_model.pkl', 'rb'))
mlb = pickle.load(open('symptom_binarizer.pkl', 'rb'))

# Load datasets
df = pd.read_csv('dataset.csv')
desc_df = pd.read_csv('symptom_Description.csv')
prec_df = pd.read_csv('symptom_precaution.csv')

# Identify symptom columns
symptom_cols = [col for col in df.columns if 'Symptom' in col]

# Combine all symptoms into one column
df['Symptoms'] = df[symptom_cols].apply(
    lambda x: ','.join([s.strip().lower() for s in x.dropna().astype(str).tolist() if s.strip() != '']),
    axis=1
)

# Get all unique symptoms for dropdown
all_symptoms = sorted(set(
    [s for sublist in df['Symptoms'].apply(lambda x: [s.strip().lower() for s in x.split(',')]) for s in sublist]
))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    description = None
    precautions = []

    if request.method == 'POST':
        selected_symptoms = request.form.getlist('symptoms')
        if selected_symptoms:
            user_input = mlb.transform([selected_symptoms])
            prediction = model.predict(user_input)[0]

            # Get description
            desc_row = desc_df.loc[desc_df['Disease'].str.lower() == prediction.lower(), 'Description']
            if not desc_row.empty:
                description = desc_row.values[0]

            # Get precautions
            prec_row = prec_df.loc[prec_df['Disease'].str.lower() == prediction.lower()]
            if not prec_row.empty:
                precautions = [p for p in prec_row.iloc[:, 1:].values.flatten().tolist() if isinstance(p, str) and p.strip()]

    return render_template('index.html',
                           all_symptoms=all_symptoms,
                           prediction=prediction,
                           description=description,
                           precautions=precautions)

if __name__ == '__main__':
    app.run(debug=True)
