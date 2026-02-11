from fastapi import FastAPI
import pickle

input_file = 'pipeline_v1.bin'

with open(input_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = FastAPI('subscription')

@app.route('/predict', methods=['POST'])
def predict(customer):

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0,1]

    result = {
        'subscribe_probability': float(y_pred)
    }

    return result

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)