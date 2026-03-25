from fastapi import FastAPI, Body
import pickle
import uvicorn

input_file = 'pipeline_v1.bin'

with open(input_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = FastAPI()

@app.post('/predict')
def predict(customer: dict=Body()):

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0,1]

    result = {
        'subscribe_probability': float(y_pred)
    }

    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)