# Ride Sharing Price Prediction (VS Code Project)

Predict ride prices from trip and contextual features using classic regression models (Linear Regression, Decision Tree, Random Forest, and an ensemble). Includes a Streamlit app for interactive predictions.

## Project structure
- `train_model.py` — trains models and saves the best pipeline
- `app.py` — Streamlit UI that loads the saved model and predicts price
- `ride_sharing_dataset.csv` — dataset used for training/evaluation
- `requirements.txt` — Python dependencies

## 1) Setup (Windows PowerShell)
```powershell
cd path\to\this\folder
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2) Train and save the model
```powershell
python train_model.py
```
This creates `ride_price_model.joblib`.

## 3) Run Streamlit
```powershell
streamlit run app.py
```

## 4) (Optional) Use a notebook in VS Code
- Install VS Code extensions: **Python** and **Jupyter**
- Open `Assignment.md` for the questions and write your answers in a notebook.
- Select interpreter: the `.venv` you created.
\

## App Preview
![App Preview](images/app.png)
