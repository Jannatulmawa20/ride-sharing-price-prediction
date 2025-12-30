# Ride Sharing Price Prediction (VS Code Project)

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
