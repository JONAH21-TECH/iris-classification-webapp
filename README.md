# 🌸 Iris Classification & Evaluation Metrics Dashboard

A modern, interactive Machine Learning web application built using **Python, Flask, and Scikit-Learn**. 

This application demonstrates a complete classification workflow on the classic **Iris flower dataset** with a trained **Random Forest Classifier**, interactive inference sliders, dynamic probability charts, and a complete suite of model evaluation metrics.

---

## 🚀 Features

1. **Interactive Live Predictor**:
   - 4 real-time sliders for *Sepal Length*, *Sepal Width*, *Petal Length*, and *Petal Width*.
   - Sample preset buttons (*Typical Setosa*, *Typical Versicolor*, *Typical Virginica*) for quick testing.
   - Dynamic class probability bar chart powered by Chart.js.
2. **Comprehensive Evaluation Metrics**:
   - Accuracy (`90.0%`), Balanced Accuracy (`90.0%`).
   - Precision (Macro: `90.2%`), Recall (Macro: `90.0%`), F1-Score (Macro: `90.0%`).
   - Multiclass ROC-AUC (`0.9933`), Log Loss (`0.1204`).
   - Cohen's Kappa (`0.8500`), Matthews Correlation Coefficient (`0.8514`).
   - Zero-One Misclassification Error Rate (`10.0%`).
3. **Interactive Confusion Matrix Visualizer**:
   - Color-coded actual vs. predicted matrix with per-class accuracy breakdowns.
4. **Classification Report**:
   - Per-class precision, recall, f1-score, and sample support table.
5. **Dataset & Methodology Overview**:
   - Details on data preprocessing, feature normalization (`StandardScaler`), and the Random Forest ensemble model.

---

## 📁 Project Structure

```text
iris_webapp/
├── api/
│   ├── index.py              # WSGI Flask App entry point (works locally & on Vercel)
│   └── templates/
│       └── index.html        # Duplicate for Vercel serverless bundling
├── templates/
│   └── index.html            # Main UI (Tailwind CSS + Chart.js via CDN)
├── app.py                    # Local runner script
├── requirements.txt          # Python dependencies for local & Vercel
├── vercel.json               # Serverless routing config for Vercel
├── .gitignore                # Git ignore configuration
└── README.md                 # Project documentation
```

---

## 💻 Running Locally

To run the application on your local machine:

1. Open PowerShell and navigate to the project directory:
   ```powershell
   cd C:\Users\LENOVO\.gemini\antigravity\scratch\iris_webapp
   ```

2. Start the local server:
   ```powershell
   & "C:\Users\LENOVO\.gemini\antigravity\scratch\ml_env\Scripts\python.exe" app.py
   ```

3. Open your web browser and go to:
   **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 🌐 How to Host on Vercel

The project is pre-configured with `vercel.json` and `/api/index.py` for 100% seamless Vercel deployment.

### Method 1: Deploy via GitHub (Recommended & Easiest)
1. Initialize a git repository and commit your files:
   ```powershell
   cd C:\Users\LENOVO\.gemini\antigravity\scratch\iris_webapp
   git init
   git add .
   git commit -m "Initial commit of Iris Classification Webapp"
   ```
2. Push this folder to a new repository on your [GitHub account](https://github.com/new).
3. Log in to [Vercel](https://vercel.com).
4. Click **"Add New..."** → **"Project"**.
5. Select your GitHub repository.
6. Click **"Deploy"**! Vercel will automatically build the Python serverless function and give you a live production URL (e.g., `https://iris-classifier.vercel.app`) to send to your supervisor.

### Method 2: Deploy via Vercel CLI
If you have Node.js / npm installed:
```powershell
npx vercel
```
Follow the on-screen prompts to log in and deploy.
