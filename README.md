 Iris Classification & Evaluation Metrics Dashboard

A modern, interactive Machine Learning web application built using Python, Flask, and Scikit-Learn. 

This application demonstrates a complete classification workflow on the classic Iris flower dataset with a trained Random Forest Classifier, interactive inference sliders, dynamic probability charts, and a complete suite of model evaluation metrics.

---

 Features

1. Interactive Live Predictor:
   - 4 real-time sliders for *Sepal Length*, *Sepal Width*, *Petal Length*, and *Petal Width*.
   - Sample preset buttons (*Typical Setosa*, *Typical Versicolor*, *Typical Virginica*) for quick testing.
   - Dynamic class probability bar chart powered by Chart.js.
2. Comprehensive Evaluation Metrics:
   - Accuracy (`90.0%`), Balanced Accuracy (`90.0%`).
   - Precision (Macro: `90.2%`), Recall (Macro: `90.0%`), F1-Score (Macro: `90.0%`).
   - Multiclass ROC-AUC (`0.9933`), Log Loss (`0.1204`).
   - Cohen's Kappa (`0.8500`), Matthews Correlation Coefficient (`0.8514`).
   - Zero-One Misclassification Error Rate (`10.0%`).
3. Interactive Confusion Matrix Visualizer:
   - Color-coded actual vs. predicted matrix with per-class accuracy breakdowns.
4. Classification Report:
   - Per-class precision, recall, f1-score, and sample support table.
5. Dataset & Methodology Overview:
   - Details on data preprocessing, feature normalization (`StandardScaler`), and the Random Forest ensemble model.

---

 📁 Project Structure

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

