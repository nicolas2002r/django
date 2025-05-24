from django.shortcuts import render
import numpy as np
import pandas as pd
import os
from django.conf import settings
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.utils import resample

def main(request):
    return render(request, 'index.html', context={})

def prediccion(request):
    # Cargar datos
    csv_path = os.path.join(settings.BASE_DIR, 'proyecto', 'data', 'data.csv')
    data = pd.read_csv(csv_path, sep=",", quotechar='"')
    #data = pd.read_csv(r"C:\Users\nicol\Documents\Corhuila\IA\django_ml\proyecto\data\data.csv", sep=",", quotechar='"')
    data.columns = data.columns.str.strip().str.lower()
    data.drop(columns=['id', 'unnamed: 32'], inplace=True, errors='ignore')
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

    # Balanceo de clases
    df_majority = data[data['diagnosis'] == 0]
    df_minority = data[data['diagnosis'] == 1]

    df_minority_upsampled = resample(df_minority,
                                     replace=True,
                                     n_samples=len(df_majority),
                                     random_state=42)

    balanced_data = pd.concat([df_majority, df_minority_upsampled])

    # Separar características y etiqueta
    X = balanced_data.drop(columns=['diagnosis'])
    y = balanced_data['diagnosis']

    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modelo árbol de decisión
    model = DecisionTreeClassifier(max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Métricas
    precision = round(precision_score(y_test, y_pred), 2)
    recall = round(recall_score(y_test, y_pred), 2)
    f1 = round(f1_score(y_test, y_pred), 2)

    sum_real = sum(y_test.tolist())
    sum_pred = sum(y_pred.tolist())

    context = {
        'test': y_test.tolist(),
        'prediction': y_pred.tolist(),
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'metrics': [precision, recall, f1],  # Para la segunda gráfica
        'sum_real': sum_real,
        'sum_pred': sum_pred
    }

    return render(request, 'index.html', context=context)
