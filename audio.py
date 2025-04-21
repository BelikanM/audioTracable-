import numpy as np
import librosa
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import torch
import torch.nn as nn
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import tempfile
import traceback

# Initialisation de Flask
app = Flask(__name__)
CORS(app)  # Permet CORS pour accepter les requêtes du frontend

# Définir des chemins pour les modèles et le scaler
MODEL_PATH = 'source/mlp_model.pkl'
SCALER_PATH = 'source/scaler.pkl'

# Assurez-vous que les répertoires nécessaires existent
os.makedirs('source', exist_ok=True)

def load_model_and_scaler():
    """
    Charger le modèle et le scaler. Si le modèle ou le scaler n'existent pas,
    créer un modèle de base MLP (Multi-Layer Perceptron).
    """
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Modèle chargé à partir de :", MODEL_PATH)
    else:
        # Modèle de base avec scikit-learn
        model = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', max_iter=1000)
        
        # Initialiser le modèle avec des données factices
        dummy_features = np.random.rand(10, 40)  # 40 correspond à n_mfcc
        dummy_labels = np.random.randint(0, 5, size=10)  # 5 classes (0-4)
        model.fit(dummy_features, dummy_labels)
        
        print("Modèle créé et initialisé avec des données factices.")

    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        print("Scaler chargé à partir de :", SCALER_PATH)
    else:
        scaler = StandardScaler()
        # Initialiser le scaler avec quelques données factices pour éviter l'erreur
        dummy_data = np.random.rand(10, 40)  # 40 correspond à n_mfcc dans extract_features
        scaler.fit(dummy_data)
        print("Scaler créé et initialisé avec des données factices.")
    
    return model, scaler

def save_model_and_scaler(model, scaler):
    """
    Sauvegarder le modèle et le scaler pour réutilisation future.
    """
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("Modèle et scaler sauvegardés.")

def analyze_audio(file_path, model, scaler):
    """
    Analyser un fichier audio pour détecter les sources de bruit.
    Retourne les prédictions sous forme de liste de détections.
    """
    try:
        # Charger le fichier audio
        y, sr = librosa.load(file_path, sr=None)
        print(f"Fichier audio chargé, durée: {len(y)/sr:.2f}s, sample rate: {sr}Hz")

        # Extraire des caractéristiques audio
        features = extract_features(y, sr)
        print(f"Caractéristiques extraites, forme: {features.shape}")

        # Normaliser les caractéristiques
        normalized_features = scaler.transform(features.reshape(1, -1))
        print("Caractéristiques normalisées avec succès")

        # Vérifier si le modèle est entraîné avant de prédire
        if hasattr(model, 'classes_'):
            # Le modèle est entraîné, on peut faire la prédiction
            predicted_class = model.predict(normalized_features)[0]
            confidence = max(model.predict_proba(normalized_features)[0])
            print(f"Prédiction: classe {predicted_class}, confiance {confidence:.2f}")
        else:
            # Le modèle n'est pas entraîné, on retourne une valeur par défaut
            predicted_class = 4  # 'Autre' dans votre liste labels
            confidence = 0.5
            print("Modèle non entraîné, utilisation de valeurs par défaut")

        # Simuler plusieurs détections dans différentes directions
        detections = simulate_multiple_detections(predicted_class, confidence)
        print(f"Détections simulées: {len(detections)}")

        return detections
    except Exception as e:
        print(f"Erreur dans analyze_audio: {str(e)}")
        print(traceback.format_exc())
        raise

def extract_features(y, sr):
    """
    Extraire des caractéristiques MFCC d'un fichier audio.
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

def simulate_multiple_detections(predicted_class, confidence):
    """
    Simuler plusieurs détections pour imiter une analyse spatiale du son.
    """
    labels = ['Personne', 'Objet', 'Animal', 'Vent', 'Autre']
    source_type = labels[predicted_class]
    num_detections = np.random.randint(1, 5)
    detections = []

    for _ in range(num_detections):
        distance = np.random.uniform(10, 100)
        angle = np.random.uniform(0, 360)

        detections.append({
            "source_type": source_type,
            "confidence": float(confidence),
            "distance": float(distance),
            "angle": float(angle)
        })
    return detections

def incrementally_train(new_data, new_labels, model, scaler):
    """
    Entraîner le modèle de manière incrémentielle avec de nouvelles données.
    """
    try:
        # Normaliser les nouvelles données
        scaler.partial_fit(new_data)
        new_data_normalized = scaler.transform(new_data)

        # Accepter les nouvelles données dans le modèle
        if hasattr(model, 'classes_'):
            model.partial_fit(new_data_normalized, new_labels)
        else:
            unique_classes = np.unique(new_labels)
            model.partial_fit(new_data_normalized, new_labels, classes=unique_classes)
        
        print("Modèle entraîné avec de nouvelles données.")

        # Sauvegarder le modèle et le scaler mis à jour
        save_model_and_scaler(model, scaler)
    except Exception as e:
        print(f"Erreur lors de l'entraînement: {str(e)}")
        print(traceback.format_exc())
        raise

# Routes Flask
@app.route('/api/analyze', methods=['POST'])
def analyze_audio_file():
    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier audio trouvé"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Aucun fichier sélectionné"}), 400

    # Sauvegarder le fichier temporairement
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    try:
        file.save(temp_file.name)
        print(f"Fichier temporaire sauvegardé: {temp_file.name}")
        
        model, scaler = load_model_and_scaler()
        print("Modèle et scaler chargés avec succès")
        
        detected_noises = analyze_audio(temp_file.name, model, scaler)
        print(f"Analyse terminée, détections: {len(detected_noises)}")
        
        return jsonify({
            "status": "success",
            "detected_noises": detected_noises
        })
    except Exception as e:
        print(f"ERREUR dans /api/analyze: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    finally:
        temp_file.close()
        os.unlink(temp_file.name)

@app.route('/api/train', methods=['POST'])
def train_model():
    try:
        data = request.get_json()
        if 'features' not in data or 'labels' not in data:
            return jsonify({"error": "Les données d'entraînement sont manquantes"}), 400

        features = np.array(data['features'])
        labels = np.array(data['labels'])
        
        print(f"Données d'entraînement reçues: {features.shape[0]} échantillons")

        model, scaler = load_model_and_scaler()
        incrementally_train(features, labels, model, scaler)

        return jsonify({"status": "success", "message": "Modèle mis à jour avec succès"})
    except Exception as e:
        print(f"ERREUR dans /api/train: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4500, debug=True)


