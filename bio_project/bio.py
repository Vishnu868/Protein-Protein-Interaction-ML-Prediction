import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from Bio.SeqUtils.ProtParam import ProteinAnalysis

def extract_protein_features(sequence):
    analysed_seq = ProteinAnalysis(sequence)
    aa_composition = analysed_seq.get_amino_acids_percent()  
    molecular_weight = analysed_seq.molecular_weight()      
    aromaticity = analysed_seq.aromaticity()                 
    instability_index = analysed_seq.instability_index()     
    isolectric_point = analysed_seq.isoelectric_point()      
    
    features = list(aa_composition.values()) + [
        molecular_weight, aromaticity, instability_index, isolectric_point
    ]
    return features

data = pd.read_csv('protein_interaction_dataset.csv')

protein1_features = np.array([extract_protein_features(seq) for seq in data['Protein1_Seq']])
protein2_features = np.array([extract_protein_features(seq) for seq in data['Protein2_Seq']])

X = np.hstack((protein1_features, protein2_features))

y = LabelEncoder().fit_transform(data['Label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
roc_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
print(f"ROC-AUC Score: {roc_auc}")

importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
print("Feature ranking:")

for f in range(X.shape[1]):
    print(f"{f + 1}. Feature {indices[f]} ({importances[indices[f]]})")
