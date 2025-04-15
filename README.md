import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Caricamento del dataset (ipotetico)
# Assumiamo che il dataset contenga colonne come 'Gol_Squadra_A', 'Gol_Squadra_B', 'Possesso_Palla_A', 
# 'Possesso_Palla_B', 'Infortuni_A', 'Infortuni_B', ecc.
df = pd.read_csv("serie_a_results.csv")

# Preprocessing: selezione delle variabili e creazione delle etichette (win = 1 per vittoria, 0 per pareggio/sconfitta)
df['Risultato'] = (df['Gol_Squadra_A'] > df['Gol_Squadra_B']).astype(int)

# Selezioniamo le colonne caratteristiche (X) e la variabile target (y)
X = df[['Gol_Squadra_A', 'Gol_Squadra_B', 'Possesso_Palla_A', 'Possesso_Palla_B', 'Infortuni_A', 'Infortuni_B']]
y = df['Risultato']

# Divisione in set di addestramento e di test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creazione e addestramento del modello Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predizioni sul set di test
y_pred = rf.predict(X_test)

# Calcolo dell'accuratezza del modello
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuratezza del modello: {accuracy:.2f}")

# Funzione di previsione per nuove partite
def predici_partita(gol_a, gol_b, possesso_a, possesso_b, infortuni_a, infortuni_b):
    partita = pd.DataFrame([[gol_a, gol_b, possesso_a, possesso_b, infortuni_a, infortuni_b]], columns=X.columns)
    previsione = rf.predict(partita)
    return "Vittoria" if previsione[0] == 1 else "Non vittoria"  # Vittoria o pareggio/sconfitta

# Esempio di previsione
print(predici_partita(2, 1, 55, 45, 1, 2))  # Esempio di partita
