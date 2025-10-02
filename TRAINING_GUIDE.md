# Q-Learning Car - Optimized Training Setup

Dieses Projekt wurde erheblich verbessert und optimiert für besseres Deep Q-Learning Training!

## 🚀 Was wurde verbessert:

### 1. **GPU-Unterstützung**
- PyTorch mit CUDA-Support installiert
- Automatische Geräteerkennung (GPU/CPU)
- Alle Tensoren werden automatisch auf die GPU verschoben

### 2. **Verbessertes Neuronales Netzwerk**
- Tiefere Architektur (128→128→64 statt 64→8)
- ReLU-Aktivierungen statt Tanh für bessere Gradienten
- Dropout-Regularisierung gegen Overfitting
- Gradient Clipping zur Stabilität

### 3. **Optimiertes Reward-System**
- Abstandsbasierte Belohnungen (näher zum Ziel = positiv)
- Geschwindigkeitsbasierte Belohnungen (Bewegung fördern)
- Sensorbasierte Belohnungen (Kollisionen vermeiden)
- Konfigurierbare Reward-Parameter

### 4. **Bessere Hyperparameter**
- Größere Replay-Buffer (100k statt 50k)
- Längere Exploration-Phase (50k steps)
- Niedrigere Lernrate für Stabilität
- MSE Loss statt Smooth L1 Loss

### 5. **Training Monitoring & Visualization**
- Automatische Plots der Trainingsfortschritte
- Speicherung aller Metriken
- Modell-Checkpoints alle 10k Steps
- Detaillierte Logs und Zusammenfassungen

### 6. **Einfache Konfiguration**
- `training_config.py` für alle Hyperparameter
- `start_training.py` für einfachen Start
- Automatische Umgebungsprüfung

## 🎮 STEP-BY-STEP ANLEITUNG

### Schritt 1: Training starten
```bash
python start_training.py
```

### Schritt 2: Strecke zeichnen 🎨
1. **Draw Boundaries Modus**: 
   - Mit der **Maus** Streckenbegrenzungen zeichnen
   - Zeichne eine geschlossene ovale/runde Strecke
   - **SPACE** drücken für nächsten Modus

2. **Draw Goals Modus**:
   - **Zielpunkte** entlang der Strecke setzen
   - Das Auto lernt diese Punkte in Reihenfolge anzufahren
   - **SPACE** drücken für nächsten Modus

### Schritt 3: Optional testen 🚗
3. **User Controls Modus** (Optional):
   - **Pfeiltasten** zum manuellen Fahren
   - Teste ob die Strecke gut fahrbar ist
   - **SPACE** für AI Training

### Schritt 4: AI Training starten 🤖
4. **AI Training Modus**:
   - Das Auto trainiert **automatisch**
   - Du siehst das Auto fahren und lernen!
   - Logs alle 1000 Steps in der Konsole
   - Plots alle 5000 Steps

## 🎯 Tipps für beste Ergebnisse

### Streckendesign:
- **Einfache ovale Strecke** für den Anfang
- **Nicht zu eng** - das Auto braucht Platz zum Lernen
- **Zielpunkte gleichmäßig verteilen** entlang der Strecke
- **Geschlossene Strecke** für kontinuierliches Training

### Training überwachen:
- **Reward steigt über Zeit** = Auto lernt
- **Epsilon fällt** = weniger Exploration, mehr Exploitation
- **Loss stabilisiert sich** = Training konvergiert
- **Auto fährt runder** = bessere Performance

## 📊 Automatisches Monitoring

### Was passiert automatisch:
- **Logs**: Alle 1000 Steps Fortschritts-Updates
- **Plots**: Alle 5000 Steps Visualisierungen
- **Modelle**: Alle 10000 Steps gespeicherte Checkpoints

### Verzeichnisstruktur nach Training:
```
├── training_logs/          # Alle Trainings-Metriken
├── models/                 # Gespeicherte Modelle
├── training_progress_*.png # Fortschritts-Plots
└── training_metrics_*.json # Rohdaten
```

## ⚙️ Konfiguration anpassen

Bearbeite `training_config.py` um Hyperparameter anzupassen:

```python
# Beispiel-Anpassungen:
LEARNING_RATE = 5e-5        # Langsameres Lernen
BATCH_SIZE = 128           # Größere Batches
EPSILON_DECAY = 100000     # Längere Exploration
CRASH_PENALTY = -200       # Härtere Bestrafung für Crashes
GOAL_REWARD = 100          # Höhere Belohnung für Ziele
```

## 🎯 Erwartete Verbesserungen

1. **Schnelleres Lernen**: Bessere Netzwerk-Architektur und Hyperparameter
2. **Stabileres Training**: Gradient Clipping und bessere Exploration
3. **GPU-Beschleunigung**: 10-50x schneller je nach GPU
4. **Bessere Fahrleistung**: Intelligenteres Reward-System
5. **Nachvollziehbarkeit**: Detailliertes Monitoring und Plots

## � Was das Auto lernt

### Phase 1 (0-10k Steps): Grundlagen
- Nicht gegen Wände fahren
- Grundlegende Bewegung
- Sensoren verstehen

### Phase 2 (10k-30k Steps): Navigation
- Richtung zu Zielen finden
- Kurven fahren lernen
- Geschwindigkeit kontrollieren

### Phase 3 (30k+ Steps): Optimierung
- Runde, effiziente Fahrweise
- Optimale Geschwindigkeit
- Präzise Zielanfahrt

## 🎉 Viel Erfolg!

Das Auto sollte jetzt viel besser lernen! Die Kombination aus GPU-Beschleunigung, besserer Architektur und optimierten Hyperparametern wird deutlich bessere Ergebnisse erzielen.

**Wichtig**: Lass das Training mehrere Stunden laufen - Q-Learning braucht Zeit, aber die Ergebnisse werden sich lohnen! 🏆