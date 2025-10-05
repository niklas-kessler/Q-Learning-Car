"""
Simplified training starter script for Q-Learning Car.
Run this to start training with optimized settings.
"""
import os

# Prevent OpenMP library conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
from training_config import *

def check_environment():
    """Check if the environment is set up correctly."""
    
    print("="*50)
    print("Q-LEARNING CAR - TRAINING ENVIRONMENT CHECK")
    print("="*50)
    
    # Check PyTorch
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA current device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
    
    # Show selected device from config
    print(f"Selected device: {DEVICE}")
    
    print("="*50)
    print("TRAINING HYPERPARAMETERS")
    print("="*50)
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Buffer Size: {BUFFER_SIZE}")
    print(f"Min Replay Size: {MIN_REPLAY_SIZE}")
    print(f"Gamma (Discount): {GAMMA}")
    print(f"Epsilon Start: {EPSILON_START}")
    print(f"Epsilon End: {EPSILON_END}")
    print(f"Epsilon Decay: {EPSILON_DECAY}")
    print(f"Network Architecture: {NETWORK_HIDDEN_LAYERS}")
    print("="*50)
    
    return DEVICE

def start_training():
    """Start the training process."""
    device = check_environment()
    
    print("\\n" + "="*60)
    print("🏁 Q-LEARNING CAR - TRAINING SETUP")
    print("="*60)
    print("\\n📋 WICHTIGE ANWEISUNGEN:")
    print("\\n1. 🎨 ZEICHNE ZUERST DIE STRECKE:")
    print("   • Das Spiel startet im 'Draw Boundaries' Modus")
    print("   • Zeichne die Streckenbegrenzungen mit der MAUS")
    print("   • Drücke SPACE um zum nächsten Modus zu wechseln")
    print("\\n2. 🎯 ZEICHNE DIE ZIELE:")
    print("   • Im 'Draw Goals' Modus zeichne Zielpunkte")
    print("   • Das Auto lernt diese Punkte anzufahren")
    print("   • Drücke wieder SPACE für den nächsten Modus")
    print("\\n3. 🚗 TESTE MANUELL (Optional):")
    print("   • 'User Controls' Modus zum manuellen Testen")
    print("   • Pfeiltasten zum Fahren")
    print("   • Drücke SPACE für AI Training")
    print("\\n4. 🤖 AI TRAINING:")
    print("   • Das Auto trainiert automatisch")
    print("   • Du siehst das Auto fahren und lernen!")
    print("   • Logs erscheinen alle 1000 Steps in der Konsole")
    print("   • Plots werden alle 5000 Steps generiert")
    print("\\n💡 TIPP: Zeichne eine einfache ovale Strecke für beste Ergebnisse!")
    print("\\n" + "="*60)
    print("\\nStarte das Spiel...")
    
    # Import and start the game
    try:
        import racegame
        import pyglet as pg
        
        print("\\n✅ Spiel geladen! Das Fenster sollte sich öffnen.")
        print("\\n🎮 STEUERUNG:")
        print("   • SPACE = Nächster Modus") 
        print("   • Maus = Zeichnen (in Draw Modi)")
        print("   • Pfeiltasten = Fahren (in User Controls)")
        print("   • ESC oder Fenster schließen = Beenden")
        print("\\n🤖 Das Training läuft automatisch im AI Training Modus!")
        print("    Logs und Plots werden automatisch erstellt.")
        
        # Start the main game loop
        pg.app.run()
        
    except KeyboardInterrupt:
        print("\\n\\n⏹️ Training durch Benutzer beendet.")
        print("📊 Speichere Trainingsergebnisse...")
        if 'racegame' in locals():
            try:
                racegame.save_training_results()
                print("✅ Ergebnisse gespeichert!")
            except Exception as e:
                print(f"❌ Fehler beim Speichern: {e}")
    except Exception as e:
        print(f"\\n❌ Fehler beim Starten: {e}")
        import traceback
        traceback.print_exc()
        print("\\n🔧 Stelle sicher, dass alle Abhängigkeiten installiert sind.")

if __name__ == "__main__":
    start_training()