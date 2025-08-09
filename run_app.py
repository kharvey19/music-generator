#!/usr/bin/env python3
"""
Simple startup script for the Melody Generator Flask app
"""

from app import app, load_melody_model

if __name__ == '__main__':
    print("🎵 Starting AI Melody Generator...")
    print("Loading melody generation model...")
    
    if load_melody_model():
        print("✅ Model loaded successfully!")
        print("🌐 Starting web server...")
        print("📱 Open your browser and go to: http://localhost:5000")
        print("🛑 Press Ctrl+C to stop the server")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Failed to load model!")
        print("Please check that:")
        print("  - The model file 'melody_generator/music21_lstm.pth' exists")
        print("  - The data file 'melody_generator/data/chord_melody_pairs.csv' exists")
        print("  - All dependencies are installed (pip install -r requirements.txt)") 