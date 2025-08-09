# Melody Generator with LSTM
This project uses a simple LSTM-based model to generate melodies over chord progressions using music theory and machine learning.

![Image](auto-music-.jpg)

## Features

- **AI-Powered Melody Generation**: Uses LSTM neural networks trained on classical chorale music
- **Interactive Web UI**: Modern, responsive interface for easy chord selection and melody generation
- **Customizable Duration**: Set how long each chord plays in your melody
- **MIDI Export**: Download generated melodies as MIDI files for use in music software
- **Real-time Generation**: Instant melody creation based on your chord choices
- **LLM-Powered Music Theory Analysis**: Advanced AI analysis of your generated melodies using GPT

## Install Dependencies
```bash
pip install -r requirements.txt
```

## LLM Integration Setup (Optional)

To enable AI-powered music theory analysis, you'll need an OpenAI API key:

1. **Get an API Key**: Sign up at [OpenAI](https://platform.openai.com/) and get your API key
2. **Set Environment Variable**: 
   ```bash
   export OPENAI_API_KEY="your_api_key_here"
   ```
   Or create a `.env` file in the project root with:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```
3. **Restart the Application**: The LLM analysis will be automatically enabled

**Note**: Without an API key, you'll still get basic melody generation, but the music theory analysis will be limited.

## How to Run

### Option 1: Web UI (Recommended)
```bash
python run_app.py
```
Then open your browser and go to: http://localhost:5000

### Option 2: Command Line
```bash
python melody_generator/predict.py
```

## Using the Web UI

1. **Select Chords**: Click on the chords you want to use in your progression
2. **Set Duration**: Choose how long each chord should play (0.1 to 2.0 seconds)
3. **Generate**: Click "Generate Melody" to create your AI-generated melody
4. **Play/Download**: Use the generated melody or download it as a MIDI file
5. **Analyze**: Click "Analyze Melody" to get AI-powered music theory insights

## Results
The melody will be generated in real-time and displayed as note names. You can also download it as a MIDI file titled `generated_melody.mid`.

## Technical Details

- **Model**: LSTM neural network with embedding layer
- **Training Data**: Classical chorale music from music21 corpus
- **Input**: Chord progressions (e.g., C-major, F-major, G-major)
- **Output**: Harmonically appropriate melody notes
- **Architecture**: Embedding → LSTM → Fully Connected layers

## File Structure

```
music/
├── app.py                 # Flask web application
├── run_app.py            # Startup script
├── templates/
│   └── index.html        # Web UI template
├── melody_generator/     # Core ML functionality
│   ├── model.py          # LSTM model definition
│   ├── predict.py        # Command-line prediction
│   ├── data/             # Training data
│   └── *.pth            # Trained model files
└── requirements.txt      # Python dependencies
```




