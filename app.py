from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import torch
import pretty_midi
import pandas as pd
import tempfile
import requests
import json
from melody_generator.model import load_model

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app)

# LLM Configuration
LLM_API_KEY = os.environ.get('OPENAI_API_KEY')  # Set this environment variable
LLM_API_URL = "https://api.openai.com/v1/chat/completions"
LLM_MODEL = "gpt-4"  # or "gpt-3.5-turbo" for faster/cheaper analysis

# Global variables for the model
model = None
chord_to_idx = None
note_to_idx = None
idx_to_note = None

# Store generated MIDI files temporarily
generated_files = {}

def load_melody_model():
    """Load the melody generation model and mappings"""
    global model, chord_to_idx, note_to_idx, idx_to_note
    
    try:
        # Load data
        data_path = os.path.join('melody_generator', 'data', 'chord_melody_pairs.csv')
        data = pd.read_csv(data_path)
        
        # Extract unique chords and notes
        unique_chords = sorted(data['Chord'].unique())
        unique_notes = sorted(data['Note'].unique())
        
        # Create mappings
        chord_to_idx = {chord: idx for idx, chord in enumerate(unique_chords)}
        note_to_idx = {note: idx for idx, note in enumerate(unique_notes)}
        idx_to_note = {idx: note for note, idx in note_to_idx.items()}
        
        # Load model
        input_dim = len(chord_to_idx)
        hidden_dim = 64
        output_dim = 128
        
        model_path = os.path.join('melody_generator', 'music21_lstm.pth')
        model = load_model(input_dim, hidden_dim, output_dim, model_path)
        model.eval()
        
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

@app.route('/')
def index():
    """Main page with the melody generator UI"""
    return render_template('index.html')

@app.route('/api/chords')
def get_available_chords():
    """Get list of available chords for the UI"""
    if chord_to_idx is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    chords = list(chord_to_idx.keys())
    return jsonify({'chords': chords})

@app.route('/api/generate', methods=['POST'])
def generate_melody():
    """Generate melody based on selected chords and duration"""
    try:
        data = request.get_json()
        selected_chords = data.get('chords', [])
        duration_per_chord = data.get('duration', 0.5)  # seconds per chord
        
        if not selected_chords:
            return jsonify({'error': 'No chords selected'}), 400
        
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Convert chords to indices
        chord_indices = []
        for chord in selected_chords:
            if chord in chord_to_idx:
                chord_indices.append(chord_to_idx[chord])
            else:
                return jsonify({'error': f'Chord not found: {chord}'}), 400
        
        # Generate melody
        input_chords = torch.tensor(chord_indices)
        
        with torch.no_grad():
            output = model(input_chords)
            predicted_notes = output.argmax(dim=1).squeeze()
        
        # Convert to MIDI
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)  # Grand Piano
        
        time = 0.0
        for note_num in predicted_notes:
            note = pretty_midi.Note(
                velocity=100,
                pitch=note_num.item(),
                start=time,
                end=time + duration_per_chord
            )
            instrument.notes.append(note)
            time += duration_per_chord
        
        midi.instruments.append(instrument)
        
        # Save to temporary file with unique ID
        import uuid
        file_id = str(uuid.uuid4())
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mid')
        midi.write(temp_file.name)
        temp_file.close()
        
        # Store file info for later access
        generated_files[file_id] = {
            'path': temp_file.name,
            'filename': f'melody_{file_id}.mid'
        }
        
        return jsonify({
            'success': True,
            'file_id': file_id,
            'notes': [idx_to_note.get(idx.item(), f"MIDI:{idx.item()}") for idx in predicted_notes]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/midi/<file_id>')
def serve_midi(file_id):
    """Serve MIDI file for audio playback"""
    try:
        if file_id in generated_files:
            file_path = generated_files[file_id]['path']
            if os.path.exists(file_path):
                return send_file(file_path, mimetype='audio/midi')
            else:
                return jsonify({'error': 'File not found'}), 404
        else:
            return jsonify({'error': 'Invalid file ID'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/<file_id>')
def download_midi(file_id):
    """Download the generated MIDI file"""
    try:
        if file_id in generated_files:
            file_path = generated_files[file_id]['path']
            filename = generated_files[file_id]['filename']
            
            if os.path.exists(file_path):
                return send_file(
                    file_path, 
                    as_attachment=True, 
                    download_name=filename,
                    mimetype='audio/midi'
                )
            else:
                return jsonify({'error': 'File not found'}), 404
        else:
            return jsonify({'error': 'Invalid file ID'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cleanup', methods=['POST'])
def cleanup_files():
    """Clean up old generated files"""
    try:
        data = request.get_json()
        file_id = data.get('file_id')
        
        if file_id and file_id in generated_files:
            file_path = generated_files[file_id]['path']
            if os.path.exists(file_path):
                os.unlink(file_path)
            del generated_files[file_id]
            return jsonify({'success': True})
        
        return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_melody():
    """Analyze the generated melody using music theory"""
    try:
        data = request.get_json()
        melody_notes = data.get('melody', [])
        selected_chords = data.get('chords', [])
        duration = data.get('duration', 0.5)
        
        if not melody_notes or not selected_chords:
            return jsonify({'error': 'Missing melody or chord data'}), 400
        
        # Create a comprehensive music theory analysis
        analysis = generate_music_theory_analysis(melody_notes, selected_chords, duration)
        
        return jsonify({
            'success': True,
            'analysis': analysis
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_music_theory_analysis(melody_notes, selected_chords, duration):
    """Generate music theory analysis using LLM"""
    
    if not LLM_API_KEY:
        return """
        <h4>🎵 Music Theory Analysis</h4>
        <p style="color: #dc3545;">⚠️ LLM API key not configured. Please set the OPENAI_API_KEY environment variable to enable AI-powered music theory analysis.</p>
        <p>For now, here's a basic analysis:</p>
        <ul>
            <li><strong>Melody:</strong> {len(melody_notes)} notes over {len(selected_chords)} chords</li>
            <li><strong>Duration:</strong> {duration * len(selected_chords):.1f} seconds</li>
            <li><strong>Chords:</strong> {', '.join(selected_chords)}</li>
        </ul>
        """.format(len(melody_notes), len(selected_chords), duration * len(selected_chords), ', '.join(selected_chords))
    
    try:
        # Prepare the prompt for the LLM
        prompt = f"""
        You are an expert music theorist and educator. Analyze the following AI-generated melody and provide comprehensive music theory insights.

        MELODY DATA:
        - Notes: {melody_notes}
        - Chords: {selected_chords}
        - Duration per chord: {duration} seconds
        - Total duration: {duration * len(selected_chords):.1f} seconds

        Please provide a detailed analysis covering:

        1. **Chord Progression Analysis**: 
           - Identify the chord progression pattern
           - Explain the harmonic function and emotional character
           - Note any common or interesting progressions

        2. **Melody Analysis**:
           - Analyze the melodic contour and structure
           - Identify any motifs or patterns
           - Discuss the rhythm and note density
           - Note the relationship between melody and harmony

        3. **Music Theory Insights**:
           - Explain the key and scale relationships
           - Discuss consonance/dissonance
           - Identify any interesting harmonic or melodic features
           - Suggest musical context or genre associations

        4. **Educational Notes**:
           - Provide insights that would help someone learn music theory
           - Explain any advanced concepts in simple terms
           - Suggest ways to expand or develop the melody

        Format your response in HTML with appropriate headings (h4, h5) and paragraphs. Be engaging, educational, and specific to the musical material provided.
        """

        # Call the LLM API
        headers = {
            "Authorization": f"Bearer {LLM_API_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": LLM_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert music theorist and educator. Provide clear, engaging, and educational music theory analysis in HTML format."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 1500,
            "temperature": 0.7
        }
        
        response = requests.post(LLM_API_URL, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        llm_analysis = result['choices'][0]['message']['content']
        
        # Format the response
        formatted_analysis = f"""
        <h4>🎵 AI-Powered Music Theory Analysis</h4>
        <p><em>Analysis generated using {LLM_MODEL}</em></p>
        {llm_analysis}
        """
        
        return formatted_analysis
        
    except requests.exceptions.RequestException as e:
        return f"""
        <h4>🎵 Music Theory Analysis</h4>
        <p style="color: #dc3545;">⚠️ Error connecting to LLM service: {str(e)}</p>
        <p>Please check your internet connection and API key configuration.</p>
        """
    except Exception as e:
        return f"""
        <h4>🎵 Music Theory Analysis</h4>
        <p style="color: #dc3545;">⚠️ Error generating analysis: {str(e)}</p>
        <p>Please try again or contact support if the issue persists.</p>
        """







if __name__ == '__main__':
    print("Loading melody generation model...")
    if load_melody_model():
        print("Model loaded successfully!")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model!") 