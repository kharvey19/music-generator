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
import openai

from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app)

openai.api_key = os.environ.get('OPENAI_API_KEY')

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
            predicted_notes_tensor = output.argmax(dim=1).squeeze()

        # Ensure predicted_notes is a list, even for a single note
        if predicted_notes_tensor.dim() == 0:
            predicted_notes = [predicted_notes_tensor.item()]
        else:
            predicted_notes = predicted_notes_tensor.tolist()
        
        # Convert to MIDI
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)  # Grand Piano
        
        time = 0.0
        for note_num in predicted_notes:
            note = pretty_midi.Note(
                velocity=100,
                pitch=note_num,
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
            'notes': [idx_to_note.get(idx, f"MIDI:{idx}") for idx in predicted_notes],
            'duration': duration_per_chord
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
    
    if not openai.api_key:
        num_notes = len(melody_notes) if hasattr(melody_notes, '__len__') else 0
        num_chords = len(selected_chords) if hasattr(selected_chords, '__len__') else 0
        total_duration = duration * num_notes
        chords_list = ", ".join(selected_chords) if isinstance(selected_chords, list) else str(selected_chords)

        return f"""
        <h4>🎵 Music Theory Analysis</h4>
        <p style=\"color: #dc3545;\">⚠️ LLM API key not configured. Please set the OPENAI_API_KEY environment variable to enable AI-powered music theory analysis.</p>
        <p>For now, here's a basic analysis:</p>
        <ul>
            <li><strong>Melody:</strong> {num_notes} notes over {num_chords} chords</li>
            <li><strong>Duration:</strong> {total_duration:.1f} seconds</li>
            <li><strong>Chords:</strong> {chords_list}</li>
        </ul>
        """
    
    try:
        # Prepare a concise prompt for the LLM
        prompt = f"""
        You are an expert music theorist and educator.

        Analyze this AI-generated melody concisely.
        MELODY DATA:
        - Notes: {melody_notes}
        - Chords: {selected_chords}
        - Duration per chord: {duration} seconds
        - Total duration: {duration * len(melody_notes):.1f} seconds

        Return HTML ONLY as a short unordered list with 4–6 bullet points (<ul><li>...</li></ul>).
        Each bullet should be one sentence, clear and actionable.
        No headings, no preambles, no code blocks, no extra text outside the <ul>.
        Keep the total under ~120 words.
        """

        # Use NVIDIA NIM endpoint directly when an nvapi key is detected to avoid SDK routing issues
        if openai.api_key:
            url = 'https://integrate.api.nvidia.com/v1/chat/completions'
            headers = {
                'Authorization': f'Bearer {openai.api_key}',
                'Content-Type': 'application/json'
            }
            payload = {
                'model': 'meta/llama-3.1-8b-instruct',
                'messages': [
                    {"role": "system", "content": "You are an expert music theorist and educator. Provide clear, engaging, and educational music theory analysis in HTML format."},
                    {"role": "user", "content": prompt}
                ],
                'max_tokens': 400,
                'temperature': 0.7
            }
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            if response.status_code != 200:
                raise Exception(f"NVIDIA API error {response.status_code}: {response.text}")
            data = response.json()
            llm_analysis = data["choices"][0]["message"]["content"]
        else:
            completion = openai.ChatCompletion.create(
                model='meta/llama-3.1-8b-instruct',
                messages=[
                    {"role": "system", "content": "You are an expert music theorist and educator. Provide clear, engaging, and educational music theory analysis in HTML format."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400,
                temperature=0.7
            )
            llm_analysis = completion["choices"][0]["message"]["content"]

        return f"""
        {llm_analysis}
        """
    except Exception as e:
        message = str(e)
        if "401" in message or "Unauthorized" in message:
            return f"""
            <h4>🎵 Music Theory Analysis</h4>
            <p style=\"color: #dc3545;\">⚠️ Unauthorized (401) from LLM service.</p>
            <ul>
                <li>If using <strong>OpenAI</strong>: verify <code>OPENAI_API_KEY</code> is correct (starts with <code>sk-</code>) and that your account has access to <code>meta/llama-3.1-8b-instruct</code>.</li>
                <li>If using <strong>NVIDIA NIM</strong>: use your <code>nvapi-</code> key in <code>OPENAI_API_KEY</code> or <code>NVIDIA_API_KEY</code>. Ensure the API base is <code>https://integrate.api.nvidia.com/v1</code> and the model is a NIM chat model (e.g., <code>meta/llama-3.1-8b-instruct</code>).</li>
                <li>Restart the server after updating environment variables.</li>
            </ul>
            """
        if "404" in message and "NVIDIA" in message:
            return f"""
            <h4>🎵 Music Theory Analysis</h4>
            <p style=\"color: #dc3545;\">⚠️ 404 from NVIDIA endpoint.</p>
            <ul>
                <li>Verify the endpoint <code>https://integrate.api.nvidia.com/v1/chat/completions</code> is reachable from your network.</li>
                <li>Confirm the model name <code>meta/llama-3.1-8b-instruct</code> is available to your API key. Try <code>meta/llama-3.1-8b-instruct</code>.</li>
                <li>Ensure there are no corporate proxies/firewalls blocking the request.</li>
            </ul>
            """
        return f"""
        <h4>🎵 Music Theory Analysis</h4>
        <p style=\"color: #dc3545;\">⚠️ Error generating analysis: {message}</p>
        <p>Please try again after checking your API key and internet connection.</p>
        """







if __name__ == '__main__':
    print("Loading melody generation model...")
    if load_melody_model():
        print("Model loaded successfully!")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model!") 