from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import torch
import pretty_midi
import pandas as pd
import tempfile
import requests
import json
import re
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
        instrument_req = (data.get('instrument') or 'piano').lower()
        total_bars = int(data.get('total_bars', 8) or 8)
        num_sections = int(data.get('num_sections', 1) or 1)
        modulation_prob = float(data.get('modulation_prob', 0.0) or 0.0)

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
        
        # Generate base melody for the provided progression
        input_chords = torch.tensor(chord_indices)
        
        with torch.no_grad():
            output = model(input_chords)
            predicted_notes_tensor = output.argmax(dim=1).squeeze()

        # Ensure list
        if predicted_notes_tensor.dim() == 0:
            base_predicted = [predicted_notes_tensor.item()]
        else:
            base_predicted = predicted_notes_tensor.tolist()

        # Build a longer melody according to total_bars and num_sections
        # Assume each chord roughly corresponds to one bar for now
        import math, random
        bars_per_cycle = max(1, len(base_predicted))
        repeats_needed = max(1, math.ceil(total_bars / bars_per_cycle))

        # Prepare section transpositions
        # We will optionally transpose each section up or down by 2 semitones
        section_transpositions = []
        for s in range(max(1, num_sections)):
            if random.random() < max(0.0, min(1.0, modulation_prob)):
                section_transpositions.append(random.choice([-2, 2, 5, -5]))
            else:
                section_transpositions.append(0)

        full_note_indices = []
        section_len_cycles = max(1, repeats_needed // max(1, num_sections))
        remaining_cycles = repeats_needed
        section_idx = 0
        while remaining_cycles > 0:
            cycles_this_section = min(section_len_cycles, remaining_cycles)
            transpose_semitones = section_transpositions[min(section_idx, len(section_transpositions)-1)]
            for _ in range(cycles_this_section):
                # Append transposed copy of base_predicted
                for idx in base_predicted:
                    # idx_to_note maps idx -> note name (e.g., 'C4'); we transpose after converting to pitch
                    full_note_indices.append((idx, transpose_semitones))
            remaining_cycles -= cycles_this_section
            section_idx += 1

        # Convert to MIDI
        midi = pretty_midi.PrettyMIDI()
        gm_program_map = {
            'piano': 0,
            'acoustic': 24,
            'violin': 40,
            'flute': 73,
            'synth': 88,
        }
        program_num = gm_program_map.get(instrument_req, 0)
        instrument = pretty_midi.Instrument(program=program_num)
        
        # Map to note names, apply transposition, and write
        predicted_note_names = []
        time = 0.0
        for idx, transpose in full_note_indices:
            note_name = idx_to_note.get(idx, 'C4')
            try:
                pitch = pretty_midi.note_name_to_number(note_name)
            except Exception:
                pitch = 60
            pitch = max(0, min(127, pitch + int(transpose)))
            # Update note_name for UI from pitch (normalize to closest name)
            try:
                ui_note_name = pretty_midi.note_number_to_name(pitch)
            except Exception:
                ui_note_name = note_name
            predicted_note_names.append(ui_note_name)
            note = pretty_midi.Note(
                velocity=100,
                pitch=pitch,
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
            'notes': predicted_note_names,
            'duration': duration_per_chord,
            'instrument': instrument_req,
            'total_bars': total_bars,
            'num_sections': num_sections,
            'modulation_prob': modulation_prob
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

@app.route('/api/musicxml/<file_id>')
def serve_musicxml(file_id):
    """Convert the generated MIDI to MusicXML and return it for sheet music rendering."""
    try:
        if file_id not in generated_files:
            return jsonify({'error': 'Invalid file ID'}), 400
        file_path = generated_files[file_id]['path']
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404

        try:
            from music21 import converter, musicxml
            # Parse MIDI into music21 stream
            s = converter.parse(file_path)
            # First try direct in-memory export
            try:
                exporter = musicxml.m21ToXml.GeneralObjectExporter(s)
                xml_string = exporter.parse()
                return app.response_class(
                    response=xml_string,
                    status=200,
                    mimetype='application/vnd.recordare.musicxml+xml'
                )
            except Exception:
                # Fallback: write to temp MusicXML via music21 write API
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.musicxml')
                tmp.close()
                try:
                    s.write('musicxml', fp=tmp.name)
                    with open(tmp.name, 'r', encoding='utf-8') as f:
                        xml_text = f.read()
                finally:
                    try:
                        os.unlink(tmp.name)
                    except Exception:
                        pass
                return app.response_class(
                    response=xml_text,
                    status=200,
                    mimetype='application/vnd.recordare.musicxml+xml'
                )
        except ImportError:
            return jsonify({'error': 'music21 is not installed on the server. Please pip install music21 and restart.'}), 500
        except Exception as e:
            return jsonify({'error': f'MusicXML conversion failed: {e}'}), 500

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

@app.route('/api/chat', methods=['POST'])
def chat_assistant():
    """Lightweight brainstorming chatbot for chord selection guidance"""
    try:
        data = request.get_json() or {}
        messages = data.get('messages', [])
        selected_chords = data.get('selected_chords', [])
        available_chords = list(chord_to_idx.keys()) if chord_to_idx else []

        # If no LLM key, provide a simple helpful fallback using dataset chord names
        if not openai.api_key:
            demo_progression = [c for c in [
                'C-major triad', 'G-major triad', 'A-minor triad', 'F-major triad'
            ] if c in available_chords][:4] or available_chords[:4]
            fallback_text = (
                "Here are some classic options to get you started. "
                "Try a bright diatonic loop with smooth voice leading.\n"
                f"Progression: {', '.join(demo_progression)}"
            )
            return jsonify({
                'success': True,
                'reply': fallback_text
            })

        # Build system prompt with context
        context = (
            "You are a friendly, expert chord progression coach for a melody generator web app. "
            "Help users pick chord progressions and briefly explain why they work for catchy melodies. "
            "Keep replies under ~120 words. Include exactly one concrete progression and a brief why. "
            "Rules for the progression line: start with 'Progression: ' and list chords as a comma-separated list. "
            "Use ONLY chord spellings from AVAILABLE_CHORDS. If a chord isn't available, substitute the closest available diatonic option. "
            "Avoid code blocks."
        )
        system_extra = (
            f"\nAVAILABLE_CHORDS: {', '.join(available_chords)}\n"
            f"CURRENT_SELECTION: {', '.join(selected_chords)}\n"
            "If the user asks for ideas in a key, prefer diatonic chords and common patterns (e.g., I–V–vi–IV, ii–V–I)."
        )
        system_msg = context + system_extra

        # Call NVIDIA NIM Chat Completions (using the configured OPENAI_API_KEY as bearer)
        url = 'https://integrate.api.nvidia.com/v1/chat/completions'
        headers = {
            'Authorization': f'Bearer {openai.api_key}',
            'Content-Type': 'application/json'
        }
        payload = {
            'model': 'meta/llama-3.1-8b-instruct',
            'messages': ([{"role": "system", "content": system_msg}] + messages),
            'max_tokens': 400,
            'temperature': 0.7
        }

        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code != 200:
            raise Exception(f"NVIDIA API error {response.status_code}: {response.text}")
        data = response.json()
        reply = data["choices"][0]["message"]["content"]

        # Sanitize the Progression line to ensure only dataset chords are suggested
        try:
            match = re.search(r"Progression:\s*([^\n]+)", reply, flags=re.IGNORECASE)
            if match and available_chords:
                suggested = [s.strip() for s in match.group(1).split(',') if s.strip()]
                filtered = [s for s in suggested if s in available_chords]
                if not filtered:
                    # Build a safe fallback progression from available chords
                    fallback = [c for c in [
                        'C-major triad', 'G-major triad', 'A-minor triad', 'F-major triad'
                    ] if c in available_chords][:4] or available_chords[:4]
                    filtered = fallback
                # Replace the Progression line in the reply
                reply = re.sub(r"Progression:\s*[^\n]+",
                               "Progression: " + ", ".join(filtered),
                               reply,
                               flags=re.IGNORECASE)
        except Exception:
            pass

        return jsonify({'success': True, 'reply': reply})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

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