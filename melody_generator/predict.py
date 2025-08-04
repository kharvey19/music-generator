import torch
from model import load_model
import pretty_midi
import pandas as pd

# Load data
data = pd.read_csv("data/chord_melody_pairs.csv")

# Extract unique chords and notes from DataFrame columns
unique_chords = sorted(data['Chord'].unique())
unique_notes = sorted(data['Note'].unique())

# Step 2: Assign each one an index
chord_to_idx = {chord: idx for idx, chord in enumerate(unique_chords)}
note_to_idx = {note: idx for idx, note in enumerate(unique_notes)}

# Optional: reverse maps for decoding
idx_to_chord = {idx: chord for chord, idx in chord_to_idx.items()}
reverse_note = {idx: note for note, idx in note_to_idx.items()}

input_dim = len(chord_to_idx)
hidden_dim = 64  # This must match the saved model (was 64, not 128)
output_dim = 128  # This must match the saved model, regardless of actual unique notes

print(f"Model dimensions: input_dim={input_dim}, hidden_dim={hidden_dim}, output_dim={output_dim}")
print(f"Actual unique notes in data: {len(unique_notes)}")

model = load_model(input_dim, hidden_dim, output_dim, "music21_lstm.pth")

test_chords = ['C-major triad', 'F-major triad', 'G-major triad']
# Check if test chords exist in our vocabulary
missing_chords = [chord for chord in test_chords if chord not in chord_to_idx]
if missing_chords:
    print(f"Warning: Test chords not found in training data: {missing_chords}")
    # Use some chords that esxist in the data
    available_chords = list(chord_to_idx.keys())[:3]
    test_chords = available_chords
    print(f"Using available chords instead: {test_chords}")

input_chords = torch.tensor([chord_to_idx[chord] for chord in test_chords])

with torch.no_grad():
    output = model(input_chords)
    predicted_notes = output.argmax(dim=1).squeeze()

print("Generated Melody:")
for idx in predicted_notes:
    print(reverse_note.get(idx.item(), f"MIDI:{idx.item()}"), end=" ")
print()

# Convert to MIDI
midi = pretty_midi.PrettyMIDI()
instrument = pretty_midi.Instrument(program=0) # Grand Piano

tim = 0.0 
duration = 0.5

for note_num in predicted_notes:
    note = pretty_midi.Note(
        velocity=100,
        pitch=note_num.item(),
        start=tim,
        end=tim + duration
    )
    instrument.notes.append(note)
    tim += duration

midi.instruments.append(instrument)
midi.write("generated_melody.mid")
print("Generated MIDI file saved as generated_melody.mid")