import torch
from model import load_model
import pretty_midi

# Chord Mappings

chord_to_idx = {'C_major': 0, 'F_major': 1, 'G_major': 2}
reverse_note = {64: 'E4', 67: 'G4', 69: 'A4', 72: 'C5', 65: 'F4', 62: 'D4'}

input_dim = 3
hidden_dim = 64
output_dim = 128

model = load_model(input_dim, hidden_dim, output_dim, "melody_lstm.pth")

test_chords = ['C_major', 'F_major', 'G_major']
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