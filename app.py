import collections
import glob
import numpy as np
import pandas as pd
import pathlib
import pretty_midi
import tensorflow as tf
import streamlit as st

# Load the trained model
model = tf.keras.models.load_model("C:\\Users\\yadwa\\OneDrive\\Desktop\\BE project\\Model files\\final model of project.h5")

# Set random seed for reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Constants
vocab_size = 128
seq_length = 25

# Use the pre-downloaded and extracted MAESTRO dataset
data_dir = pathlib.Path('D:/BE project/data/maestro-v2.0.0')
filenames = glob.glob(str(data_dir / '**/*.mid*'))
print('Number of files:', len(filenames))

# Choose a sample file from the dataset
sample_file = filenames[1]  # or another index if you prefer

# Utility functions for MIDI processing
def midi_to_notes(midi_file: str) -> pd.DataFrame:
    pm = pretty_midi.PrettyMIDI(midi_file)
    instrument = pm.instruments[0]
    notes = collections.defaultdict(list)
    sorted_notes = sorted(instrument.notes, key=lambda note: note.start)
    prev_start = sorted_notes[0].start

    for note in sorted_notes:
        start = note.start
        end = note.end
        notes['pitch'].append(note.pitch)
        notes['start'].append(start)
        notes['end'].append(end)
        notes['step'].append(start - prev_start)
        notes['duration'].append(end - start)
        prev_start = start

    return pd.DataFrame({name: np.array(value) for name, value in notes.items()})

def notes_to_midi(notes: pd.DataFrame, out_file: str, instrument_name: str, velocity: int = 100) -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=pretty_midi.instrument_name_to_program(instrument_name))
    prev_start = 0
    for _, note in notes.iterrows():
        start = float(prev_start + note['step'])
        end = float(start + note['duration'])
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=int(note['pitch']),
            start=start,
            end=end,
        )
        instrument.notes.append(note)
        prev_start = start

    pm.instruments.append(instrument)
    pm.write(out_file)
    return pm

def predict_next_note(
    notes: np.ndarray,
    model: tf.keras.Model,
    temperature: float = 1.0) -> tuple[int, float, float]:
    """Generates a note as a tuple of (pitch, step, duration), using a trained sequence model."""
    assert temperature > 0
    inputs = tf.expand_dims(notes, 0)
    predictions = model.predict(inputs)
    pitch_logits = predictions['pitch']
    step = predictions['step']
    duration = predictions['duration']
    pitch_logits /= temperature
    pitch = tf.random.categorical(pitch_logits, num_samples=1)
    pitch = tf.squeeze(pitch, axis=-1)
    duration = tf.squeeze(duration, axis=-1)
    step = tf.squeeze(step, axis=-1)
    step = tf.maximum(0, step)
    duration = tf.maximum(0, duration)
    return int(pitch), float(step), float(duration)

# Streamlit app
st.title("Fun Soundify 😍")

# Parameters input
temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
num_predictions = st.number_input("Number of Predictions", min_value=1, max_value=500, value=120)

if st.button("Generate Music"):
    raw_notes = midi_to_notes(sample_file)
    st.write("Sample File:", sample_file)
    st.write("Raw Notes:", raw_notes.head())

    instrument_name = pretty_midi.program_to_instrument_name(raw_notes['pitch'][0])
    sample_notes = np.stack([raw_notes[key] for key in ['pitch', 'step', 'duration']], axis=1)
    input_notes = sample_notes[:seq_length] / np.array([vocab_size, 1, 1])
    st.write("Initial Input Notes:", input_notes)

    generated_notes = []
    prev_start = 0

    for _ in range(num_predictions):
        pitch, step, duration = predict_next_note(input_notes, model, temperature)
        start = prev_start + step
        end = start + duration
        input_note = (pitch, step, duration)
        generated_notes.append((*input_note, start, end))
        input_notes = np.delete(input_notes, 0, axis=0)
        input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
        prev_start = start

    generated_notes_df = pd.DataFrame(generated_notes, columns=['pitch', 'step', 'duration', 'start', 'end'])
    st.write("Generated Notes DataFrame:", generated_notes_df.head())

    out_file = 'generated_output.midi'
    out_pm = notes_to_midi(generated_notes_df, out_file=out_file, instrument_name=instrument_name)

    st.success("Music generation complete. Click below to download the generated MIDI file.")
    with open(out_file, 'rb') as file:
        st.download_button("Download MIDI", file, file_name=out_file)
