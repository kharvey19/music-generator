# Melody Generator with LSTM
This project uses a simple LSTM-based model to generate melodies over chord progressions using music theory and machine learning.

## How to run 

```
python predict.py
```

## Results
The melody will be downloaded in a file titles `generated_melody.mid`

Idea: Train a model that learns to write melodies that match a chord progression

1. It sees a sequence of chords and generates a melody that sounds good on top of those chords 



What we need:

1. Collect data: chord progressions and melody notes that go with them 
2. Convert music to numbers: for the model 
3. Train the model to learn those patterns

LSTM model (Long Short-Term Memory): Memory Based model that learns sequences like sentences, melodies, or time series 

