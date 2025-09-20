import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- App Title and Description ---
st.title("Next Word Prediction 📝")
st.write("Choose a model (LSTM or GRU) and enter a sequence of words to predict the next one.")

# --- Model Selection ---
model_choice = st.selectbox("Choose a model", ("LSTM", "GRU"))

# --- Load the Selected Model and Tokenizer ---
try:
    if model_choice == "LSTM":
        model = load_model('next_word_lstm.h5')
        st.info("LSTM model loaded successfully! 👍")
    else: # GRU
        model = load_model('next_word_gru.h5')
        st.info("GRU model loaded successfully! 👍")

    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

except Exception as e:
    st.error(f"Error loading model or tokenizer: {e}")
    st.stop() # Stop the app if files are not found

# --- Prediction Function ---
def predict_next_word(model, tokenizer, text, max_sequence_len):
    """Predicts the next word in a sequence."""
    token_list = tokenizer.texts_to_sequences([text])[0]
    
    # Ensure the sequence length is appropriate
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
        
    # Pad the sequence
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    
    # Predict probabilities
    predicted_probs = model.predict(token_list, verbose=0)
    
    # Get the index of the word with the highest probability
    predicted_word_index = np.argmax(predicted_probs, axis=1)[0]
    
    # Find the corresponding word
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
            
    return None

# --- Streamlit User Interface ---
input_text = st.text_input("Enter a sequence of words", "to be or not to")

if st.button("Predict Next Word"):
    if input_text:
        # Determine the max sequence length from the loaded model's input shape
        max_sequence_len = model.input_shape[1] + 1
        
        # Get the prediction
        next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        
        if next_word:
            st.success(f"The predicted next word is: **{next_word}**")
        else:
            st.warning("Could not predict the next word.")
    else:
        st.warning("Please enter a sequence of words.")