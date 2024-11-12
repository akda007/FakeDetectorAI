import pickle
from keras.api.preprocessing.sequence import pad_sequences
from keras.api.models import load_model

def predict_fake_or_true(text):
    model = load_model("model.keras")
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    text_seq = tokenizer.texts_to_sequences([text])
    text_pad = pad_sequences(text_seq, maxlen=100, padding='post', truncating='post')
    prediction = model.predict(text_pad)
    return 'True' if prediction >= 0.5 else 'Fake'

# Example usage:
new_text = "Israeli football fans and the violence in Amsterdam: what we know The trouble in the Netherlands when Maccabi Tel Aviv played Ajax last week horrified people around the world Violence in Amsterdam around a Europa League football match between the local team Ajax and Israel’s Maccabi Tel Aviv sparked horror around the world, against a backdrop of soaring antisemitic and Islamophobic abuse and attacks across Europe fuelled by the Middle East conflict. The Amsterdam mayor, Femke Halsema, has said she had not been told the match was high-risk, although earlier last week the Turkish club Beşiktaş moved their match against Maccabi to a neutral country for fear of “provocative actions”. What happened on Wednesday night? The first incidents were reported on Wednesday evening, the day before the match. Police say Maccabi fans tore a Palestinian flag down from the facade of a building and burned it, shouted “fuck you, Palestine”, and vandalised a taxi."
result = predict_fake_or_true(new_text)
print(f"The article is: {result}")