import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.api.models import Sequential
from keras.api.layers import Embedding, LSTM, Dense, SpatialDropout1D, Dropout
from keras.api.preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from keras.api.callbacks import EarlyStopping


# Load dataset
df = pd.read_csv("evaluation.csv", sep=';')
df = df.drop('index', axis=1)
df['text'] = df['title'] + ' ' + df['text']

# Features and Labels
X = df['text']
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenization
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Padding sequences
X_train_pad = pad_sequences(X_train_seq, maxlen=80, padding='post', truncating='post')  # Reduced maxlen
X_test_pad = pad_sequences(X_test_seq, maxlen=80, padding='post', truncating='post')

# Model architecture
model = Sequential()

# Use pre-trained embeddings if possible (e.g., GloVe) or keep current embeddings
model.add(Embedding(input_dim=10000, output_dim=128, input_length=80))  # Adjusted for new maxlen
model.add(SpatialDropout1D(0.3))  # Slightly higher dropout to reduce overfitting
model.add(LSTM(64, dropout=0.3, recurrent_dropout=0.3))  # Fewer units to lighten the model
model.add(Dropout(0.5))  # Additional Dropout layer before dense layer
model.add(Dense(1, activation='sigmoid'))  # Binary classification

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Model summary
model.summary()

# Early stopping callback to prevent overfitting and reduce epochs
early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# Train the model
history = model.fit(X_train_pad, y_train, epochs=10, batch_size=64, validation_data=(X_test_pad, y_test),
                    verbose=2, callbacks=[early_stopping])

# Evaluate the model
loss, accuracy = model.evaluate(X_test_pad, y_test, verbose=2)
print(f"Test Accuracy: {accuracy:.4f}")

# Prediction and confusion matrix
y_pred = (model.predict(X_test_pad) >= 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake', 'True'])
disp.plot(cmap='Blues')
plt.show()

# Save tokenizer 
