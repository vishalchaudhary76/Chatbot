import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?','!',',','.']

print("Processing intents...")
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each pattern
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

print(f"Found {len(documents)} patterns")
print(f"Raw words count: {len(words)}")

# Lemmatize and clean words
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))  # Remove duplicates

classes = sorted(set(classes))

print(f"✅ Final vocabulary: {len(words)} words")
print(f"✅ Classes: {classes}")
print(f"Sample words: {words[:10]}...")  # Show first 10 words

# Save words and classes
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))
print("✅ Saved words and classes to pickle files")

# Create training data
training = []
output_empty = [0] * len(classes)

print("Creating training data...")
for document in documents:
    bag = []
    word_patterns = document[0]
    
    # Lemmatize and lowercase the pattern words
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    
    # Create bag of words
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    # Create output row
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    
    training.append([bag, output_row])

print(f"Training samples: {len(training)}")

# Shuffle training data
random.shuffle(training)
training = np.array(training, dtype=object)

# Split into features and labels
train_x = np.array([item[0] for item in training])
train_y = np.array([item[1] for item in training])

print(f"Training data shape - X: {train_x.shape}, Y: {train_y.shape}")

# Build the model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

print("Starting training...")
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save model in the new Keras format
model.save('chatbot_model.keras')
print("✅ Model training completed and saved as 'chatbot_model.keras'")

