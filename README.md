# Chatbot
# 🤖 AI Chatbot using Python, TensorFlow & NLTK

## 🧠 Project Summary
I’ve built a **Python-based AI Chatbot** that can understand user input, detect intents using a **trained neural network model**, and respond intelligently to various categories such as **greetings, jokes, weather, politics, math, and programming help**.  

This chatbot leverages **Natural Language Processing (NLP)** techniques with **TensorFlow**, **NLTK**, and a **custom intents dataset (`intents.json`)** to process user queries and generate meaningful responses.  

To enhance the user experience, I’ve added **colored text responses** (green for bot messages) and designed a **clean, interactive command-line interface**, making conversations more engaging and visually clear.  

The project demonstrates strong fundamentals in **machine learning, deep learning, and text preprocessing**, serving as a practical introduction to **intent classification and conversational AI systems**.

---

## ⚙️ Features
- 🧩 Intent recognition using a trained neural network model  
- 💬 10+ conversational categories (greetings, jokes, math, weather, etc.)  
- 🌈 Colored chatbot responses in the terminal  
- 📁 Customizable intents through `intents.json`  
- 🧠 NLP preprocessing using NLTK (tokenization, lemmatization)  
- ⚡ Built and trained using TensorFlow  

---

## 🛠️ Technologies Used
- Python  
- TensorFlow / Keras  
- NLTK (Natural Language Toolkit)  
- NumPy  
- JSON  

---

## 🚀 How to Run


# 1️⃣ Clone the repository


# 2️⃣ Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # For macOS/Linux
# or
.venv\Scripts\activate     # For Windows

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Run the chatbot
python chatbot.py

AI-Chatbot/
│
├── intents.json          
├── chatbot_model.keras   
├── words.pkl              
├── classes.pkl            
├── chatbot.py             
├── train_chatbot.py 
