# 🧠 Emotion Recognition in Multilingual Text Using BiLSTM

## 📌 Introduction

Understanding human emotions from text is a key aspect of **Natural Language Processing (NLP)**, particularly in the domains of **sentiment analysis**, **human-computer interaction**, and **affective computing**. This project focuses on **Emotion Recognition in Multilingual Text** using a **Bidirectional Long Short-Term Memory (BiLSTM)** based deep learning model tailored for **Indian languages**.

India is a linguistically diverse country with **22 official languages** and multiple dialects. Emotion detection in such a multilingual environment presents challenges due to:
- Linguistic complexity  
- Code-mixing  
- Lack of labeled datasets

Our approach addresses these challenges by:
- ✅ Utilizing **BiLSTM**, which captures both forward and backward dependencies in a sequence  
- ✅ Supporting emotion classification in multiple Indian languages including **Hindi**, **Tamil**, **Telugu**, and **Bengali**  
- ✅ Leveraging **word embeddings** and **language-specific preprocessing** techniques  

## 📌 Working

The proposed model uses one of the deep learning model which is BiLSTM.  Bilstm is a type of Recurrent nural network (RNN) which processes the input text in both forward and backward direction[17]. It combines LSTM in bidirectional there by capturing both past and future contents.LSTM resembles the functionality of the human brain, as human brain  has the ability to retain the memory over a period of time. Like wise LSTM has an internal memory which can store or delete the 
information using three gates(input,forget,output).In the course of forward pass of the input text, LSTM   enumerates its hidden layers each time and updates its memory cells based on the input given and the previous hidden memory.
To process sequential data, a Bidirectional LSTM layer with 64 units have been used, enabling it to learn context from both past and future tokens. To strengthen the focus on emotion-relevant words, a custom attention mechanism  computes attention scores, normalizes them via softmax, and aggregates weighted information from the LSTM output.
