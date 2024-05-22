import nltk
nltk.download('punkt')
import pickle
import pandas as pd
import re
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from pywebio import start_server, output
from pywebio.input import input
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask

app = Flask(__name__)


# Load the SVM model and TF-IDF vectorizer
model_path = 'tfidf_svm.sav'
model = pickle.load(open(model_path, 'rb'))

# Initialize lists to store conversation outputs and distraction levels
conversation_outputs = []
distraction_levels = []

# Define functions to handle input and generate graphs
def handle_input(user_input):
    # Perform inference with SVM model
    predicted_label = model.predict([user_input])[0]
    # Output the user input
    output.put_text("User:", user_input)
    # Output the predicted label
    output.put_text("Chatbot:", f"The predicted emotion is: {predicted_label}")
    # Calculate distraction level (for demonstration purposes, a random value is used)
    distraction_level = np.random.uniform(0, 1)
    distraction_levels.append(distraction_level)

def generate_probability_distribution_graph():
    # Plot probability distribution bar graph
    labels = ['joy', 'sadness', 'fear', 'anger', 'surprise', 'neutral', 'disgust', 'shame', 'guilt']
    probs = np.random.rand(len(labels))  # Random probabilities for demonstration
    plt.rcParams['axes.facecolor'] = 'white'
    plt.figure(figsize=(10, 6))
    colors = ['#3D657A', '#9FB9C7', '#677881', '#7DCFFB', '#5B96A9', '#3D657B', '#FFFF00', '#800000', '#A52A2A']
    bars = plt.bar(labels, probs, color=colors)
    emoji_dict = {
        'joy': '😀',
        'sadness': '😔',
        'fear': '😨😱',
        'anger': '😠',
        'surprise': '😮',
        'neutral': '😐',
        'disgust': '😖',
        'shame': '😳',
        'guilt': '😞'
    }

    x_positions = [bar.get_x() + bar.get_width() / 2 for bar in bars]  # Center of each bar
    y_positions = [bar.get_height() + 0.1 for bar in bars]  # Slightly above the bar
    for x, y, label in zip(x_positions, y_positions, labels):
        plt.text(x, y, emoji_dict[label], ha='center', va='bottom', fontsize=20)

    plt.xlabel('Emotion')
    plt.ylabel('Probability')
    plt.title('Probability Distribution of Emotions')
    plt.grid(False)
    plt.tight_layout()

    # Save plot as image
    plt.savefig('probability_distribution_graph.png')
    plt.close()

    # Display the generated image
    output.put_image(open('probability_distribution_graph.png', 'rb').read(), width='80%')

def generate_distraction_graph():
    # Plot distraction graph
    plt.figure(figsize=(10, 6))
    turns = range(1, len(distraction_levels) + 1)
    plt.plot(turns, distraction_levels, marker='o', linestyle='-', color='r')
    plt.xlabel('Turns')
    plt.ylabel('Distraction Level')
    plt.title('Distraction Throughout Conversation')
    plt.grid(False)
    plt.tight_layout()

    # Save plot as image
    plt.savefig('distraction_graph.png')
    plt.close()

    # Display the generated image
    output.put_image(open('distraction_graph.png', 'rb').read(), width='80%')

# PyWebIO chatbot interface
def chatbot_app():
    output.put_html("<div style='background-color: white; padding: 20px;'> <h1>Emotion Depicter</h1> </div>")

    def handle_text_input():
        user_input = input("User:", placeholder="Type here...")
        conversation_outputs.append(user_input)  # Save conversation output
        handle_input(user_input)
        generate_probability_distribution_graph()

    def generate_graph():
        if not distraction_levels:
            output.put_text("No distraction levels available to generate graph.")
            return
        generate_distraction_graph()

    output.put_buttons(['Please Write', 'Generate Distraction Graph'], onclick=[handle_text_input, generate_graph])

if __name__ == "__main__":
    start_server(chatbot_app, port=8082)
