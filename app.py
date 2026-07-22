tuimport pandas as pd
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State

# -----------------------------
# Download NLTK tokenizer
# -----------------------------
nltk.download("punkt")

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("chatbot_dataset.csv")

# Remove missing values
data = data.dropna(subset=["Question", "Answer"])

data["Question"] = data["Question"].astype(str)
data["Answer"] = data["Answer"].astype(str)

# -----------------------------
# Text Preprocessing
# -----------------------------
def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    return " ".join(tokens)

data["Question"] = data["Question"].apply(preprocess)

# -----------------------------
# Split Data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data["Question"],
    data["Answer"],
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Train Model
# -----------------------------
model = make_pipeline(
    TfidfVectorizer(),
    MultinomialNB()
)

model.fit(X_train, y_train)

print("Model trained successfully!")

# -----------------------------
# Evaluate Model
# -----------------------------
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy:.2f}")

# -----------------------------
# Chatbot Function
# -----------------------------
def get_response(question):

    if question.strip() == "":
        return "Please enter a question."

    question = preprocess(question)

    return model.predict([question])[0]

# -----------------------------
# Dash App
# -----------------------------
app = dash.Dash(__name__)

app.layout = html.Div([

    html.H1(
        "AI Chatbot",
        style={"textAlign": "center"}
    ),

    dcc.Textarea(
        id="user-input",
        placeholder="Type your question here...",
        style={
            "width": "100%",
            "height": "100px"
        }
    ),

    html.Br(),

    html.Button(
        "Submit",
        id="submit-button",
        n_clicks=0
    ),

    html.Br(),
    html.Br(),

    html.Div(
        id="chatbot-output",
        style={
            "padding": "15px",
            "fontSize": "18px"
        }
    )

])

# -----------------------------
# Callback
# -----------------------------
@app.callback(
    Output("chatbot-output", "children"),
    Input("submit-button", "n_clicks"),
    State("user-input", "value")
)
def update_output(n_clicks, user_input):

    if n_clicks == 0:
        return "Ask me something!"

    response = get_response(user_input)

    return html.Div([
        html.P(f"You: {user_input}"),
        html.Hr(),
        html.P(f"Bot: {response}")
    ])

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
