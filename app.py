import pandas as pd
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State

# -----------------------------
# Download NLTK
# -----------------------------
nltk.download("punkt")

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("chatbot_dataset_500.csv")

data = data.dropna()

data["Question"] = data["Question"].astype(str)
data["Answer"] = data["Answer"].astype(str)

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)
    tokens = nltk.word_tokenize(text)
    return " ".join(tokens)

data["Question"] = data["Question"].apply(preprocess)

# -----------------------------
# Split Data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data["Question"],
    data["Answer"],
    test_size=0.1,
    random_state=42
)

# -----------------------------
# Build Model
# -----------------------------
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        ngram_range=(1,2)
    )),
    ("nb", MultinomialNB())
])

# -----------------------------
# Train
# -----------------------------
model.fit(X_train, y_train)

print("Model trained successfully!")

# -----------------------------
# Accuracy
# -----------------------------
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy : {accuracy*100:.2f}%")

# -----------------------------
# Chatbot Function
# -----------------------------
def get_response(question):

    if question is None or question.strip() == "":
        return "Please enter a question."

    question = preprocess(question)

    response = model.predict([question])[0]

    return response

# -----------------------------
# Dash App
# -----------------------------
app = dash.Dash(__name__)

app.layout = html.Div(

    style={
        "width":"75%",
        "margin":"auto",
        "fontFamily":"Arial",
        "padding":"20px"
    },

    children=[

        html.H1(
            "🌾 Crop Yield Prediction Assistant",
            style={
                "textAlign":"center",
                "color":"green"
            }
        ),

        html.H3(
            f"Model Accuracy : {accuracy*100:.2f}%",
            style={
                "textAlign":"center",
                "color":"blue"
            }
        ),

        html.Hr(),

        dcc.Textarea(

            id="user-input",

            placeholder="""
Examples:

What is crop yield?
Explain rainfall.
Tell me about irrigation.
What is loam soil?
Can farmers use this project?
            """,

            style={
                "width":"100%",
                "height":"150px",
                "fontSize":"16px",
                "padding":"10px"
            }

        ),

        html.Br(),
        html.Br(),

        html.Button(
            "Ask",
            id="submit-button",
            n_clicks=0,
            style={
                "backgroundColor":"green",
                "color":"white",
                "padding":"10px 30px",
                "fontSize":"18px",
                "border":"none",
                "cursor":"pointer"
            }
        ),

        html.Br(),
        html.Br(),

        html.Div(

            id="chatbot-output",

            style={
                "border":"2px solid green",
                "borderRadius":"10px",
                "padding":"20px",
                "backgroundColor":"#f4fff4",
                "fontSize":"18px"
            }

        )

    ]

)

# -----------------------------
# Callback
# -----------------------------
@app.callback(

    Output("chatbot-output","children"),

    Input("submit-button","n_clicks"),

    State("user-input","value")

)

def update_output(n_clicks,user_input):

    if n_clicks == 0:

        return html.Div([

            html.H3("Welcome!"),

            html.P("I can answer questions about:"),

            html.Ul([
                html.Li("Crop Yield"),
                html.Li("Rainfall"),
                html.Li("Temperature"),
                html.Li("Soil Types"),
                html.Li("Weather"),
                html.Li("Fertilizer"),
                html.Li("Irrigation"),
                html.Li("Machine Learning")
            ])

        ])

    response = get_response(user_input)

    return html.Div([

        html.H4("👤 You"),

        html.P(user_input),

        html.Hr(),

        html.H4("🌾 Assistant"),

        html.P(response)

    ])

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
