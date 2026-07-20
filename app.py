!pip install dash pyngrok
import pandas as pd
import nltk
data = pd.read_csv('chatbot_dataset.csv')
nltk.download('punkt')
nltk.download('punkt_tab')
data['Question'] = data['Question'].apply(lambda x: ' '.join(nltk.word_tokenize(x.lower())))
print(data.head())

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['Question'])
print(X.shape)

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data['Question'], data['Answer'], test_size=0.2, random_state = 42)
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)
print("Model training complete")

def get_response(question):
  question = ' '.join(nltk.word_tokenize(question.lower()))
  answer = model.predict([question])[0]
  return answer

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import random
from pyngrok import ngrok

app = dash.Dash(__name__)

from dash.dependencies import Input, Output


app.layout = html.Div([
   html.H1("Chatbot", style={'textAlign': 'center'}),
   dcc.Textarea(
     id='user-input',
     value='Type your question here...',
     style={'width': '100%', 'height': 100}
    ),
    html.Button('Submit', id='submit-button', n_clicks=0),
    html.Div(id='chatbot-output', style={'padding': '10px'})
])

@app.callback(
    Output('chatbot-output', 'children'),
    Input('submit-button', 'n_clicks'),
    [dash.dependencies.State('user-input', 'value')]
)
def update_output(n_clicks, user_input):
  if n_clicks > 0:
    response = get_response(user_input)
    return html.Div([
        html.P(f"You: {user_input}", style={'margin':'10px'}),
        html.P(f"Bot: {response}", style={'margin':'10px', 'backgroundColor':'#f0f0f0', 'padding':'10px'})
    ])
  return "Ask me something!"

if __name__ == '__main__':
    # Terminate any existing ngrok tunnels
    ngrok.kill()
    # Set your ngrok authtoken here. Get it from https://dashboard.ngrok.com/get-started/your-authtoken
    ngrok.set_auth_token("YOUR_AUTHTOKEN")
    # Set up ngrok tunnel
    public_url = ngrok.connect(8050)
    print(f"Dash app running on: {public_url}")
    app.run(debug=True, host='0.0.0.0', port=8050)