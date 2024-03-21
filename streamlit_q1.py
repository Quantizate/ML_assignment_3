import torch
import torch.nn.functional as F
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt # for making figures
from pprint import pprint
from sklearn.manifold import TSNE
import string
import streamlit as st
import numpy as np

st.title('Next Character Prediction')

class NextChar(nn.Module):
  def __init__(self, block_size, vocab_size, emb_dim, hidden_size):
    super().__init__()
    self.emb = nn.Embedding(vocab_size, emb_dim)
    self.lin1 = nn.Linear(block_size * emb_dim, hidden_size)
    self.lin2 = nn.Linear(hidden_size, vocab_size)

  def forward(self, x):
    x = self.emb(x)
    x = x.view(x.shape[0], -1)
    x = torch.sin(self.lin1(x))
    x = self.lin2(x)
    return x
  
def plot_emb(emb, itos):
    fig = plt.figure(figsize=(8, 6))

    if emb.weight.shape[1] == 1:
      for i in range(len(itos)):
        x = emb.weight[i].detach().cpu().numpy()
        plt.plot(x, np.zeros_like(x), '|', color='b', markersize=30)
        plt.text(x + 0.025, 0 + 0.025, itos[i])
      plt.title('Embedding visualization')

    elif emb.weight.shape[1] == 2:
      for i in range(len(itos)):
        x, y = emb.weight[i].detach().cpu().numpy()
        plt.scatter(x, y, color='k')
        plt.text(x + 0.025, y + 0.025, itos[i])
      plt.title('Embedding visualization')
    else:
      tsne = TSNE(n_components=2, perplexity=5,random_state=4, learning_rate=200.0)
      X_tsne = tsne.fit_transform(emb.weight.detach().cpu().numpy())

      for i in range(len(itos)):
        x, y = X_tsne[i, 0], X_tsne[i, 1]
        plt.scatter(x, y, color='k')
        plt.text(x + 5, y + 5, itos[i])
      plt.title('t-SNE visualization')
      plt.xlabel('First t-SNE')
      plt.ylabel('Second t-SNE')
    
    return fig

def predit_k(model, itos, stoi, block_size, input_str, k, toggle):
    context = [0]*block_size
    # adding some checks
    if any(c not in chars for c in input_str):
      return -1   

    sub_str = input_str[max(-block_size, -len(input_str)):]
    context[max(-block_size, -len(input_str)):] = [stoi[ch] for ch in sub_str]
    output = ''

    for i in range(k):
        x = torch.tensor(context).view(1, -1).to(device)
        y_pred = model(x)
        ix = torch.distributions.categorical.Categorical(logits=y_pred).sample().item()
        ch = itos[ix]
        if ch == '.':
          if not toggle:
            break
          else:
            output += ' '
        else:
          output += ch
        # context = torch.cat((context[1:], torch.tensor([ix], dtype=torch.long).to(device)))
        context = context[1:] + [ix]
    return output

def load_model(block_size, stoi, emb_dim, device):
    model_path = f'./trained_models/model_{block_size}_{emb_dim}.pt'
    try:
      model = NextChar(block_size, len(stoi), emb_dim, 10).to(device)
      model.load_state_dict(torch.load(model_path, map_location=device))
    except:       
      model = NextChar(block_size, len(stoi), emb_dim, 20).to(device)
      model.load_state_dict(torch.load(model_path, map_location=device))
    return model

# Load the text
with open('./data/input.txt', 'r') as file:
    text = file.read()

# Convert the text to lower case
text = text.lower()

# Create a translation table that maps every punctuation character to None
translator = str.maketrans('', '', string.punctuation.replace("'", ""))

# Remove punctuation from the text
text = text.translate(translator)

# Split the text into words
words = text.split()

# Convert the list to a DataFrame
df = pd.DataFrame(words, columns=['Words'])

# Shuffle the DataFrame
df_shuffled = df.sample(frac=1).reset_index(drop=True)
words = df_shuffled['Words'].tolist()

words = [word for word in words if 2 < len(word)]

# build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}


col1, col2 = st.columns(2)

on = st.sidebar.toggle('Predict even after seeing an empty character')

if on:
  st.sidebar.write('Feature activated!')

k = st.sidebar.slider('Select the number of characters to predict', 1, 30, 1)
st.sidebar.write('You selected:', k, 'characters to predict')

block_size = st.sidebar.slider('Select the context length', min_value=1, max_value=15, step=1)
st.sidebar.write('Your selected context length:', block_size)

emb_dim = st.sidebar.slider('Select the embedding dimension', min_value=1, max_value=8, step=1)
st.sidebar.write('Your selected embedding dimension:', emb_dim)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_input = st.text_input(
        "Enter some text to generate the next characters:",
        placeholder="Type here",
    )

if text_input:
    st.session_state.placeholder = text_input
    model = load_model(block_size=block_size, stoi=stoi, emb_dim=emb_dim, device=device)
    emb = model.emb
    pred_text = predit_k(model, itos, stoi, block_size, text_input, k, on)
    if pred_text == -1:
      st.error('Please enter a valid character', icon="🚨")
    else:
      st.write('The next ', k, ' chars are:', pred_text)
      st.write('Concatenated text:', text_input + pred_text)

st.subheader('Embedding visualization')
try:
  st.pyplot(plot_emb(emb, itos))
except:
  pass

