---
title: "Fine-tune Sentence Transformers Models for Science Exam Questions"
date: 2023-08-23 11:33:00 +0800
categories: [Text Analysis]
tags: [sentence embeddings, transformers]
pin: true
math: true
---

In this post we will be training a `sentence-transformers` model, for the [Kaggle - LLM Science Exam](https://www.kaggle.com/competitions/kaggle-llm-science-exam/overview) competetion. If you don't know how sentence transformers work I recommend checking out the package [documentation](https://www.sbert.net/docs/training/overview.html) and this [post](https://huggingface.co/blog/how-to-train-sentence-transformers#how-sentence-transformers-models-work) from HuggingFace. We will be using the two approaches available in the library: bi-encoder and cross-encoder. The first computes sentence embeddings of two sentences separately, then we can measure how similar these embeddings are by some function like dot product or cosine, the latter takes both sentence at once an outputs a value indicating how similar they are.

The dataset was generated using gpt3.5 by asking it to generate a series of multiple choice questions, with a known answer, from scientific wikipedia articles. We will be focusing more on how to train models from the sentence-transformer package and evaluating them rather than the competetion itself.

## Dowload and transform data
The first step is to retrieve the data from kaggle into our colab environment, for that we will have to use the kaggle api, there's a really good article on how to download kaggle datasets [here](https://saturncloud.io/blog/how-to-use-kaggle-datasets-in-google-colab/).


```python
# Name of the competetion
competition_name = "kaggle-llm-science-exam"

# Mount your Google Drive.
from google.colab import drive
drive.mount("/content/drive")

# Download the token from your account and store it in your drive
kaggle_creds_path = "drive/MyDrive/text_analysis/kaggle.json"

# Install kaggle api
! pip install kaggle --quiet

# Create a kaggle folder to root (first remove if one already exists)
# This is were the token will go
! rm -r ~/.kaggle
! mkdir ~/.kaggle
# Copy the token into the kaggle folder
! cp {kaggle_creds_path} ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
# Download the datasets
! kaggle competitions download -c {competition_name}
# Create a folder to store the data
! mkdir kaggle_data
# Unzip the data into the folder
! unzip {competition_name + ".zip"} -d kaggle_data

# Unmount your Google Drive
drive.flush_and_unmount()

```

    Mounted at /content/drive
    Downloading kaggle-llm-science-exam.zip to /content
    100% 72.5k/72.5k [00:00<00:00, 342kB/s]
    100% 72.5k/72.5k [00:00<00:00, 342kB/s]
    mkdir: cannot create directory ‘kaggle_data’: File exists
    Archive:  kaggle-llm-science-exam.zip
      inflating: kaggle_data/sample_submission.csv  
      inflating: kaggle_data/test.csv    
      inflating: kaggle_data/train.csv   


Once the data is downloaded we have to separate it into train, development and test sets. On the first we will fit the model, the second is used for fine-tuning and the third for the final evaluation. Given that only the `train.csv` file is labaled, we will only split this one.


```python
import pandas as pd
train_full = pd.read_csv("kaggle_data/train.csv")
train_full.head()
```





  <div id="df-35b7ba2b-d01a-4924-a6d2-fe34af0c02cf" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>prompt</th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>D</th>
      <th>E</th>
      <th>answer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Which of the following statements accurately d...</td>
      <td>MOND is a theory that reduces the observed mis...</td>
      <td>MOND is a theory that increases the discrepanc...</td>
      <td>MOND is a theory that explains the missing bar...</td>
      <td>MOND is a theory that reduces the discrepancy ...</td>
      <td>MOND is a theory that eliminates the observed ...</td>
      <td>D</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Which of the following is an accurate definiti...</td>
      <td>Dynamic scaling refers to the evolution of sel...</td>
      <td>Dynamic scaling refers to the non-evolution of...</td>
      <td>Dynamic scaling refers to the evolution of sel...</td>
      <td>Dynamic scaling refers to the non-evolution of...</td>
      <td>Dynamic scaling refers to the evolution of sel...</td>
      <td>A</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Which of the following statements accurately d...</td>
      <td>The triskeles symbol was reconstructed as a fe...</td>
      <td>The triskeles symbol is a representation of th...</td>
      <td>The triskeles symbol is a representation of a ...</td>
      <td>The triskeles symbol represents three interloc...</td>
      <td>The triskeles symbol is a representation of th...</td>
      <td>A</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>What is the significance of regularization in ...</td>
      <td>Regularizing the mass-energy of an electron wi...</td>
      <td>Regularizing the mass-energy of an electron wi...</td>
      <td>Regularizing the mass-energy of an electron wi...</td>
      <td>Regularizing the mass-energy of an electron wi...</td>
      <td>Regularizing the mass-energy of an electron wi...</td>
      <td>C</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>Which of the following statements accurately d...</td>
      <td>The angular spacing of features in the diffrac...</td>
      <td>The angular spacing of features in the diffrac...</td>
      <td>The angular spacing of features in the diffrac...</td>
      <td>The angular spacing of features in the diffrac...</td>
      <td>The angular spacing of features in the diffrac...</td>
      <td>D</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-35b7ba2b-d01a-4924-a6d2-fe34af0c02cf')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-35b7ba2b-d01a-4924-a6d2-fe34af0c02cf button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-35b7ba2b-d01a-4924-a6d2-fe34af0c02cf');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-2441ad3d-372a-4ab0-bc86-760dd95c62be">
  <button class="colab-df-quickchart" onclick="quickchart('df-2441ad3d-372a-4ab0-bc86-760dd95c62be')"
            title="Suggest charts."
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
    background-color: #E8F0FE;
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: #1967D2;
    height: 32px;
    padding: 0 0 0 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: #E2EBFA;
    box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: #174EA6;
  }

  [theme=dark] .colab-df-quickchart {
    background-color: #3B4455;
    fill: #D2E3FC;
  }

  [theme=dark] .colab-df-quickchart:hover {
    background-color: #434B5C;
    box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
    filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
    fill: #FFFFFF;
  }
</style>

  <script>
    async function quickchart(key) {
      const charts = await google.colab.kernel.invokeFunction(
          'suggestCharts', [key], {});
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-2441ad3d-372a-4ab0-bc86-760dd95c62be button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>




First we select 30% of the indices for testing and development, then we split those into two, one for each task. The remaining 70% will be used for training.


```python
import numpy as np
# Select a seed
random_state = np.random.RandomState(123)

# Fraction of data for the first split (e.g., 0.7 for 70%)
fraction_first_split = 0.3

# Sample rows for the first split
num_samples = int(len(train_full) * fraction_first_split)
test_indices = random_state.choice(train_full.index, num_samples, replace=False)

# Drop the test_indices and use the rest for training
train_set = train_full.drop(test_indices)

# Divide the test_indices in two equal parts
num_samples_dev = int(len(test_indices) * 0.5)
dev_indices = random_state.choice(test_indices, num_samples_dev, replace=False)

# Split the dataset
dev_set = train_full.loc[dev_indices]
test_set = train_full.loc[test_indices].drop(dev_indices)
```

As we can see above, the dataset is on a wide format, this means that the question and all the options are on the same row, with a column indicating which is the correct one. If we want the model to read the data correctly we need to have the question or prompt and the option in every row, with an additional column indicating if it is the correct answer: `[prompt, sentence, label]`.


```python
def pivot_long(df):
  df_long = df.melt(id_vars=['prompt', 'id'],
            value_vars=['A', 'B', 'C', 'D', 'E'],
            value_name="option"). \
            merge(df[["id", "answer"]], how='left', on='id')
  df_long["label"] = (df_long.variable == df_long.answer).astype(int)
  return df_long

train_long = pivot_long(train_set)
dev_long = pivot_long(dev_set)
test_long = pivot_long(test_set)
train_long.head()
```





  <div id="df-e90eef9e-432f-4194-9ba5-4c61f4734a5f" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prompt</th>
      <th>id</th>
      <th>variable</th>
      <th>option</th>
      <th>answer</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Which of the following statements accurately d...</td>
      <td>0</td>
      <td>A</td>
      <td>MOND is a theory that reduces the observed mis...</td>
      <td>D</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Which of the following is an accurate definiti...</td>
      <td>1</td>
      <td>A</td>
      <td>Dynamic scaling refers to the evolution of sel...</td>
      <td>A</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Which of the following statements accurately d...</td>
      <td>2</td>
      <td>A</td>
      <td>The triskeles symbol was reconstructed as a fe...</td>
      <td>A</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>What is the significance of regularization in ...</td>
      <td>3</td>
      <td>A</td>
      <td>Regularizing the mass-energy of an electron wi...</td>
      <td>C</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Which of the following statements accurately d...</td>
      <td>5</td>
      <td>A</td>
      <td>Gauss's law holds only for situations involvin...</td>
      <td>B</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-e90eef9e-432f-4194-9ba5-4c61f4734a5f')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-e90eef9e-432f-4194-9ba5-4c61f4734a5f button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-e90eef9e-432f-4194-9ba5-4c61f4734a5f');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-e8fb4cfb-6192-431d-975d-204962c22999">
  <button class="colab-df-quickchart" onclick="quickchart('df-e8fb4cfb-6192-431d-975d-204962c22999')"
            title="Suggest charts."
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
    background-color: #E8F0FE;
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: #1967D2;
    height: 32px;
    padding: 0 0 0 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: #E2EBFA;
    box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: #174EA6;
  }

  [theme=dark] .colab-df-quickchart {
    background-color: #3B4455;
    fill: #D2E3FC;
  }

  [theme=dark] .colab-df-quickchart:hover {
    background-color: #434B5C;
    box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
    filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
    fill: #FFFFFF;
  }
</style>

  <script>
    async function quickchart(key) {
      const charts = await google.colab.kernel.invokeFunction(
          'suggestCharts', [key], {});
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-e8fb4cfb-6192-431d-975d-204962c22999 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>




## Models
As mentioned before, we will be using the sentence-transformers package, if you don't have it installed you can run the following code.


```python
! pip install sentence-transformers --quiet
```

Next we import the `SentenceTransformer` and `CrossEncoder` functions. And initialize the models. For our Cross encoder we will be using ['distilroberta-base'](https://huggingface.co/distilroberta-base), and for our bi-encoder we will use ['multi-qa-mpnet-base-cos-v1'](https://huggingface.co/sentence-transformers/multi-qa-mpnet-base-cos-v1), which has been pre-train in question, answer pairs from different sites like StackExchange, Yahoo Answers, Google & Bing search queries.


```python
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.cross_encoder import CrossEncoder

# Set batch size and epochs
num_epochs = 4
batch_size = 16

# Initialize model
model_name = 'multi-qa-mpnet-base-cos-v1'
model = SentenceTransformer(model_name)

# Initialize cross-encoder, we set num_labels to 1 since we're dealing with a binary classification task.
ce_name = 'distilroberta-base'
cross_encoder = CrossEncoder(ce_name, num_labels=1)
```

    Some weights of RobertaForSequenceClassification were not initialized from the model checkpoint at distilroberta-base and are newly initialized: ['classifier.dense.weight', 'classifier.out_proj.weight', 'classifier.dense.bias', 'classifier.out_proj.bias']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.


Now we want to check how the models perform before "seeing" any of our labeled data. For that we will make predict functions, bare in mind that the bi-encoder doesn't classify a pair of sentences on its own. The cross encoder, on the other hand, takes a (prompt, sentence) pair and directly outputs a score on how likely this sentence is the actual answer to the question.


```python
def predict(data: pd.DataFrame, model):
  # Predict using the bi-encoder model
  options = ['A', 'B', 'C', 'D', 'E']
  predictions = []
  for i, row in data.iterrows():
    # Compute the prompt embeddings
    query_embed = model.encode(row.prompt, convert_to_tensor=True).cuda()
    # compute the embeddings for each option
    corpus_embed = model.encode(row[options],  convert_to_tensor=True).cuda()
    # Perform semantic search using cosine similarity as metric
    hits = util.semantic_search(query_embed, corpus_embed, top_k=3, score_function=util.cos_sim)[0]

    predictions.append({'prompt': row.prompt,
                        'pred_ans': [options[hit['corpus_id']] for hit in hits],
                        'answer': row.answer})
  return pd.DataFrame(predictions)

def predict_ce(data, model):
  options = ['A', 'B', 'C', 'D', 'E']
  predictions = []
  for _, row in data.iterrows():
    # Build the input format [(prompt, A), (prompt, B), ...]
    cross_inp = [[row.prompt, row[option]] for option in options]
    # Calculate the scores
    cross_scores = model.predict(cross_inp)
    # Sort the options by their scores
    order_options = [options[i] for i in np.argsort(cross_scores)[::-1]]

    predictions.append({'prompt': row.prompt,
                        'pred_ans': order_options,
                        'answer': row.answer})
  return pd.DataFrame(predictions)

bi_encoder_init_pred = predict(test_set, model)
cross_init_pred = predict_ce(test_set, cross_encoder)
bi_encoder_init_pred.head()
```





  <div id="df-12ceacd4-94f1-497f-acc8-512a16c4e758" class="colab-df-container">
    <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prompt</th>
      <th>pred_ans</th>
      <th>answer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>What is the De Haas-Van Alphen effect?</td>
      <td>[C, D, A]</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1</th>
      <td>What is the reason behind the adoption of a lo...</td>
      <td>[A, D, C]</td>
      <td>A</td>
    </tr>
    <tr>
      <th>2</th>
      <td>What is the role of IL-10 in the formation of ...</td>
      <td>[C, A, E]</td>
      <td>B</td>
    </tr>
    <tr>
      <th>3</th>
      <td>What is the Landau-Lifshitz-Gilbert equation u...</td>
      <td>[B, A, C]</td>
      <td>C</td>
    </tr>
    <tr>
      <th>4</th>
      <td>What is the cause of the observed change in th...</td>
      <td>[E, A, C]</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>
    <div class="colab-df-buttons">

  <div class="colab-df-container">
    <button class="colab-df-convert" onclick="convertToInteractive('df-12ceacd4-94f1-497f-acc8-512a16c4e758')"
            title="Convert this dataframe to an interactive table."
            style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px" viewBox="0 -960 960 960">
    <path d="M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z"/>
  </svg>
    </button>

  <style>
    .colab-df-container {
      display:flex;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    .colab-df-buttons div {
      margin-bottom: 4px;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

    <script>
      const buttonEl =
        document.querySelector('#df-12ceacd4-94f1-497f-acc8-512a16c4e758 button.colab-df-convert');
      buttonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';

      async function convertToInteractive(key) {
        const element = document.querySelector('#df-12ceacd4-94f1-497f-acc8-512a16c4e758');
        const dataTable =
          await google.colab.kernel.invokeFunction('convertToInteractive',
                                                    [key], {});
        if (!dataTable) return;

        const docLinkHtml = 'Like what you see? Visit the ' +
          '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
          + ' to learn more about interactive tables.';
        element.innerHTML = '';
        dataTable['output_type'] = 'display_data';
        await google.colab.output.renderOutput(dataTable, element);
        const docLink = document.createElement('div');
        docLink.innerHTML = docLinkHtml;
        element.appendChild(docLink);
      }
    </script>
  </div>


<div id="df-e2db1a8c-e35a-417e-8068-72903f310602">
  <button class="colab-df-quickchart" onclick="quickchart('df-e2db1a8c-e35a-417e-8068-72903f310602')"
            title="Suggest charts."
            style="display:none;">

<svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
     width="24px">
    <g>
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
    </g>
</svg>
  </button>

<style>
  .colab-df-quickchart {
    background-color: #E8F0FE;
    border: none;
    border-radius: 50%;
    cursor: pointer;
    display: none;
    fill: #1967D2;
    height: 32px;
    padding: 0 0 0 0;
    width: 32px;
  }

  .colab-df-quickchart:hover {
    background-color: #E2EBFA;
    box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
    fill: #174EA6;
  }

  [theme=dark] .colab-df-quickchart {
    background-color: #3B4455;
    fill: #D2E3FC;
  }

  [theme=dark] .colab-df-quickchart:hover {
    background-color: #434B5C;
    box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
    filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
    fill: #FFFFFF;
  }
</style>

  <script>
    async function quickchart(key) {
      const charts = await google.colab.kernel.invokeFunction(
          'suggestCharts', [key], {});
    }
    (() => {
      let quickchartButtonEl =
        document.querySelector('#df-e2db1a8c-e35a-417e-8068-72903f310602 button');
      quickchartButtonEl.style.display =
        google.colab.kernel.accessAllowed ? 'block' : 'none';
    })();
  </script>
</div>
    </div>
  </div>




We can see that we are predicting 3 possible answers, this is because of the evaluation metric that we'll be using.

### Evaluation
The competetion uses the Mean Average Precision @3 (MAP@3) to evaluate submissions:

$$
MAP@3 = \frac{1}{U}\sum_{u=1}^U \sum_{k=1}^{\min(n,3)}P(k)\times rel(k)
$$

Where $U$ is the number of questions in the test set, $P(k)$ is the precision at cutoff $k$, $n$ is the number of predictors per question and $rel(k)$ is an indicator function equaling 1 if the item at rank $k$ is a correct label, zero otherwise.

I built a python function to compute this metric as I understood it.


```python
def precision(pred_labels, true_label):
    true_positives = sum(pred_labels == true_label)
    false_positives = sum(pred_labels != true_label)
    return true_positives/(true_positives + false_positives)

def mean_avg_precision(pred_labels, true_labels, max_labels:int=3):
    """
    pred_labels: a numpy array of size u containing the predicted labels.
    true_labels: a numpy array of size u containing the true labels.
    max_labels: maximum number of predicted labels allowed per question.
    u: number of questions.
    """
    u = len(pred_labels)
    # Initialize precision array
    precisions = np.empty(u)
    for i, true_label in enumerate(true_labels):
        # Ge the top 3 perdictors
        predictions = np.array(pred_labels[i][:max_labels])

        # Indicator function
        rel = (predictions == true_label)
        # Extract the first True in the function if there are any
        # Otherwise retrieve an array of False
        rel = rel[:np.argmax(rel)+1] if any(rel) else rel
        rel = rel.astype(int)

        # Compute the precision
        p = np.empty(len(rel))
        for k in range(len(rel)):
            p[k] = precision(predictions[:k+1], true_label)

        # Multiply the indicator and precision arrays
        precisions[i] = np.sum(rel * p)

    return np.mean(precisions)

init_metric_bi = mean_avg_precision(bi_encoder_init_pred.pred_ans, bi_encoder_init_pred.answer)
init_metric_cross = mean_avg_precision(cross_init_pred.pred_ans, cross_init_pred.answer)
print(f'MAP@3 of bi-encoder before training: {init_metric_bi}')
print(f'MAP@3 of cross-encoder before training: {init_metric_cross}')
```

    MAP@3 of bi-encoder before training: 0.41111111111111115
    MAP@3 of cross-encoder before training: 0.43333333333333335


### More data transformations
The models won't read the input of a pandas DataFrame directly, therefore we have to transform the data into the required format. This done by using the `InputExample` and `DataLoader` functions.


```python
from sentence_transformers import InputExample
from torch.utils.data import DataLoader

# Function to format data
def format_dataloader(df, batch_size = batch_size):
  examples = []
  for i, row in df.iterrows():
    examples.append(InputExample(texts = [row.prompt, row.option], label = row.label))
  return examples

train_samples = format_dataloader(train_long)
dev_samples = format_dataloader(dev_long)
test_samples = format_dataloader(test_long)

# Create the dataloader to feed to the model
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)
```

## Train
To train the models we must specify a loss function, given that we are dealing with a Binary classification task we will use the [ContrastiveLoss](https://www.sbert.net/docs/package_reference/losses.html#contrastiveloss), which, as described by the sentence-transformers documentation, *expects as input two texts and a label of either 0 or 1. If the label == 1, then the distance between the two embeddings is reduced. If the label == 0, then the distance between the embeddings is increased.*

We will also need an evaluator to fine tune the model using the dev set, given our task we can use the `BinaryClassificationEvaluator` for the bi-encoder and `CEBinaryClassificationEvaluator` for the cross encoder.


```python
from sentence_transformers import losses
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator

train_loss = losses.ContrastiveLoss(model=model)
evaluator_bi = BinaryClassificationEvaluator.from_input_examples(dev_samples, name="llm-exam-dev")
evaluator_ce = CEBinaryClassificationEvaluator.from_input_examples(dev_samples)
```

### Training bi-encoder
Next we use the fit function to train the models, but first we must establish a directory to store the fitted models.


```python
bi_encoder_save_path = "/content/output/" + model_name
warmup_steps = int(len(train_dataloader) * num_epochs * 0.1)
model.fit([(train_dataloader, train_loss)],
          show_progress_bar=True,
          epochs=num_epochs,
          evaluator=evaluator_bi,
          warmup_steps=warmup_steps,
          output_path=bi_encoder_save_path)
```


    Epoch:   0%|          | 0/4 [00:00<?, ?it/s]



    Iteration:   0%|          | 0/44 [00:00<?, ?it/s]



    Iteration:   0%|          | 0/44 [00:00<?, ?it/s]



    Iteration:   0%|          | 0/44 [00:00<?, ?it/s]



    Iteration:   0%|          | 0/44 [00:00<?, ?it/s]


### Training cross-encoder


```python
cross_encoder_save_path = "/content/output/" + ce_name
cross_encoder.fit(train_dataloader=train_dataloader,
                  show_progress_bar=True,
                  epochs=num_epochs,
                  evaluator=evaluator_ce,
                  warmup_steps=warmup_steps,
                  evaluation_steps=10000,
                  output_path=cross_encoder_save_path)
```


    Epoch:   0%|          | 0/4 [00:00<?, ?it/s]



    Iteration:   0%|          | 0/44 [00:00<?, ?it/s]



    Iteration:   0%|          | 0/44 [00:00<?, ?it/s]



    Iteration:   0%|          | 0/44 [00:00<?, ?it/s]



    Iteration:   0%|          | 0/44 [00:00<?, ?it/s]


### Evaluate bi-encoder
Finally we evaluate the fitted models, for that we first have to load them from the directory where they were stored.


```python
bi_encoder_fit = SentenceTransformer(bi_encoder_save_path)
bi_test_predictions = predict(test_set, bi_encoder_fit)
bi_test_metric = mean_avg_precision(bi_test_predictions.pred_ans, bi_test_predictions.answer)
print(f'MAP@3 of trained bi-encoder: %.3f'%bi_test_metric)
```

    MAP@3 of trained bi-encoder: 0.478


There's a minor improvment from the base model. We could also use our evaluator function directly.


```python
evaluator = BinaryClassificationEvaluator.from_input_examples(test_samples, name="llm-exam-test")
evaluator(bi_encoder_fit)
```




    0.24911479845579265



### Evaluate cross-encoder


```python
ce_fit = CrossEncoder(cross_encoder_save_path)
test_pred_ce = predict_ce(test_set, ce_fit)
cross_test_metric = mean_avg_precision(test_pred_ce.pred_ans, test_pred_ce.answer)
print(f'MAP@3 of trained cross-encoder: %.3f'%cross_test_metric)
```

    MAP@3 of trained cross-encoder: 0.511



```python
evaluator = CEBinaryClassificationEvaluator.from_input_examples(test_samples, name='llm-exam-test')
evaluator(ce_fit)
```




    0.22773457900968527



## Conclusion
The fine-tuned sentence transformer models performed a little better over the base models, we could marginally improve the performance by increasing the number of epochs, but our main constrains are dataset and model sizes. Some users in the kaggle competetion have increase the size of the dataset by generating more questions with gpt3.5, which yields better performance. On a later post we will be looking at how to use the sentence-transformers package for data augmentation.
