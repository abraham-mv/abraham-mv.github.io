---
title: "Topic Modelling (Part 1): Introduction to text embeddings"
date: 2023-08-13 11:33:00 +0800
categories: [Text Analysis]
tags: [word embeddings]
pin: true
math: true
---

Topic modelling is an unsupervised document classification task, in which we assign each document or phrase in a corpus with a label, depending on the abstract properties of each corpus entry. So the difficulty comes from trying to find these properties from the text. 

In regular classification tasks we would have a set of features and an output variable, for text these features could be: number of words in the document, number of times a specific word is mentioned, even the number of positive and negative words if we use a sentiment lexicon. Then we would use some classification model, like logistic regression, and predict how likely is a document to have a certain label. However, in real world scenarios this isn't very useful, it would be a monumental task to do extract the most relevant features from a corpus of thousands of text with thousands of words each. So, what if instead of having to engineer features from the text we let the text be the features; meaning, use the words in the documents to compare them. But we can't input a sequence of characters into a mathematical formula, this is when vector semantics appear. The idea is to represent the meaning of word according to the context it appears in.

In this post we will take a look at the simplest methods to represent text as numerical vectors using the 20newsgroups dataset. We will import the dataset from the sklearn package and use string and pandas to do some manipulation. Firstly we choose 4 random categories for simplicity and put the dataset into a dataframe.


```python
import string
import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.datasets import fetch_20newsgroups
newsgroups = fetch_20newsgroups(subset = 'train', categories=['comp.graphics', 'rec.autos', 
                                                              'sci.space', 'talk.politics.misc'])
categories = [newsgroups["target_names"][i] for i in newsgroups["target"]]
newsgroups_df = pd.DataFrame({"data": newsgroups["data"],
                             "target": newsgroups["target"],
                             "category": categories})
```
We will be using the news categories as documents; therefore, we can use the `groupby` and `join` functions from pandas to join all the documents in one category together.

```python
def join_strings(series):
    return ' '.join(series)
news_by_category = newsgroups_df.groupby("category")['data'].agg(join_strings).reset_index()
```

## Term-document representations
The simplest approach is to have a term-document matrix, where every column is the name of a document and every row is a term from a vocabulary made out of the whole corpus, and every cell is the number of times that term appears in that document. First step to build a term-document matrix is to separate the documents into individual words, this process is called *tokenization*. Here we convert the whole text to lower case, and we split it by word (it splits every time it detects a space), then we remove the punctuation characters.


```python
def contains_number(s):
    return bool(re.search(r'\d', s))

news_by_category["tokens"] = None
for i, text in enumerate(news_by_category.data):
    tokenized_cat = []
    for token in text.lower().split():
        token = token.strip(string.punctuation)
        if len(token) > 0 and not contains_number(token):
            tokenized_cat.append(token)
    news_by_category["tokens"][i] = tokenized_cat 
```

Next we want to count the number of words per category.


```python
news_tokenized = news_by_category.drop('data', axis=1). \
    explode('tokens').  \
    groupby(['category', 'tokens']).size(). \
    reset_index(name='count')

news_tokenized.head()
```




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
      <th>category</th>
      <th>tokens</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>comp.graphics</td>
      <td>a</td>
      <td>2628</td>
    </tr>
    <tr>
      <th>1</th>
      <td>comp.graphics</td>
      <td>a&amp;m</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>comp.graphics</td>
      <td>a)bort</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>comp.graphics</td>
      <td>a,b,c</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>comp.graphics</td>
      <td>a-b</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Now we are ready to build the matrix, we will choose a few specific words for visualization purposes.


```python
news_tokenized[news_tokenized.tokens.isin(["computer", "space", "cars", "from", "republican"])]. \
    pivot(index='tokens', columns='category', values='count'). \
    fillna(0).astype(int)
```




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
      <th>category</th>
      <th>comp.graphics</th>
      <th>rec.autos</th>
      <th>sci.space</th>
      <th>talk.politics.misc</th>
    </tr>
    <tr>
      <th>tokens</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>cars</th>
      <td>2</td>
      <td>365</td>
      <td>2</td>
      <td>8</td>
    </tr>
    <tr>
      <th>computer</th>
      <td>205</td>
      <td>68</td>
      <td>86</td>
      <td>40</td>
    </tr>
    <tr>
      <th>from</th>
      <td>1177</td>
      <td>945</td>
      <td>1280</td>
      <td>1146</td>
    </tr>
    <tr>
      <th>republican</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>25</td>
    </tr>
    <tr>
      <th>space</th>
      <td>47</td>
      <td>15</td>
      <td>1194</td>
      <td>19</td>
    </tr>
  </tbody>
</table>
</div>



In the following we have a simple two-dimensional vectror representation of the documents


```python
# Create a figure and axis
fig, ax = plt.subplots()

# Create a quiver plot for the vectors
ax.quiver(0, 0, 205, 2, angles='xy', scale_units='xy', scale=1, color='blue', label='comp.graphics')
ax.quiver(0, 0, 40, 8, angles='xy', scale_units='xy', scale=1, color='red', label='talk.politics.misc')
ax.quiver(0, 0, 86, 2, angles='xy', scale_units='xy', scale=1, color='green', label='sci.space')
ax.quiver(0, 0, 68, 365, angles='xy', scale_units='xy', scale=1, color='orange', label='rec.autos')

# Set x and y axis limits
ax.set_xlim(-1, 250)
ax.set_ylim(-1, 150)

# Add labels and legend
ax.set_xlabel('computer')
ax.set_ylabel('cars')
ax.set_title('Term-document vector representations')
ax.legend()

# Show the plot
plt.show()
```


    
![png](/img/word-embeddings/output_12_0.png)
    


## Term-term representations
Another way of achieving this is with the term-term co-occurrence matrices, this a matrix of $|V|\times|V|$, where $|V|$ is the vocabulary size, made up of context words in the columns and target words in the rows. Each entry corresponds to the number of times those terms appear together in the document. 

We will build a term-term matrix for only one category for simplicity. First we need to create a vocabulary, we will remove stop words like *the*, *from*, *there*, using a corpus from `nltk`.
```python
from nltk.corpus import stopwords
import re

stop_words = set(stopwords.words('english'))
tokens = [word for word in news_by_category.iloc[0,:]['tokens'] if word not in stop_words and not contains_number(word)]
counts = pd.Series(tokens).value_counts()

vocabulary = list(set(tokens))
term_to_index = {term: idx for idx, term in enumerate(vocabulary)}
```

Next we build the matrix in an iterative fashion, here if the context and target words are the same it store the number of times the word appears in the text.

```python
window = 2
numpy_matrix = np.zeros((len(vocabulary), len(vocabulary)), dtype=int)
for i in range(len(tokens)):
    for j in range(i-window, i+window+1):
        if 0 <= j < len(tokens):
            if i == j:  
                numpy_matrix[term_to_index[tokens[i]],term_to_index[tokens[j]]] = counts[tokens[i]]
            else:
                numpy_matrix[term_to_index[tokens[i]],term_to_index[tokens[j]]] += 1

```

Now we can convert the numpy matrix into a pandas dataframe.

```python
co_occurrences = pd.DataFrame(numpy_matrix, columns=vocabulary, index=vocabulary)
co_occurrences.loc[["computer", "lines", "graphics", "spline", "image"], 
                   ["computer", "lines", "graphics", "spline", "image"]]
```




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
      <th>computer</th>
      <th>lines</th>
      <th>graphics</th>
      <th>spline</th>
      <th>image</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>computer</th>
      <td>205</td>
      <td>13</td>
      <td>57</td>
      <td>0</td>
      <td>7</td>
    </tr>
    <tr>
      <th>lines</th>
      <td>13</td>
      <td>640</td>
      <td>9</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>graphics</th>
      <td>57</td>
      <td>9</td>
      <td>437</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>spline</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>15</td>
      <td>0</td>
    </tr>
    <tr>
      <th>image</th>
      <td>7</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>514</td>
    </tr>
  </tbody>
</table>
</div>



Next is another simple visualization of the word vectors.


```python
# Define the x and y axis (context words)
x = co_occurrences["computer"]
y = co_occurrences["image"]

# Create a figure and axis
fig, ax = plt.subplots()

# Create a quiver plot for the vectors
ax.quiver([0,0], [0,0], 
          x[["lines", "graphics"]], y[["lines", "graphics"]], 
          angles='xy', scale_units='xy', scale=1, color=['blue','red'])

# Set x and y axis limits
ax.set_xlim(0, 73)
ax.set_ylim(0, 6)

# Add labels and legend
ax.set_xlabel('computer')
ax.set_ylabel('image')
ax.set_title('Term-term vector representations')

ax.annotate("lines[{},{}]".format(x.lines, y.lines),(x["lines"],y["lines"]))
ax.annotate("graphics[{},{}]".format(x.graphics, y.graphics),(x["graphics"],y["graphics"]))
# Show the plot
plt.show()
```


    
![png](/img/word-embeddings/output_19_0.png)
    


## TF-IDF
Both methods that we've explored so far used raw frequencies to represent words, in practice these aren't very useful because the most frequent words will always be *the*, *for*, *a*, etc. So what we want to extract are those words that appear very often in a specific document but not too often in all documents. The **tf-idf** weighting, where **tf** stands for *term frequency* and **idf** for *inverse document frequency*,  attempts to do just that. As the name suggest **tf** is just the number of times a word *t* appears in document *d*, while **idf** is the inverse proportion of documents that contain word *t*, both functions are often squash by a base 10 logarithm.


$$
\begin{aligned}
\mathrm{tf}_{t,d} &= \log_{10}(\mathrm{count}(t,d)+1) \\
\mathrm{idf}_t &= \log_{10}\left(\frac{N}{\mathrm{df}_t}\right)
\end{aligned}
$$


Where $N$ is the number of documents in the corpus and $$\text{df}_t$$ is the number of documents containing term $$t$$, we add one to the argument of the logarithm in case we have a count of zero. Then the **tf-idf** weights $$w_{t,d}$$ is the product of the two:

$$
w_{t,d} = \mathrm{tf}_{t,d} \times \mathrm{idf}_t
$$

Since we already counted the tokens per category when we built the term-document matrix, we will compute the inverse document frequency.


```python
tokens_idf = news_tokenized.groupby('tokens').size().reset_index(name='idf')
tokens_idf["idf"] = np.log10(4/tokens_idf['idf'])
```

For the next step we will merge this dataframe with the previous one and multiply the log of the counts and the idf.


```python
news_tfidf = news_tokenized.merge(tokens_idf, how='left', on='tokens')
news_tfidf["tf-idf"] = np.log10(news_tfidf['count']+1)*news_tfidf['idf']
news_tfidf.head()
```




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
      <th>category</th>
      <th>tokens</th>
      <th>count</th>
      <th>idf</th>
      <th>tf-idf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>comp.graphics</td>
      <td>a</td>
      <td>2628</td>
      <td>0.00000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>comp.graphics</td>
      <td>a&amp;m</td>
      <td>6</td>
      <td>0.30103</td>
      <td>0.254400</td>
    </tr>
    <tr>
      <th>2</th>
      <td>comp.graphics</td>
      <td>a)bort</td>
      <td>4</td>
      <td>0.60206</td>
      <td>0.420822</td>
    </tr>
    <tr>
      <th>3</th>
      <td>comp.graphics</td>
      <td>a,b,c</td>
      <td>2</td>
      <td>0.60206</td>
      <td>0.287256</td>
    </tr>
    <tr>
      <th>4</th>
      <td>comp.graphics</td>
      <td>a-b</td>
      <td>1</td>
      <td>0.60206</td>
      <td>0.181238</td>
    </tr>
  </tbody>
</table>
</div>



Next we just have to pivot wider.


```python
news_tfidf[news_tfidf['tokens'].isin(["computer", "cars", "from", "republican", "spline", "ferrari"])]. \
    pivot(index='tokens', columns='category', values='tf-idf').fillna(0)
```




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
      <th>category</th>
      <th>comp.graphics</th>
      <th>rec.autos</th>
      <th>sci.space</th>
      <th>talk.politics.misc</th>
    </tr>
    <tr>
      <th>tokens</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>cars</th>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>computer</th>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>ferrari</th>
      <td>0.000000</td>
      <td>0.5088</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>from</th>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>republican</th>
      <td>0.000000</td>
      <td>0.0000</td>
      <td>0.090619</td>
      <td>0.425949</td>
    </tr>
    <tr>
      <th>spline</th>
      <td>0.724952</td>
      <td>0.0000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



As we can see many of the words have values of zero, this is because they appear in every category so the tf-idf function is flagging them as "not significant". In the next dataframe we are retrieving the 3 words with the highest tf-idf per category.


```python
news_tfidf.groupby('category', group_keys=False).apply(lambda group: group.nlargest(3, 'tf-idf'))
```




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
      <th>category</th>
      <th>tokens</th>
      <th>count</th>
      <th>idf</th>
      <th>tf-idf</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8395</th>
      <td>comp.graphics</td>
      <td>polygon</td>
      <td>111</td>
      <td>0.60206</td>
      <td>1.233752</td>
    </tr>
    <tr>
      <th>11232</th>
      <td>comp.graphics</td>
      <td>tiff</td>
      <td>98</td>
      <td>0.60206</td>
      <td>1.201492</td>
    </tr>
    <tr>
      <th>11887</th>
      <td>comp.graphics</td>
      <td>vga</td>
      <td>81</td>
      <td>0.60206</td>
      <td>1.152231</td>
    </tr>
    <tr>
      <th>16968</th>
      <td>rec.autos</td>
      <td>honda</td>
      <td>71</td>
      <td>0.60206</td>
      <td>1.118226</td>
    </tr>
    <tr>
      <th>13245</th>
      <td>rec.autos</td>
      <td>automotive</td>
      <td>70</td>
      <td>0.60206</td>
      <td>1.114569</td>
    </tr>
    <tr>
      <th>22423</th>
      <td>rec.autos</td>
      <td>toyota</td>
      <td>68</td>
      <td>0.60206</td>
      <td>1.107097</td>
    </tr>
    <tr>
      <th>31089</th>
      <td>sci.space</td>
      <td>lunar</td>
      <td>224</td>
      <td>0.60206</td>
      <td>1.416155</td>
    </tr>
    <tr>
      <th>35630</th>
      <td>sci.space</td>
      <td>spacecraft</td>
      <td>160</td>
      <td>0.60206</td>
      <td>1.328642</td>
    </tr>
    <tr>
      <th>32640</th>
      <td>sci.space</td>
      <td>orbital</td>
      <td>97</td>
      <td>0.60206</td>
      <td>1.198838</td>
    </tr>
    <tr>
      <th>49463</th>
      <td>talk.politics.misc</td>
      <td>stephanopoulos</td>
      <td>348</td>
      <td>0.60206</td>
      <td>1.530933</td>
    </tr>
    <tr>
      <th>41814</th>
      <td>talk.politics.misc</td>
      <td>drugs</td>
      <td>156</td>
      <td>0.60206</td>
      <td>1.322063</td>
    </tr>
    <tr>
      <th>43046</th>
      <td>talk.politics.misc</td>
      <td>gay</td>
      <td>128</td>
      <td>0.60206</td>
      <td>1.270702</td>
    </tr>
  </tbody>
</table>
</div>



This is an example of a class-based tf-idf, which is very useful when doing something like topic modelling, since we want to know what are the keywords for each of our predicted topics. We will cover how to use text embeddings for topic modelling in later posts.
