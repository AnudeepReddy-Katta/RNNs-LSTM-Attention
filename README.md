# RNNs-LSTM-Attention

Part 1:
This assignement is about building an LSTM network for sentiment analysis on StanfordSentimentAnalysis Dataset. (http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip) which contains movie reviews and sentiment lables which can be considered into 5 labels. This dataset contains just over 10,000 pieces of Stanford data from HTML files of Rotten Tomatoes.
The dataset is loaded from the text files to dataframes, following files were used:  sentiment_labels.txt and datasetSentences.txt These files were uploaded to Colab and used pandas to read and create dataframes.
we have joined sentences to the sentiments using the file dictionary, datasetSentences and sentiment_labels after this data was around 12k records.
The sentiment values were converted to 5 classes using following code:
```python
for i in sentiment['sentiment values'] :
  if i >=0 and i<0.2:
    sentiment_class.append(1)
  elif i>=0.2 and i<0.4:
    sentiment_class.append(2)
  elif i>=0.4 and i<0.6:
    sentiment_class.append(3)
  elif i>=0.6 and i<0.8:
    sentiment_class.append(4)
  else:
    sentiment_class.append(5)
````
sentences were cleaned using code:
```python
def pre_process_sentences(string):
  string=string.replace('-LRB-','(')
  string=string.replace('-RRB-',')')
  string=string.replace('Â', '')
  string=string.replace('Ã©', 'e')
  string=string.replace('Ã¨', 'e')
  string=string.replace('Ã¯', 'i')
  string=string.replace('Ã³', 'o')
  string=string.replace('Ã´', 'o')
  string=string.replace('Ã¶', 'o')
  string=string.replace('Ã±', 'n')
  string=string.replace('Ã¡', 'a')
  string=string.replace('Ã¢', 'a')
  string=string.replace('Ã£', 'a')
  string=string.replace('\xc3\x83\xc2\xa0', 'a')
  string=string.replace('Ã¼', 'u')
  string=string.replace('Ã»', 'u')
  string=string.replace('Ã§', 'c')
  string=string.replace('Ã¦', 'ae')
  string=string.replace('Ã­', 'i')
  string=string.replace('\xa0', ' ')
  string=string.replace('\xc2', '')
  return string
````
dictionary dataframe column named phrase was cleaned using code:
```python
def pre_process_phrases(string):
    string=string.replace('é','e')
    string=string.replace('è','e')
    string=string.replace('ï','i')
    string=string.replace('í','i')
    string=string.replace('ó','o')
    string=string.replace('ô','o')
    string=string.replace('ö','o')
    string=string.replace('á','a')
    string=string.replace('â','a')
    string=string.replace('ã','a')
    string=string.replace('à','a')
    string=string.replace('ü','u')
    string=string.replace('û','u')
    string=string.replace('ñ','n')
    string=string.replace('ç','c')
    string=string.replace('æ','ae')
    string=string.replace('\xa0', ' ')
    string=string.replace('\xc2', '')    
    return string
````

After this we converted data to small case and cleaned using regular expression:
```python
def smallCase(data):
  for i in data.index:
    data['sentence'][i] = data['sentence'][i].lower()
  return data
  
import re
def cleanText(data):
  data_small_case = smallCase(data)
  for i in data_small_case.index:
    data_small_case.sentence[i] = re.sub("[^-9A-Za-z ]", "" , data_small_case.sentence[i])
  return data_small_case
````
we combined dataframes using pandas merge in following code:
```python
dataset = pd.merge(sentiment,dictionary,on='phrase_id')
dataset = dataset.drop(columns=['phrase_id'])

dataset_1 = pd.merge(left = dataset, right = sentences,left_on='phrase',right_on='sentence')
dataset_1 = dataset_1.drop(columns=['phrase'])
dataset_1.rename(columns = {'sentiment_values':'sentiments'}, inplace = True)
````

Then data was divided into train and test dataset in the ratio 70:30
Then we defined LSTM classifier with following hyperparameters:
```python
size_of_vocab = len(sentence.vocab)
embedding_dim = 200
num_hidden_nodes = 300
num_output_nodes = 6
num_layers = 4
dropout = 0.4
````
Model parameters:
```python

classifier(
  (embedding): Embedding(15092, 200)
  (encoder): LSTM(200, 300, num_layers=4, batch_first=True, dropout=0.4)
  (fc): Linear(in_features=300, out_features=6, bias=True)
)
The model has 5,789,806 trainable parameters
````
With Adam optimizer and learning rate of 2e-4,
we trained the model and training logs show:
```python
        Train Loss: 1.741 | Train Acc: 26.47%
	 Test. Loss: 1.715 |  Test. Acc: 30.17% 

	Train Loss: 1.706 | Train Acc: 32.45%
	 Test. Loss: 1.724 |  Test. Acc: 29.19% 

	Train Loss: 1.681 | Train Acc: 35.17%
	 Test. Loss: 1.694 |  Test. Acc: 32.52% 

	Train Loss: 1.651 | Train Acc: 38.59%
	 Test. Loss: 1.680 |  Test. Acc: 34.54% 

	Train Loss: 1.623 | Train Acc: 41.69%
	 Test. Loss: 1.681 |  Test. Acc: 34.20% 

	Train Loss: 1.594 | Train Acc: 45.05%
	 Test. Loss: 1.669 |  Test. Acc: 35.90% 

	Train Loss: 1.559 | Train Acc: 49.01%
	 Test. Loss: 1.677 |  Test. Acc: 33.77% 

	Train Loss: 1.526 | Train Acc: 52.89%
	 Test. Loss: 1.668 |  Test. Acc: 35.68% 

	Train Loss: 1.489 | Train Acc: 56.87%
	 Test. Loss: 1.663 |  Test. Acc: 36.29% 

	Train Loss: 1.456 | Train Acc: 60.22%
	 Test. Loss: 1.681 |  Test. Acc: 34.09% 
````
After training we tried on ten test sentences, and with following mapping of class:
categories = {1:"neutral", 2:"negative", 3:"positive", 4:"very positive", 5:"very negative"}

```python

The performances are so leaden , Michael Rymer 's direction is so bloodless and the dialogue is so corny that the audience laughs out loud .
predicted:  very negative
actual:  1


Lacks the inspiration of the original and has a bloated plot that stretches the running time about 10 minutes past a child 's interest and an adult 's patience .
predicted:  very negative
actual:  0


Works because , for the most part , it avoids the stupid cliches and formulaic potholes that befall its brethren .
predicted:  very negative
actual:  3


Cox is far more concerned with aggrandizing madness , not the man , and the results might drive you crazy .
predicted:  neutral
actual:  2


Stanley Kwan has directed not only one of the best gay love stories ever made , but one of the best love stories of any stripe .
predicted:  negative
actual:  4


Grown-up quibbles are beside the point here .
predicted:  very negative
actual:  2


Though the book runs only about 300 pages , it is so densely packed ... that even an ambitious adaptation and elaborate production like Mr. Schepisi 's seems skimpy and unclear .
predicted:  very negative
actual:  1


In the process , they demonstrate that there 's still a lot of life in Hong Kong cinema .
predicted:  very negative
actual:  3


Highly recommended viewing for its courage , ideas , technical proficiency and great acting .
predicted:  negative
actual:  4


This is rote drivel aimed at Mom and Dad 's wallet .
predicted:  very negative
actual:  1

````
