# RNNs-LSTM-Attention

## Part 1:
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
## Part 2:
Task was to train model we wrote in the class on the following two datasets taken from https://kili-technology.com/blog/chatbot-training-datasets/
1) http://www.cs.cmu.edu/~ark/QA-data/ 
2) https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs

These are Natural languange translation kind of task and we have following struture for both task.

download spacy:
```python
python -m spacy download en
````
Load spacy:
```python
spacy_en = spacy.load('en_core_web_sm')
````
Define the tokenize function:
```python

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]
````
In CMU dataset task is to generate answers for questions given as input.
For CMU dataset which is a question/answer dataset we defined fields like:

```python
Question = Field(tokenize = tokenize_en, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)

Answer = Field(tokenize = tokenize_en, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)
````
```python
s08_data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/cmuQA/Question_Answer_Dataset_v1.2/S08/question_answer_pairs.txt", sep='\t',encoding='ISO-8859-1')
s09_data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/cmuQA/Question_Answer_Dataset_v1.2/S09/question_answer_pairs.txt", sep='\t',encoding='ISO-8859-1')
s10_data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/cmuQA/Question_Answer_Dataset_v1.2/S10/question_answer_pairs.txt", sep='\t', quoting=3,encoding='ISO-8859-1')

s08_data = s08_data.drop(columns=['ArticleTitle','DifficultyFromQuestioner','DifficultyFromAnswerer','ArticleFile'])
s09_data = s09_data.drop(columns=['ArticleTitle','DifficultyFromQuestioner','DifficultyFromAnswerer','ArticleFile'])
s10_data = s10_data.drop(columns=['ArticleTitle','DifficultyFromQuestioner','DifficultyFromAnswerer','ArticleFile'])

QA_data = pd.concat([s08_data,s09_data,s10_data], axis =0)
````
After loading data we converted question and answer columns to string type and changed case to small:

```python
def smallCase(data):
    data['Question'] = data['Question'].str.lower()
    data['Answer'] = data['Answer'].str.lower()
    return data

QA_data.Question = np.array(QA_data.Question).astype(str)
QA_data.Answer = np.array(QA_data.Answer).astype(str)
QA_data = smallCase(pd.DataFrame(QA_data))

print(QA_data.head())

Question Answer
0  was abraham lincoln the sixteenth president of...    yes
1  was abraham lincoln the sixteenth president of...   yes.
2  did lincoln sign the national banking act of 1...    yes
3  did lincoln sign the national banking act of 1...   yes.
4                   did his mother die of pneumonia?     no

````
Using SKLearn we divided data into train, test and validation dataset:

```python

from sklearn.model_selection import train_test_split

train, test_data = train_test_split(QA_data, test_size=0.3)
train = train.reset_index().drop(columns=['index'])
test_data = test_data.reset_index().drop(columns=['index'])
train_data, valid_data = train_test_split(train, test_size=0.25)
train_data = train_data.reset_index().drop(columns=['index'])
valid_data = valid_data.reset_index().drop(columns=['index'])
````
```python
print(train_data.shape)
print(valid_data.shape)
print(test_data.shape)

(2098, 2)
(700, 2)
(1200, 2)
````
With following parameters we trained the model:
```python
INPUT_DIM = len(Question.vocab)
OUTPUT_DIM = len(Answer.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)
````
Training Log:
```python
Epoch: 01 | Time: 0m 2s
	Train Loss: 5.929 | Train PPL: 375.834
	 Val. Loss: 5.238 |  Val. PPL: 188.334
Epoch: 02 | Time: 0m 2s
	Train Loss: 5.083 | Train PPL: 161.283
	 Val. Loss: 5.365 |  Val. PPL: 213.806
Epoch: 03 | Time: 0m 2s
	Train Loss: 4.963 | Train PPL: 143.016
	 Val. Loss: 5.439 |  Val. PPL: 230.202
Epoch: 04 | Time: 0m 2s
	Train Loss: 4.870 | Train PPL: 130.366
	 Val. Loss: 5.497 |  Val. PPL: 244.036
Epoch: 05 | Time: 0m 2s
	Train Loss: 4.783 | Train PPL: 119.492
	 Val. Loss: 5.579 |  Val. PPL: 264.817
Epoch: 06 | Time: 0m 2s
	Train Loss: 4.771 | Train PPL: 118.024
	 Val. Loss: 5.639 |  Val. PPL: 281.226
Epoch: 07 | Time: 0m 2s
	Train Loss: 4.640 | Train PPL: 103.537
	 Val. Loss: 5.662 |  Val. PPL: 287.791
Epoch: 08 | Time: 0m 2s
	Train Loss: 4.597 | Train PPL:  99.199
	 Val. Loss: 5.737 |  Val. PPL: 310.028
Epoch: 09 | Time: 0m 2s
	Train Loss: 4.550 | Train PPL:  94.644
	 Val. Loss: 5.805 |  Val. PPL: 331.919
Epoch: 10 | Time: 0m 2s
	Train Loss: 4.523 | Train PPL:  92.139
	 Val. Loss: 5.805 |  Val. PPL: 331.918
````
Test log:
```python
| Test Loss: 5.263 | Test PPL: 192.987 |
````

**************************************************************************************************************************************************************

For Quora dataset which contains question and duplicate question we have following steps after defining our tokenize_en function:
The task here is to generate a duplicate question for given input question.
Define field as:
```python
question1 = Field(tokenize = tokenize_en, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)

question2 = Field(tokenize = tokenize_en, 
            init_token = '<sos>', 
            eos_token = '<eos>', 
            lower = True)
            
````
read data:
```python
data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/quoraQA/quora_duplicate_questions.tsv", sep='\t')
````
We then filter the dataset and keep only those records where dupicate question is confirmed.
```python
data = data[data['is_duplicate'] == 1]
print(data.head())

    id  qid1  ...                                          question2 is_duplicate
5    5    11  ...  I'm a triple Capricorn (Sun, Moon and ascendan...            1
7    7    15  ...          What should I do to be a great geologist?            1
11  11    23  ...             How can I see all my Youtube comments?            1
12  12    25  ...            How can you make physics easy to learn?            1
13  13    27  ...             What was your first sexual experience?            1
````
We then process the data as follows:
```python
data = data.drop(columns=['id', 'qid1', 'qid2','is_duplicate'])

def smallCase(data):
    data['question1'] = data['question1'].str.lower()
    data['question2'] = data['question2'].str.lower()
    return data
    
data.question1 = np.array(data.question1).astype(str)
data.question2 = np.array(data.question2).astype(str)
data = smallCase(pd.DataFrame(data))    

````

we then divide the data as per following test, train and validation set:
```python
from sklearn.model_selection import train_test_split

train, test_data = train_test_split(data, test_size=0.3)
train = train.reset_index().drop(columns=['index'])
test_data = test_data.reset_index().drop(columns=['index'])
train_data, valid_data = train_test_split(train, test_size=0.25)
train_data = train_data.reset_index().drop(columns=['index'])
valid_data = valid_data.reset_index().drop(columns=['index'])
````
With following parameters we define the model:
```python
INPUT_DIM = len(question1.vocab)
OUTPUT_DIM = len(question2.vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)
````

Training Log:
```python

Epoch: 01 | Time: 5m 9s
	Train Loss: 5.221 | Train PPL: 185.147
	 Val. Loss: 5.228 |  Val. PPL: 186.498
Epoch: 02 | Time: 5m 9s
	Train Loss: 4.358 | Train PPL:  78.126
	 Val. Loss: 4.727 |  Val. PPL: 112.998
Epoch: 03 | Time: 5m 10s
	Train Loss: 3.826 | Train PPL:  45.856
	 Val. Loss: 4.440 |  Val. PPL:  84.811
Epoch: 04 | Time: 5m 10s
	Train Loss: 3.463 | Train PPL:  31.907
	 Val. Loss: 4.225 |  Val. PPL:  68.366
Epoch: 05 | Time: 5m 10s
	Train Loss: 3.202 | Train PPL:  24.593
	 Val. Loss: 4.155 |  Val. PPL:  63.738
Epoch: 06 | Time: 5m 11s
	Train Loss: 2.980 | Train PPL:  19.691
	 Val. Loss: 4.085 |  Val. PPL:  59.412
Epoch: 07 | Time: 5m 11s
	Train Loss: 2.817 | Train PPL:  16.725
	 Val. Loss: 4.001 |  Val. PPL:  54.656
Epoch: 08 | Time: 5m 10s
	Train Loss: 2.672 | Train PPL:  14.467
	 Val. Loss: 4.005 |  Val. PPL:  54.847
Epoch: 09 | Time: 5m 10s
	Train Loss: 2.560 | Train PPL:  12.935
	 Val. Loss: 3.973 |  Val. PPL:  53.169
Epoch: 10 | Time: 5m 10s
	Train Loss: 2.434 | Train PPL:  11.403
	 Val. Loss: 3.979 |  Val. PPL:  53.480
````
Test log:
```python
| Test Loss: 3.995 | Test PPL:  54.317 |
````
