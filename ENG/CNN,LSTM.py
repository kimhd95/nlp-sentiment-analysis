import json
import re
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import Text
import collections
from keras.layers.core import Dense, SpatialDropout1D 
from keras.layers.convolutional import Conv1D 
from keras.layers.embeddings import Embedding
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers import LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences 
from keras.utils import np_utils 
from sklearn.model_selection import train_test_split

# ======================== cell 1 =============================

nltk.download('stopwords')
stops = set(stopwords.words('english'))
stemmer = nltk.stem.SnowballStemmer('english')

with open('./Friends/friends_train.json') as json_file:
    json_train = json.load(json_file)
with open('./Friends/friends_test.json') as json_file:
    json_test = json.load(json_file)
with open('./Friends/friends_dev.json') as json_file:
    json_dev = json.load(json_file)

def cleaning(str):
    replaceAll= str
    only_english = re.sub('[^a-zA-Z]', ' ', replaceAll)
    no_capitals = only_english.lower().split()
    no_stops = [word for word in no_capitals if not word in stops]
    stemmer_words = [stemmer.stem(word) for word in no_stops]
    return ' '.join(stemmer_words)

i = 0
train_data=[]
for rows in json_train:
    for row in rows:
        train_data.append([cleaning(row['utterance']), row['emotion']])
for rows in json_test:
    for row in rows:
        train_data.append([cleaning(row['utterance']), row['emotion']])
for rows in json_dev:
    for row in rows:
        train_data.append([cleaning(row['utterance']), row['emotion']])

# ======================== cell 2 =============================

cnt = 0
tagged = []
counter = collections.Counter()
for d in train_data:
    cnt = cnt + 1
    if cnt % 1000 == 0:
        print(cnt)
    words = pos_tag(word_tokenize(d[0]))
    for t in words:
        word = "/".join(t)
        tagged.append(word)
        counter[word] += 1

# ======================== cell 3 =============================

VOCAB_SIZE = 5000
word2index = collections.defaultdict(int)
for wid, word in enumerate(counter.most_common(VOCAB_SIZE)):
    word2index[word[0]] = wid + 1
vocab_sz = len(word2index) + 1
index2word = {v:k for k, v in word2index.items()}

# ======================== cell 4 =============================

def labeltoint(str):
    return {'non-neutral': 0,
             'neutral': 1, 
             'joy': 2,
             'sadness': 3,
             'fear': 4,
             'anger': 5,
             'surprise': 6,
             'disgust': 7}[str]

xs, ys = [], []
cnt = 0
maxlen = 0
for d in train_data:
    cnt = cnt + 1
    ys.append(labeltoint(d[1]))
    if cnt % 1000 == 0:
        print(cnt)
    ang = pos_tag(word_tokenize(d[0]))
    words=[]
    for t in ang:
        words.append("/".join(t))
    if len(words) > maxlen: 
        maxlen = len(words)
    wids = [word2index[word] for word in words]
    xs.append(wids)

# ======================== cell 5 =============================

X = pad_sequences(xs, maxlen=maxlen) 
Y = np_utils.to_categorical(ys)
 
EMBED_SIZE = 100 
NUM_FILTERS = 256 
NUM_WORDS = 3 
BATCH_SIZE = 64 
NUM_EPOCHS = 20

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
model = Sequential() 
model.add(Embedding(vocab_sz, EMBED_SIZE, input_length=maxlen)) 
model.add(SpatialDropout1D(0.2)) 
#model.add(Conv1D(filters=NUM_FILTERS, kernel_size=NUM_WORDS, activation="relu")) 
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2)) 
model.add(GlobalMaxPooling1D()) 
model.add(Dense(8, activation="softmax")) 
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]) 

history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS, validation_data=(x_test, y_test)) 

# ======================== cell 6 =============================

fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(history.history['loss'], 'y', label='train loss')
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')

acc_ax.plot(history.history['accuracy'], 'b', label='train acc')
acc_ax.plot(history.history['val_accuracy'], 'g', label='val acc')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='upper left')

plt.show()

# ======================== cell 7 =============================

def inttolabel(idx):
    return {0:'non-neutral',
             1:'neutral', 
             2:'joy',
             3:'sadness',
             4:'fear',
             5:'anger',
             6:'surprise',
             7:'disgust'}[idx]

def predict(text): 
    aa = pos_tag(word_tokenize(text))
    pp = []
    for t in aa:
        pp.append("/".join(t))
    wids = [word2index[word] for word in pp]
    x_predict = pad_sequences([wids], maxlen=maxlen) 
    y_predict = model.predict(x_predict) 
    c = 0
    cnt = 0
    for y in y_predict[0]:
        if c < y:
            c = y
            ans = cnt
        cnt += 1
    ans = inttolabel(ans)
    return ans;

# ======================== cell 8 =============================     

with open('en_data.csv', 'r', newline='') as csvfile:
    df = pd.read_csv(csvfile)
cnt = 0
dap = []
for i in df['utterance']:
    cnt+=1
    if cnt % 1000 == 0 : 
        print(cnt)
    dap.append(predict(i))
result = [['Id','Predicted']]
cnt = -1

for i in dap:
    cnt += 1
    result.append([cnt, i])
dataframe = pd.DataFrame(result)
dataframe.to_csv("test.csv", header=False, index=False)

# ======================== cell 9 =============================  

# for real test

ans = predict('i love it')
print(ans)

# ======================== cell 10 =============================  