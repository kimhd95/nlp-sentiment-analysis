## Englsih emotion analysis

**모델 및 각종 아이디어의 경우 오픈되어 있는 소스를 참고했습니다.**(아래 링크 첨부)

## Dataset information

CNN, LSTM 모델 의 경우 'friends_train.json' , 'frineds_test.json' , 'frineds_dev.json' 데이터를 모두 합친 후,  sklearn 패키지의 train_test_split을 이용해 70% 30% 비율로 train, test 를 설정했습니다. Bert 모델의 경우 'frineds_train', 'frined_dev' 를 train 으로 'frineds_test'  를 test set 으로 설정했습니다. 각 데이터셋의 경우 발화(utterance) (최대길이  = maxlen) 와 그에 해당하는 감정 라벨이 주어집니다. 각 감정라벨에 해당하는 인덱스는 아래에 소개됩니다.

## Requirements

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `nltk`
- `keras` with `TensorFlow` backend
- `transformers` (for BERT model)
- `torch` (for BERT model)

## Usage(+code elxplanation)

- 저희의 실행환경은 colab 을 사용하였으며, 따라서 해당 코드를 colab 에서 아래 순서 그대로 실행시켜 주시면 됩니다.

각 번호는 cell 번호를 의미합니다.

#### CNN, LSTM (gpu 사용 x)

1. 필요한 package 모두 import

2. json 파일 읽은 후, 3파일 모두 cleaning 함수를 통해 아래 과정 진행 후 train_data에 저장

   1. 영어 이외 data re 패키지를 이용해 제거
   2. 소문자로 모두 통일
   3. nltk 의 stopwords를 이용해 불용어 제거
   4. nltk 의 stemmer를 이용해 stemming

3. 구해진 문장을 nltk 패키지를 이용해 품사를 tagging 한 후, 품사와 함께 join 해 단어 빈도를 collection 패키지로 저장 (실행 경과 1000 단위로 찍어냄)

4. 3번 cell에서 구해진 collection을 바탕으로, 가장 많이 사용된 단어 순으로 VOCAB_SIZE 개수만큼 추려내, indexing 해줌 (word2index,  index2word 두 dictionary 생성)

5. 만든 dictionary를 이용해 train_data를 word2index를 이용해 vector화 xs, ys 에 각 train_data 와 label을 저장

6. maxlen 에 맞추어 xs padding, train, test split 후, 파라미터를 사용해 모델 정의후 학습 진행 (history 에 저장)

   아래 두 라인 중 하나를 선택 후 하나는 주석처리 후 실행(위 : CNN, 아래: LSTM)

   ```
   # model.add(Conv1D(filters=NUM_FILTERS, kernel_size=NUM_WORDS, activation="relu"))
   model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2)) 
   ```

7. history 에 저장된 loss 와 accuracy plotting

8. 실제 test를 위해 문장 주어질 시 감정을 return 하는 predict 함수 선언

9. kaggle 데이터를 불러온 후 제출을 위해 dataframe 생성하는 부분

10. 실제 문장 확인하는 구문 predict('') 안의 내용을 원하는 문장으로 바꿔주면 됩니다.

#### BERT (gpu 있을시 사용)

1. 필요한 package 모두 import
2. json 파일 읽은 후, 3파일 모두 cleaning 함수를 통해 아래 과정 진행 후 train, dev는 train_data에 저장 test는 test_data에 저장
   1. 영어 이외 data re 패키지를 이용해 제거
   2. 소문자로 모두 통일
   3. nltk 의 stopwords를 이용해 불용어 제거
   4. nltk 의 stemmer를 이용해 stemming
3. 구해진 문장을 bert tokenizer를 사용해 토큰으로 분리하기 위해, 문장 편집 (앞에 [CLS] , 뒤에 [SEP] 을 달아줍니다. cls : classification , sep : 문장 구분) 후, token으로 분리
4. 3번에서 구해진 token을 숫자 값으로 indexing 하고, maxlen을 이용해 padding 진행, 그리고 attention_masks를 설정 (데이터가 >0 인 단어부분에 attention을 주어서 학습 속도와 성능을 향상시킵니다.)
5. labeltoint 함수를 생성해 labels에 train_data의 label을 저장, 학습을 위해 torch tensor 형태로 모든 데이터들을 변환해줍니다.
6. cell3~cell5에 해당하는 과정을 test_data에 대해서도 진행해줍니다.
7. GPU 사용가능 여부를 확인(colab의 경우 가능) 후, pretrained 된 모델을 model에 불러오고, optimizer와 각종 파라미터, scheduler들을 세팅
8. training 진행
9. kaggle 데이터를 불러온 후 제출을 위해 dataframe 생성하는 부분
10. 실제 문장 확인하는 구문 predict('') 안의 내용을 원하는 문장으로 바꿔주면 됩니다.

## References

https://myjamong.tistory.com/77

http://aidev.co.kr/chatbotdeeplearning/8709

https://neurowhai.tistory.com/294

