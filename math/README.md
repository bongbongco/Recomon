# 수식 학습
## 첫 번째 논문 : '추천 시스템 기법 연구동향 분석' 에 포함된 수식

### 콘텐츠 기반 접근 방식

#### 아이템 속성 분석

##### TF - IDF(단어 가중치)
```python
from __future__ import division, unicode_literals

import math
from textblob import TextBlob as tb

def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)
```
+ tf(word, blob) 용어의 빈도 계산
+ n_containing(word, bloblist) 문서 내 포함된 단어의 수를 반환
+ idf(word, bloblist) 역 문서 빈도 계산 (일반적인 단어일 수록 IDF가 낮음)
+ tfidf(word, blob, bloblist) TF-IDF 점수를 계산

```python
document1 = tb("""Python is a 2000 made-for-TV horror movie directed by Richard
Clabaugh. The film features several cult favorite actors, including William
Zabka of The Karate Kid fame, Wil Wheaton, Casper Van Dien, Jenny McCarthy,
Keith Coogan, Robert Englund (best known for his role as Freddy Krueger in the
A Nightmare on Elm Street series of films), Dana Barron, David Bowe, and Sean
Whalen. The film concerns a genetically engineered snake, a python, that
escapes and unleashes itself on a small town. It includes the classic final
girl scenario evident in films like Friday the 13th. It was filmed in Los Angeles,
 California and Malibu, California. Python was followed by two sequels: Python
 II (2002) and Boa vs. Python (2004), both also made-for-TV films.""")

document2 = tb("""Python, from the Greek word (πύθων/πύθωνας), is a genus of
nonvenomous pythons[2] found in Africa and Asia. Currently, 7 species are
recognised.[2] A member of this genus, P. reticulatus, is among the longest
snakes known.""")

document3 = tb("""The Colt Python is a .357 Magnum caliber revolver formerly
manufactured by Colt's Manufacturing Company of Hartford, Connecticut.
It is sometimes referred to as a "Combat Magnum".[1] It was first introduced
in 1955, the same year as Smith &amp; Wesson's M29 .44 Magnum. The now discontinued
Colt Python targeted the premium revolver market segment. Some firearm
collectors and writers such as Jeff Cooper, Ian V. Hogg, Chuck Hawks, Leroy
Thompson, Renee Smeets and Martin Dougherty have described the Python as the
finest production revolver ever made.""")

bloblist = [document1, document2, document3]
for i, blob in enumerate(bloblist):
    print("Top words in document {}".format(i + 1))
    scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words[:3]:
        print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))
```
~~~
Top words in document 1
    Word: films, TF-IDF: 0.00997
    Word: film, TF-IDF: 0.00665
    Word: California, TF-IDF: 0.00665
Top words in document 2
    Word: genus, TF-IDF: 0.02192
    Word: among, TF-IDF: 0.01096
    Word: Currently, TF-IDF: 0.01096
Top words in document 3
    Word: Magnum, TF-IDF: 0.01382
    Word: revolver, TF-IDF: 0.01382
    Word: Colt, TF-IDF: 0.01382
~~~
Reference Url : http://stevenloria.com/finding-important-words-in-a-document-using-tf-idf/
##### 코사인 유사도

### 협력 필터링

#### 기억 기반 협력 필터링 - 유사도 측정

##### 피어슨 상관계수
##### 보완 코사인 유사도
##### 스피어만 순위 상관계수

#### 기억 기반 협력 필터링 - 선호도 예측

##### 가중합
##### 단순 가중 평균

#### 모델 기반 협력 필터링

##### 나이브 베이즈

#### 차원 축소

##### Truncated-SVD

#### 하이브리스 시스템

##### LSI
##### PLSI

### 평가 방법

#### 점수 예측 평가방법

##### MSE
##### RMSE
##### MAE
##### NMAE

#### 아이템 추천 평가방법

##### misclassification ratio
##### Precision
##### Recall
##### F-measure

#### 정확도 기반 평가방법

##### EU(시스템 효용)
##### hit-rate
##### hit-rank

#### 다양성 기반 평가방법

##### diversity
##### novelty
##### 추천 아이템의 개인화 정도
##### 특정한 상황에서의 특이성
##### unexpecteness
##### prediction coverage
