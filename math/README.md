# 수식 학습
## 첫 번째 논문 : '추천 시스템 기법 연구동향 분석' 에 포함된 수식

### 1. 콘텐츠 기반 접근 방식

#### 1.1. 아이템 속성 분석

##### 1.1.1. TF - IDF(단어 가중치)
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
[Reference Url](http://stevenloria.com/finding-important-words-in-a-document-using-tf-idf/)  

##### 1.1.2. 코사인 유사도
```python
def cosine_similarity(vector1, vector2):
    dot_product = sum(p*q for p,q in zip(vector1, vector2))
    magnitude = math.sqrt(sum([val**2 for val in vector1])) * math.sqrt(sum([val**2 for val in vector2]))
    if not magnitude:
        return 0
return dot_product/magnitude
```
[Reference Url](http://billchambers.me/tutorials/2014/12/22/cosine-similarity-explained-in-python.html)  
### 2. 협력 필터링

#### 2.1. 기억 기반 협력 필터링 - 유사도 측정

##### 2.1.1. 피어슨 상관계수
```python
#def - p1과 p2에 대한 피어슨 상관 계수를 리턴
def sim_pearson(prefs, p1, p2):
	si = {}
	for item in prefs[p1]:
		for item in prefs[p2]: 
			si[item] = 1

	# 공통 요소의 개수를 구함, 없으면 종료
	n = len(si)
	if n==0 
		return 0

		# 모든 선호도를 합산함
	sum1 = sum([prefs[p1][it] for it in si])
	sum2 = sum([prefs[p2][it] for it in si])

	# 제곱의 합을 계산
	sum1Sq = sum([pow(prefs[p1][it], 2) for it in si])
	sum2Sq = sum([pow(prefs[p2][it], 2) for it in si])
	
	# 곱의 합을 계산
	pSum = sum([prefs[p1][it] * prefs[p2][it] for it in si])
	
	# 피어슨 점수 계산
	num = pSum - (sum1 * sum2 / n)
	den = sqrt((sum1Sq - pow(sum1, 2) / n) * (sum2Sq - pow(sum2, 2)/n))
	if den==0: 
		return 0
	
	r = num/den
	return r
```
[Reference](http://atin.tistory.com/57)  

##### 2.1.2. 보완 코사인 유사도

##### 2.1.3. 스피어만 순위 상관계수

#### 2.2. 기억 기반 협력 필터링 - 선호도 예측

##### 2.2.1. 가중합
##### 2.2.2. 단순 가중 평균

#### 2.3. 모델 기반 협력 필터링

##### 2.3.1. 나이브 베이즈
[Reference](How To Implement Naive Bayes From Scratch in Python)  
#### 2.4. 차원 축소

##### 2.4.1. Truncated-SVD

#### 2.5. 하이브리스 시스템

##### 2.5.1. LSI
##### 2.5.2. PLSI

### 3. 평가 방법

#### 3.1. 점수 예측 평가방법

##### 3.1.1. MSE
##### 3.1.2. RMSE
##### 3.1.3. MAE
##### 3.1.4. NMAE

#### 3.2. 아이템 추천 평가방법

##### 3.2.1. misclassification ratio
##### 3.2.2. Precision
##### 3.2.3. Recall
##### 3.2.4. F-measure

#### 3.3. 정확도 기반 평가방법

##### 3.3.1. EU(시스템 효용)
##### 3.3.2. hit-rate
##### 3.3.3. hit-rank

#### 3.4. 다양성 기반 평가방법

##### 3.4.1. diversity
##### 3.4.2. novelty
##### 3.4.3. 추천 아이템의 개인화 정도
##### 3.4.4. 특정한 상황에서의 특이성
##### 3.4.5. unexpecteness
##### 3.4.6. prediction coverage
