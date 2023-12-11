import gensim
from gensim import corpora
from gensim.models import LdaModel

# 전처리된 텍스트 데이터 (예시)
texts = [['고양이', '애교', '귀여움'],
         ['개', '충성', '친구'],
         ['고양이', '잠', '많음'],
         ]

# 단어-문서 매트릭스 생성
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
# LDA 모델 학습
lda = LdaModel(corpus, num_topics=3, id2word=dictionary, passes=20)

# 결과 해석
topics = lda.print_topics(num_words=5)
for topic in topics:
    print(topic)