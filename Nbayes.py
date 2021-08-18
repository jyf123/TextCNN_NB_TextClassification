import jieba
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from utils import get_time_dif

# 数据加载
train_df = pd.read_csv("./data/train.txt", sep='\t', names=['content','label'])
test_df = pd.read_csv("./data/test.txt", sep='\t', names=['content','label'])
# print(train_df.info())
# print(test_df.info())
# print(train_df.head(5))
# print(test_df.head(5))


X_train = train_df['content']
y_train = train_df['label']
X_test = test_df['content']
y_test = test_df['label']
class_list = [x.strip() for x in open('data/class.txt', encoding='utf-8').readlines()]

# 数据预处理
def cut_content(data):
    """
    利用jieba工具进行中文分词
    """
    words = data.apply(lambda x: ' '.join(jieba.cut(x)))
    return words


# 加载停用词表
stopwords_file = open('./data/scu_stopwords.txt', encoding='utf-8')
stopwords_list = stopwords_file.readlines()
stopwords = [x.strip() for x in stopwords_list]



# TF-IDF提取特征
tfidf_vector = TfidfVectorizer(stop_words=stopwords, max_features=10000,lowercase=False, sublinear_tf=True, max_df=0.8)
tfidf_vector.fit(cut_content(X_train))
X_train_tfidf = tfidf_vector.transform(cut_content(X_train))
X_test_tfidf = tfidf_vector.transform(cut_content(X_test))
# print(X_train_tfidf.shape)  # (180000, 10000)
# print(X_test_tfidf.toarray()) # (10000, 10000)


clf_nb = MultinomialNB(alpha=0.2)  # 模型参数可以根据分类结果进行调优
# 使用TF-IDF作为特征向量
clf_nb.fit(X_train_tfidf, y_train)  # 模型训练
y_pred = clf_nb.predict(X_test_tfidf)  # 模型预测


# 查看各类指标
print(classification_report(y_test, y_pred,target_names=class_list,digits=4))

# # 查看混淆矩阵
# print(confusion_matrix(y_test, y_pred))
# print(confusion_matrix(y_test, y_pred_chi))
