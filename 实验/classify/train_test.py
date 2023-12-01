# coding: utf-8
import sys

import numpy as np
from sklearn import metrics
import joblib
from sklearn.feature_extraction.text import HashingVectorizer

'''加载数据'''


def input_data(train_file, test_file):
    train_words = []
    train_tags = []

    for line in open(train_file, 'r', encoding='utf-8'):
        tks = line.split('\t')
        train_words.append(tks[0])
        train_tags.append(tks[1])

    test_words = []
    test_tags = []
    for line in open(test_file, 'r', encoding='utf-8'):
        tks = line.split('\t')
        test_words.append(tks[0])
        test_tags.append(tks[1])

    return train_words, train_tags, test_words, test_tags


'''文本向量化'''


def vectorize(train_words, test_words):
    # 停用词表
    with open('dict/stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = list([w.strip() for w in f])

    v = HashingVectorizer(stop_words=stopwords, n_features=30000)
    train_data = v.fit_transform(train_words)
    test_data = v.fit_transform(test_words)
    return train_data, test_data


'''计算准确率、召回率、F1'''


def evaluate(actual, pred):
    m_precision = metrics.precision_score(actual, pred, average='weighted')
    m_recall = metrics.recall_score(actual, pred, average='weighted')
    m_f1_score = metrics.f1_score(actual, pred, average='weighted')

    print(u' 准确率:     {0:0.3f}'.format(m_precision))
    print(u' 召回率:     {0:0.3f}'.format(m_recall))
    print(u' F1-score:   {0:0.3f}'.format(m_f1_score))


'''多种分类算法'''
class MultinomialNB:
    def __init__(self, alpha=1.0):
        # 构造函数，初始化多项式朴素贝叶斯模型，默认平滑参数（alpha）为1.0
        self.alpha = alpha
        self.log_class_prior = None  # 类别的先验概率的对数
        self.log_feature_proba = None  # 在给定类别的情况下，特征的条件概率的对数

    def fit(self, X, y):
        # 将多项式朴素贝叶斯模型拟合到训练数据上
        n_samples, n_features = X.shape  # 获取训练数据中的样本数和特征数
        self.classes, class_counts = np.unique(y, return_counts=True)  # 获取唯一的类别和它们的样本数

        # 计算带有平滑的对数类别先验概率
        self.log_class_prior = np.log(class_counts + self.alpha) - np.log(n_samples + len(self.classes) * self.alpha)

        # 初始化一个数组用于存储在给定类别的情况下，每个特征的计数
        feature_counts = np.zeros((len(self.classes), n_features))
        # 遍历每个类别，计算每个特征的计数
        for i, c in enumerate(self.classes):
            feature_counts[i, :] = X[y == c].sum(axis=0)
        # 计算在给定类别的情况下，每个特征的条件概率的对数
        feature_proba = (feature_counts.T + self.alpha) / (class_counts + n_features * self.alpha)
        self.log_feature_proba = np.log(feature_proba).T

    # def predict(self, X):
    #     # 预测测试数据的类别标签
    #     log_posterior = X @ self.log_feature_proba.T + self.log_class_prior
    #     return self.classes[np.argmax(log_posterior, axis=1)]
    def predict(self, X):
        # 预测测试数据的类别标签
        log_posterior = X @ self.log_feature_proba.T + self.log_class_prior
        return self.classes[np.argmax(log_posterior, axis=1)]

# def train_clf_MNB(train_data, train_tags):
#     classifer = MultinomialNB()  # 创建多项式朴素贝叶斯分类器对象
#     train_data = train_data.toarray()  # 转换训练数据为稠密数组
#     train_tags = np.array(train_tags)  # 将训练标签转换为NumPy数组
#
#     # 数据预处理，确保数据非负
#     train_data -= train_data.min()
#
#     classifer.fit(train_data, train_tags)  # 训练分类器
#     return classifer
def train_clf_MNB(train_data, train_tags):
    classifer = MultinomialNB()  # 创建多项式朴素贝叶斯分类器对象
    train_data = train_data.toarray()  # 转换训练数据为稠密数组
    train_tags = np.array(train_tags)  # 将训练标签转换为NumPy数组

    # 数据预处理，确保数据非负
    train_data -= train_data.min()

    classifer.fit(train_data, train_tags)  # 训练分类器
    return classifer

def main():
    train_file = 'data/train.txt'
    test_file = 'data/test.txt'
    # 加载数据
    print('\nload =======================================')
    train_words, train_tags, test_words, test_tags = input_data(train_file, test_file)
    print('Step 1: input_data OK')
    train_data, test_data = vectorize(train_words, test_words)
    print('Step 2: vectorize OK')

    # Multinomial Naive Bayes Classifier
    print('\nMNB =======================================')
    # 训练
    clf = train_clf_MNB(train_data, train_tags)
    print('Step 3: train_clf OK')
    # 保存模型
    joblib.dump(clf, 'model/' + str(type(clf))[8:-2] + '.model')
    print('Step 4: model save OK')
    # 预测
    pred = clf.predict(test_data)
    print('Step 5: predict OK')
    # 计算准确率、召回率、F1
    evaluate(np.asarray(test_tags), pred)
    print('Step 6: evaluate OK')


if __name__ == '__main__':
    main()
