#!/usr/bin/env python
# coding: utf-8

# # Машинное обучение. Лабораторная работа №1.

# Выполнил: Петрухин Дмитрий Олегович
# 
# Группа: М8О-301Б-18.

# In[1]:


import pandas as pd
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import operator
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


df_wine = pd.read_csv('winequality-white.csv', sep=';')


# In[3]:


df_wine.head()


# In[4]:


df_wine.info()


# In[5]:


df_wine['quality'].value_counts()


# In[6]:


df_wine.describe()


# In[7]:


df_wine.hist(bins=50, figsize=(20,15))
plt.show()


# Распределение значение параметров имеют нормальное распределение

# Введем 2 категории вина вместо 7

# In[8]:


df_wine.loc[df_wine['quality'] < 6, 'quality'] = 0
df_wine.loc[ ((df_wine['quality'] > 5)& (df_wine['quality'] < 10)), 'quality'] = 1


# In[9]:


df_wine['quality'].value_counts()


# In[10]:


numeric_data = df_wine.select_dtypes(include=[np.number])


# In[11]:


numeric_attributes = numeric_data.columns


# In[12]:


numeric_attributes


# In[13]:


categorial_attributes=df_wine.select_dtypes(exclude=[np.number]).columns


# In[14]:


categorial_attributes


# Пропуски

# In[15]:


df_wine.isna().sum()


# In[16]:


df_wine.isnull().sum()


# In[17]:


df_wine.duplicated().sum()


# Пропусков нет, но есть дубликаты. 

# In[18]:


df_wine = df_wine.drop_duplicates().reset_index(drop = True)


# In[19]:


df_wine['quality'].value_counts()


# In[20]:


df_wine.boxplot(column=['fixed acidity'])
plt.ylim(0,15)


# In[21]:


df_wine.boxplot(column=['volatile acidity'])
plt.ylim(0,1.25)


# In[22]:


df_wine.boxplot(column=['citric acid'])
plt.ylim(0,2)


# In[23]:


df_wine.boxplot(column=['residual sugar'])
plt.ylim(0,30)


# In[24]:


df_wine.boxplot(column=['chlorides'])
plt.ylim(0,0.4)


# In[25]:


df_wine.boxplot(column=['free sulfur dioxide'])
plt.ylim(0,160)


# In[26]:


df_wine.boxplot(column=['total sulfur dioxide'])
plt.ylim(0,500)


# In[27]:


df_wine.boxplot(column=['density'])
plt.ylim(0.975,1.05)


# In[28]:


df_wine.boxplot(column=['pH'])
plt.ylim(2.5,4)


# In[29]:


df_wine.boxplot(column=['sulphates'])
plt.ylim(0.2,1.2)


# In[30]:


df_wine.boxplot(column=['alcohol'])
plt.ylim(7.5,16)


# Так как в данных имеются аномалии, то нормировка будет следующей:
# 
# $$\widetilde{x_i} = \frac{x_i - \overline{x_i}}{\sigma_i}$$
# Где 
# $$ \overline{x_i} = \frac{1}{N} \sum_{k=1}^N X^k_I - среднее \ значение$$

# In[31]:


corr = df_wine.corr()
with sns.axes_style("white"):
    ax = plt.subplots(figsize=(15, 10))
    ax = sns.heatmap(corr, square=True,annot = True, linewidths=.5)


# # Нормировка данных

# Так как границы значений у параметров различные, то необходимо привести их к одной границе(от 0 до 1).

# In[32]:


parametr = df_wine.columns.tolist()
normalized_df_wine = pd.DataFrame()
for i in range(len(parametr)-1):
    normalized_df_wine[parametr[i]]=(df_wine[parametr[i]]-df_wine[parametr[i]].min())/(df_wine[parametr[i]].max()-df_wine[parametr[i]].min())
normalized_df_wine['quality'] = df_wine['quality']   


# # Создаем тестовые и обучающие данные

# In[33]:


features, target = normalized_df_wine.drop(columns=['quality']).to_numpy(), np.array(normalized_df_wine['quality'])


# In[34]:


def split_train_test(X, y, test_ratio):
  np.random.seed(42)
  shuffled_indices = np.random.permutation(len(X))
  test_set_size = int(len(X)*test_ratio)
  test_indices = shuffled_indices[:test_set_size]
  train_indices = shuffled_indices[test_set_size:]
  return X[train_indices], X[test_indices], y[train_indices], y[test_indices] # вернет строки, указанные в idices


# In[35]:


X_train, X_test, y_train, y_test = split_train_test(features, target, .2)
print(len(X_train),'train +',len(X_test),'test')


# 80% обучающие и 20 тестовые.

# ## KNN

# In[48]:


class KNN:
    
    def __init__(self, features, target):
        self.features = features
        self.target = target
        
    def euclide_metrics(self, i,object_2, length):
        distance = 0
        object_1 = self.features[i]
        for x in range (length):
            distance += pow((object_1[x] - object_2[x]), 2)
        return math.sqrt(distance)
    
    def chooseNeighbors(self, test_features, k):
        distances = []
        length = len(test_features)
        for x in range(len(self.features)):
            dist = self.euclide_metrics(x, test_features, length)
            distances.append((self.target[x], dist))
        distances.sort(key=operator.itemgetter(1))

        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])
        return neighbors
    
    def getResponse(self, neighbors):
        classVotes = {}
        for x in range(len(neighbors)):
            response = neighbors[x]
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1
        sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
        return sortedVotes[0][0]
    
    def getAccuracy(self, target, predictions):
        correct = 0
        for x in range(len(target)):
            if target[x] == predictions[x]:
                correct += 1
        return round(correct / float(len(target)), 4) * 100.0
    
    def score(self, test_features, test_target, k):
        predictions = []
        for x in range(len(test_features)):
            neighbors = self.chooseNeighbors(test_features[x], k)
            result = self.getResponse(neighbors)
            predictions.append(result)
        accuracy = self.getAccuracy(test_target, predictions)
        return(accuracy)


# In[49]:


my_knn = KNN(X_train, y_train)


# In[50]:


print('Расчет точности реализации KNN на обучающей выборке: ', my_knn.score(X_train, y_train, 3), '%')


# In[51]:


print('Расчет точности реализации KNN на тестовой выборке: ', my_knn.score(X_test, y_test, 3), '%')


# In[52]:


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print('Результат sklearn реализации KNN на обучающей выборке: {:.2%}'.format(knn.score(X_train, y_train)))
print('Результат sklearn реализации KNN на тестовой выборке: {:.2%}'.format(knn.score(X_test, y_test)))


# ## Naive bayes

# In[53]:


from scipy.stats import norm


# In[54]:


class NaiveBayes:
    
    def fit(self, X, y):
        self.label_probabilities = {
            0: len(y[y == 0]) / len(y),
            1: len(y[y == 1]) / len(y)
        }
        self.conditional_probabilities = {}
        for value in (0, 1):
            probabilities = []
            for column in range(X.shape[1]):
                x = X[y == value, column]
                probabilities.append([x.mean(), x.std()])
            self.conditional_probabilities[value] = probabilities
            
    
    def predict(self, X):
        label_probabilities = {}
        for value in (0, 1):
            conditional_feature_probability = 0
            for i in range(X.shape[0]):
                conditional_feature_probability += np.log(norm(self.conditional_probabilities[value][i][0],
                                                               self.conditional_probabilities[value][i][1]).pdf(X[i]))
            label_probabilities[value] = np.log(self.label_probabilities[value]) + conditional_feature_probability
        if (label_probabilities[1] > label_probabilities[0]):
            return 1 
        else:
            return 0


    def score(self, X, y):
        rigth_predict_number = 0
        for i in range(X.shape[0]):
            if self.predict(X[i]) == y[i]:
                rigth_predict_number += 1

        return round(rigth_predict_number / y.shape[0], 4) * 100


# In[57]:


X_train[5]


# In[55]:


sklearn_nb = naive_bayes.GaussianNB()
sklearn_nb.fit(X_train, y_train)
print('Результат sklearn реализации наивного байесовского классификатора на обучающей выборке: {:.2%}'
      .format(sklearn_nb.score(X_train, y_train)))
print('Результат sklearn реализации наивного байесовского классификатора на тестовой выборке: {:.2%}'
      .format(sklearn_nb.score(X_test, y_test)))


# In[56]:


nb = NaiveBayes()
nb.fit(X_train, y_train)
print('Результат собственной реализации наивного байесовского классификатора на обучающей выборке: ', 
      nb.score(X_train, y_train),'%')
print('Результат собственной реализации наивного байесовского классификатора на тестовой выборке: ', 
      nb.score(X_test, y_test), '%')

