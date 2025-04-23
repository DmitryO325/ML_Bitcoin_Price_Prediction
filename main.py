import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

df = pd.read_csv('dataset/bitcoin.csv')

'Информация о наборе данных'
# print(df.head())
# print(df.shape)
# print(df.describe())
# print(df.info())

'Изменение цены биткоина в разное время'
# plt.figure(figsize=(15, 5))
# plt.plot(df['Close'])
# plt.title('Цена биткоина во время закрытия')
# plt.ylabel('Цена в долларах')
# plt.show()

# print(df[df['Close'] == df['Adj Close']].shape)
# print(df.shape)

'''Столбцы Close и Adj Close имеют идентичные значения, 
   поэтому удалим один из этих столбцов'''
df = df.drop(['Adj Close'], axis=1)

'Проверим наличие Null'
# print(df.isnull().sum())

'Посмотрим, как распределены признаки...'
# features = df.columns[1:-1]
# plt.figure(figsize=(16, 8))

# for index, column in enumerate(features):
#     plt.subplot(2, 2, index + 1)
#     sb.histplot(df[column], kde=True)
#     plt.title(column)
#
# plt.tight_layout()
# plt.show()

'...и насколько много выбросов'
# for index, column in enumerate(features):
#     plt.subplot(2, 2, index + 1)
#     sb.boxplot(df[column])
#
# plt.show()

'''Анализ показывает, что в данных достаточно много выбросов,
   т.е. за короткий период времени цены на акции меняются сильно'''

'Сгенерируем новые признаки из даты'
splitted_data = df['Date'].str.split('-', expand=True)

df['year'] = splitted_data[0].astype(int)
df['month'] = splitted_data[1].astype(int)
df['day'] = splitted_data[2].astype(int)

df['Date'] = pd.to_datetime(df['Date'])
# print(df.head())

'Оценим цены на биткоин по годам'
# grouped_data = df.groupby('year').mean()
# plt.figure(figsize=(16, 8))
#
# for index, column in enumerate(grouped_data.columns[1: -3]):
#     plt.subplot(2, 2, index + 1)
#     grouped_data[column].plot.bar()
#     plt.title(column)
#
# plt.tight_layout()
# plt.show()

'Анализ показывает, что пик цен на биткоин был в 2021 году'

'Конец расчётного периода'
df['is_quarter_end'] = np.where(df['month'] % 3 == 0, 1, 0)
# print(df.head(20))

'''Создаём столбцы с показателями разницы и целевые значения,
   где 1 - акции поднимутся на следующий день, 0 - нет'''

df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# print(df.head(20))

'Проверим, сбалансированы ли целевые значения'
# plt.pie(df['target'].value_counts().values, labels=['0', '1'], autopct='%1.1f%%')
# plt.show()

'Анализ показывает, что целевые значения достаточно сбалансированы'

'Проверим, насколько признаки коррелируют между собой'
# plt.figure(figsize=(10, 10))

'''Поскольку нас интересуют только сильно коррелирующие признаки, то мы будем
   визуализировать heatmap по этим критериям'''
# sb.heatmap(df.corr(), annot=True, cbar=False)
# sb.heatmap(df.corr() > 0.9, annot=True, cbar=False)
# plt.show()

'''Видим, что изначально признаки линейно зависят друг от друга, 
   поэтому для обучения модели будем использовать новые признаки'''

features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']

'Нормируем признаки для более корректной работы алгоритмов'
scaler = StandardScaler()
features = scaler.fit_transform(features)

'''Не будем использовать train_test_split,
   возьмём первые ~70% для обучения и ~30% для тестирования'''
X_train, X_test = features[:len(features) * 6 // 7], features[len(features) * 6 // 7:]
y_train, y_test = target[:len(target) * 6 // 7], target[len(target) * 6 // 7:]

# print(len(features))
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

'''Обучим 3 модели:
   1. Логистическая регрессия
   2. Метод опорных векторов
   3. Градиентный бустинг
'''
models = [LogisticRegression(), SVC(kernel='poly', probability=True), XGBClassifier()]

for i in range(len(models)):
    models[i].fit(X_train, y_train)

    print(f'Модель: {models[i]}')
    print(f'ROC-AUC на обучающей выборке: {metrics.roc_auc_score(y_train, models[i].predict_proba(X_train)[:, 1])}')
    print(f'ROC-AUC на тестовой выборке: {metrics.roc_auc_score(y_test, models[i].predict_proba(X_test)[:, 1])}')
    print()

'Построим матрицу ошибок'
ConfusionMatrixDisplay.from_estimator(models[0], X_test, y_test)
plt.show()

'''Вывод:
   XGBClassifier сильно переобучается
   Данных недостаточно для обучения таких простых моделей
   В данном случае лучше использовать логистическую регрессию
'''
