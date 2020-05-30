from pandas import read_csv
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
sns.set()
df = read_csv('C:/Users/Владислав/Desktop/Diplom/flats_moscow.csv')
# фильтруем от выбросов
def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3 - q1  # Interquartile range
    fence_low = q1 - 1.5 * iqr
    fence_high = q3 + 1.5 * iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out
col_name = 'price'
df = remove_outlier(df, col_name)
df = df.drop(df.columns[0], axis = 1)  #удаляем столбец с номером наблюдения
#--------------------------------------------------------------
trg = df[['price']] # выбираем то что будем предсказывать
trn = df.drop(df.columns[0], axis = 1)  # выбираем то по чему будем предсказывать
# распределяем выборку на тестовую и тренировочную
Xtrn, Xtest, Ytrn, Ytest = train_test_split(trn, trg, test_size=0.3)
#собираем регрессию
model = linear_model.Lasso(alpha=0.085)
model.fit(Xtrn, Ytrn)
# высчитываю r^2
score = model.score(Xtest, Ytest)
print('r2 = ',score)
#записываю предсказания
predictions = model.predict(Xtest)
#для дальнейших вычислений переиндексирую ДатаФрейм
Ytest = Ytest.reset_index()

x1 = []
y1 = []
#высчитываю среднюю ошибку для графика
for i in range(len(predictions)):
    y1.append(float(Ytest.loc[i,'price'] - predictions[i]))
    x1.append(float(predictions[i]))
fig, ax = plt.subplots()
ax.scatter(x1, y1)
plt.title(' RSSE для Lasso, alpha = 0.085')
ax.set_xlabel('предсказанная цена')
ax.set_ylabel('разница с реальной ценой')
plt.show()

RSS = []
#высчитываю среднюю ошибку
for i in range(len(predictions)):
    RSS.append((float(Ytest.loc[i,'price'] - predictions[i]))**2)
RSSE = (np.sum(RSS)/len(predictions))**0.5

print('RSSE = ', RSSE)

# Рисую график зависимости RSSE от альфа
x = []
y = []
for i in range(1,100,1):
    model = linear_model.Lasso(alpha=i/1000)
    model.fit(Xtrn, Ytrn)
    # записываю предсказания
    predictions = model.predict(Xtest)
    RSS = []
    # высчитываю среднюю ошибку
    for j in range(len(predictions)):
        RSS.append((float(Ytest.loc[j, 'price'] - predictions[j])) ** 2)
    RSSE = (np.sum(RSS) / len(predictions)) ** 0.5
    x.append(i/1000)
    y.append(RSSE)
fig, ax = plt.subplots()
ax.plot(x, y)
plt.title(' RSSE в зависимости от alpha для Lasso')
ax.set_xlabel('alpha')
ax.set_ylabel('RSSE')
plt.show()