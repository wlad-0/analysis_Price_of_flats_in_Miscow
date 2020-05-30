from pandas import read_csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
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
Xtrn, Xtest, Ytrn, Ytest = train_test_split(trn, trg, test_size=0.4)
#собираем модель случайного леса
model = RandomForestRegressor(n_estimators=100, max_features ='sqrt', max_depth = 13)
model.fit(Xtrn, np.ravel(Ytrn))
score = model.score(Xtest, Ytest)
print('r2 = ',score)
#записываю предсказания
predictions = model.predict(Xtest)
#для дальнейших вычислений переиндексирую ДатаФрейм
Ytest = Ytest.reset_index()

x1 = []
y1 = []
#высчитываю среднюю ошибку
for i in range(len(predictions)):
    y1.append(float(Ytest.loc[i,'price'] - predictions[i]))
    x1.append(float(predictions[i]))
fig, ax = plt.subplots()
ax.scatter(x1, y1)
plt.title(' RSSE для случайного леса')
ax.set_xlabel('предсказанная цена')
ax.set_ylabel('разница с реальной ценой')
plt.show()

RSS = []
#высчитываю среднюю ошибку
for i in range(len(predictions)):
    RSS.append((float(Ytest.loc[i,'price'] - predictions[i]))**2)
RSSE = (np.sum(RSS)/len(predictions))**0.5
#print(q)
print('RSSE = ', RSSE)

#Ищем лучшую глубину
x = []
y = []
for i in range(2,40,1):
    max_depth = i

    model = RandomForestRegressor(n_estimators=100, max_features='sqrt', max_depth = max_depth)
    model.fit(Xtrn, np.ravel(Ytrn))
    # записываю предсказания
    predictions = model.predict(Xtest)
    RSS = []
    # высчитываю среднюю ошибку
    for i in range(len(predictions)):
        RSS.append((float(Ytest.loc[i, 'price'] - predictions[i])) ** 2)
    RSSE = (np.sum(RSS) / len(predictions)) ** 0.5
    x.append(max_depth)
    y.append(RSSE)
    print(RSSE)

fig, ax = plt.subplots()
ax.plot(x, y)
plt.title(' RSSE в зависимости от глубины дерева')
ax.set_xlabel('глубина')
ax.set_ylabel('RSSE')
plt.show()