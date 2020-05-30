from pandas import read_csv
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR
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
model = SVR(kernel='linear', degree = 2)
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
plt.title(' RSSE для SVR (linear) регрессии')
ax.set_xlabel('предсказанная цена')
ax.set_ylabel('разница с реальной ценой')
plt.show()


RSS = []
#высчитываю среднюю ошибку
for i in range(len(predictions)):
    RSS.append((float(Ytest.loc[i,'price'] - predictions[i]))**2)
RSSE = (np.sum(RSS)/len(predictions))**0.5
print('RSSE = ', RSSE)

x = []
y = []
models = ['linear', 'poly', 'rbf', 'sigmoid']
for i in range(len(models)):
    model = SVR(kernel=models[i])
    model.fit(Xtrn, Ytrn)
    # записываю предсказания
    predictions = model.predict(Xtest)

    RSS = []
    # высчитываю среднюю ошибку
    for j in range(len(predictions)):
        RSS.append((float(Ytest.loc[j, 'price'] - predictions[j])) ** 2)
    RSSE = (np.sum(RSS) / len(predictions)) ** 0.5
    x.append(models[i])
    y.append(RSSE)

fig, ax = plt.subplots()
ax.plot(x, y)
plt.title(' RSSE в зависимости от модели для SVR')
ax.set_xlabel('предсказанная цена')
ax.set_ylabel('разница с реальной ценой')
plt.show()

#Сравниваем RSSE для разных степеней
x = []
y = []
for i in range(2,10,1):
    model = SVR(kernel='poly', degree = i)
    model.fit(Xtrn, Ytrn)
    # записываю предсказания
    predictions = model.predict(Xtest)

    RSS = []
    # высчитываю среднюю ошибку
    for j in range(len(predictions)):
        RSS.append((float(Ytest.loc[j, 'price'] - predictions[j])) ** 2)
    RSSE = (np.sum(RSS) / len(predictions)) ** 0.5
    x.append(i)
    print(i)
    y.append(RSSE)
    print(RSSE)
fig, ax = plt.subplots()
ax.plot(x, y)
plt.title('  RSSE в зависимости от степери для SVR (poly)')
ax.set_xlabel('степень')
ax.set_ylabel('RSSE')
plt.show()