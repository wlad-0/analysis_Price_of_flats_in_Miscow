from pandas import read_csv
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
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
#собираем регрессию
regression = linear_model.LinearRegression()
poly = PolynomialFeatures(degree=2)
poly_X_train = poly.fit_transform(Xtrn)
poly_X_test = poly.fit_transform(Xtest)
# обучаю модель
model = regression.fit(poly_X_train, Ytrn)
score = model.score(poly_X_test, Ytest)
print('r2 = ',score)

#записываю предсказания
predictions = model.predict(poly_X_test)
#для дальнейших вычислений переиндексирую ДатаФрейм
Ytest = Ytest.reset_index()


x1 = []
y1 = []
#высчитываю среднюю ошибку
for i in range(len(predictions)):
    y1.append(float(Ytest.loc[i,'price'] - predictions[i]))
    x1.append(float(predictions[i]))

#строим график
fig, ax = plt.subplots()
ax.scatter(x1, y1)
plt.title(' RSSE для множественной полиномиальной регрессии')
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
for i in range(2,6,1):
    degree = i
    regression = linear_model.LinearRegression()
    poly = PolynomialFeatures(degree=degree)
    poly_X_train = poly.fit_transform(Xtrn)
    poly_X_test = poly.fit_transform(Xtest)
    model = regression.fit(poly_X_train, Ytrn)
    # записываю предсказания
    predictions = model.predict(poly_X_test)
    RSS = []
    # высчитываю среднюю ошибку
    for i in range(len(predictions)):
        RSS.append((float(Ytest.loc[i, 'price'] - predictions[i])) ** 2)
    RSSE = (np.sum(RSS) / len(predictions)) ** 0.5
    #записываю результаты
    x.append(degree)
    y.append(RSSE)


fig, ax = plt.subplots()
ax.plot(x, y)
plt.title(' RSSE в зависимости от степени регрессии')
ax.set_xlabel('степень')
ax.set_ylabel('RSSE')
plt.show()
