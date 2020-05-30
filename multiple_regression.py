from pandas import read_csv
from sklearn.linear_model import LinearRegression, LogisticRegression
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

col_names = ['price', 'totsp','livesp', 'kitsp','dist']
for i in range(len(col_names)):
    df = remove_outlier(df, col_names[i])
df = df.drop(df.columns[0], axis = 1)  #удаляем столбец с номером наблюдения

#--------------------------------------------------------------
#график корреляции
hm = sns.heatmap(df.corr(), cbar=True, annot=True)
plt.show()
#---------------------------------------------------------------
trg = df[['price']] # выбираем то что будем предсказывать
trn = df.drop(df.columns[0], axis = 1)  # выбираем то по чему будем предсказывать
#высчитываем среднеквадратическое отклонение от цены
print('stdev = ', np.std(trg))
# распределяем выборку на тестовую и тренировочную
Xtrn, Xtest, Ytrn, Ytest = train_test_split(trn, trg, test_size=0.3)
#собираем регрессию
model = LinearRegression()
model.fit(Xtrn, Ytrn)
# высчитываю r^2
score = model.score(Xtest, Ytest)
print('r2 = ',score)
#записываю предсказания
predictions = model.predict(Xtest)
#для дальнейших вычислений переиндексирую ДатаФрейм
Ytest = Ytest.reset_index()

RSS = [] # Массив с квадратами отклонений
#высчитываю среднюю ошибку
for i in range(len(predictions)):
    RSS.append((float(Ytest.loc[i,'price'] - predictions[i]))**2)
RSSE = (np.sum(RSS)/len(predictions))**0.5
print('RSSE = ', RSSE)

#Строю график RSSE
x1 = []
y1 = []
for i in range(len(predictions)):
    y1.append(float(Ytest.loc[i,'price'] - predictions[i]))
    x1.append(float(predictions[i]))
fig, ax = plt.subplots()
ax.scatter(x1, y1)
plt.title(' RSSE для множественной линейной регрессии')
ax.set_xlabel('предсказанная цена')
ax.set_ylabel('разница с реальной ценой')
plt.show()
