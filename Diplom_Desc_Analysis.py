import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
import pylab
sb.set()
def graph1(data):
    #Цена - жилая площадь
    sb.jointplot(x='livesp', y='price', data=data, height = 10)
    plt.show()


def graph2(data):
    sb.boxplot(x="brick", y="price", palette=["m", "g"], data=data)
    plt.title('Boxplot для 0 – кирпичный, монолит, 1 - другие')
    plt.show()

def graph3(data):
    sb.boxplot(x="floor", y="price", palette=["m", "g"], data=data)
    plt.title('Boxplot для 0 – первый и последний этажи, 1 - другие')
    plt.show()

def graph4(data):
    #Цена - общая площадь
    sb.jointplot(x='totsp', y='price', data=data, height=10)
    plt.show()

def graph5(data):
    #Цена - растояние от центра
    sb.jointplot(x='dist', y='price', data=data, height=10)
    plt.show()

def graph6(data):
    #Цена - время на преодоление пути до метро на машине
    data = data[data['walk']==0]
    sb.jointplot(x='metrdist', y='price', data=data, height=10)
    plt.show()

def graph7(data):
    #Цена - время на преодоление пути до метро пешком
    data = data[data['walk']==1]
    sb.jointplot(x='metrdist', y='price', data=data, height=10)
    plt.show()

def graph8(data):
    sb.boxplot(x="code", y="price", palette=["m", "g"], data=data)
    plt.title('Boxplot для разных районов')
    plt.show()

def graph9(data):
    #Цена - площадь кухни
    sb.jointplot(x='kitsp', y='price', data=data, kind='scatter', height=10)
    plt.show()

def graph10(data):
    #Цена - расстояние от центра для сортированных квартир по площади
    data = data.sort_values('totsp').reset_index()
    sb.jointplot(x='dist', y='price', data=data[:500], height = 10)
    pylab.text(-430,168, 'Цена - растояние от центра, 0-500',fontsize=30)
    sb.jointplot(x='dist', y='price', data=data[500:1000], height = 10)
    pylab.text(-480, 250, 'Цена - растояние от центра, 500-1000',fontsize=30)
    sb.jointplot(x='dist', y='price', data=data[1000:1500], height = 10)
    pylab.text(-480, 250, 'Цена - растояние от центра, 1000-1500',fontsize=30)
    sb.jointplot(x='dist', y='price', data=data[1500:], height = 10)
    pylab.text(-450, 700, 'Цена - растояние от центра, 1500-2040',fontsize=30)
    plt.show()

df = pd.read_csv('C:/Users/Владислав/Desktop/Diplom/flats_moscow.csv')
#
def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3 - q1  # Межквартильный размах
    fence_low = q1 - 1.5 * iqr
    fence_high = q3 + 1.5 * iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out
col_names = ['price', 'totsp','livesp', 'kitsp','dist']
for i in range(len(col_names)):
    df = remove_outlier(df, col_names[i])



graph1(df)
graph2(df)
graph3(df)
graph4(df)
graph5(df)
graph6(df)
graph7(df)
graph8(df)
graph9(df)
graph10(df)
