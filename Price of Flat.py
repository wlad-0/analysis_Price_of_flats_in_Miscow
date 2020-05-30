import sys
# Импортируем наш интерфейс из файла
from MQt import *
from PyQt5 import QtCore, QtGui, QtWidgets
from pandas import read_csv
from sklearn.linear_model import LinearRegression, LogisticRegression
import pandas as pd
class MyWin(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.MyFunction1)
    def MyFunction1(self):
        df = read_csv('C:/Users/Владислав/Desktop/Diplom/flats_moscow.csv')
        # фильтруем от выбросов
        col_name = 'price'
        q1 = df[col_name].quantile(0.25)
        q3 = df[col_name].quantile(0.75)
        iqr = q3 - q1  # Interquartile range
        fence_low = q1 - 1.5 * iqr
        fence_high = q3 + 1.5 * iqr
        df = df.loc[(df[col_name] > fence_low) & (df[col_name] < fence_high)]
        col_names = ['price', 'totsp', 'livesp', 'kitsp', 'dist']
        df = df.drop(df.columns[0], axis=1)  # удаляем столбец с номером наблюдения
        trg = df[['price']]  # выбираем то что будем предсказывать
        trn = df.drop(df.columns[0], axis=1)  # выбираем то по чему будем предсказывать
                # собираем регрессию
        model = LinearRegression()
        model.fit(trn, trg)

        totsp = self.ui.textEdit.toPlainText()
        livesp = self.ui.textEdit_2.toPlainText()
        kitsp = self.ui.textEdit_3.toPlainText()
        dist = self.ui.textEdit_4.toPlainText()
        metrdist = self.ui.textEdit_5.toPlainText()
        walk = self.ui.textEdit_6.toPlainText()
        brick = self.ui.textEdit_7.toPlainText()
        floor = self.ui.textEdit_8.toPlainText()
        code = self.ui.textEdit_9.toPlainText()
        data = {'totsp': [totsp], 'livesp': [livesp], 'kitsp': [kitsp], 'dist': [dist], 'metrdist': [metrdist],
                'walk': [walk], 'brick': [brick], 'floor': [floor], 'code': [code]}

        data = pd.DataFrame(data=data)

        predictions = model.predict(data)
        predictions = str(predictions[0][0])

        self.ui.textEdit_10.setText(predictions)


if __name__=="__main__":
    app = QtWidgets.QApplication(sys.argv)
    myapp = MyWin()
    myapp.show()
    sys.exit(app.exec_())
