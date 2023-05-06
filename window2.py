# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'window2.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from Cricket_shot_Classifier import stance
import test_rc
class Ui_SecondWindow(object):
    def setupUi(self, SecondWindow):
        SecondWindow.setObjectName("SecondWindow")
        SecondWindow.resize(608, 423)
        SecondWindow.setMinimumSize(QtCore.QSize(608, 423))
        SecondWindow.setMaximumSize(QtCore.QSize(608, 423))
        SecondWindow.setStyleSheet("QMainWindow > QWidget {background-image: url(:/bgImage/EtS-1cOXMAAaZgJ.jpg);}\n""\n""\n""\n""")
        self.centralwidget = QtWidgets.QWidget(SecondWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 0, 611, 321))
        self.label.setAutoFillBackground(False)
        self.label.setStyleSheet("background-image: url(:/bgImage/Cricket-shots-Forward-Defence-Shot-v2-1024x464.jpg);\n""")
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap(":/bgImage/Cricket-shots-Forward-Defence-Shot-v2-1024x464.jpg"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget,clicked=lambda:self.buttonClicked())
        self.pushButton.setGeometry(QtCore.QRect(210, 330, 191, 81))
        font = QtGui.QFont()
        font.setFamily("Playbill")
        font.setPointSize(48)
        self.pushButton.setFont(font)
        self.pushButton.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.pushButton.setStyleSheet("color: Black;\n"
                                        "background-color: White;\n"
                                        "border-style: outset;\n"
                                        "border-radius:15px;\n"
                                        "border-width:4px;\n"
                                        "border-color: Red;")
        self.pushButton.setObjectName("pushButton")
        SecondWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(SecondWindow)
        QtCore.QMetaObject.connectSlotsByName(SecondWindow)
        
    def buttonClicked(self):
         stance()

    def retranslateUi(self, SecondWindow):
        _translate = QtCore.QCoreApplication.translate
        SecondWindow.setWindowTitle(_translate("SecondWindow", "MainWindow"))
        self.pushButton.setText(_translate("SecondWindow", "Practice"))



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    SecondWindow = QtWidgets.QMainWindow()
    ui = Ui_SecondWindow()
    ui.setupUi(SecondWindow)
    SecondWindow.show()
    sys.exit(app.exec_())
