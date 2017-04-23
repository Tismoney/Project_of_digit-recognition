#!/home/paul/anaconda2/bin/python2.7

import sys
from PyQt4 import QtCore, QtGui
import PyQt4
import NerNet
import numpy as np

class ScribbleArea(QtGui.QWidget):
    def __init__(self, parent = None):
        QtGui.QWidget.__init__(self, parent)

        self.setAttribute(QtCore.Qt.WA_StaticContents)
        self.modified = False
        self.scribbling = False
        self.myPenWidth = 30
        self.myPenColor = QtCore.Qt.black
        self.image = QtGui.QImage()
        self.lastPoint = QtCore.QPoint()

    def openImage(self, fileName):
        loadedImage = QtGui.QImage()
        if not loadedImage.load(fileName):
            return False

        newSize = loadedImage.size().expandedTo(size())
        self.resizeImage(loadedImage, newSize)
        self.image = loadedImage
        self.modified = False
        self.update()
        return True

    def saveImage(self, fileName, fileFormat):
        visibleImage = self.image
        self.resizeImage(visibleImage, size())

        if visibleImage.save(fileName, fileFormat):
            self.modified = False
            return True
        else:
            return False

    def setPenColor(self, newColor):
        self.myPenColor = newColor

    def setPenWidth(self, newWidth):
        self.myPenWidth = newWidth

    def clearImage(self):
        self.image.fill(QtGui.qRgb(255, 255, 255))
        self.modified = True
        self.update()

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.lastPoint = event.pos()
            self.scribbling = True

    def mouseMoveEvent(self, event):
        if (event.buttons() & QtCore.Qt.LeftButton) and self.scribbling:
            self.drawLineTo(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.scribbling:
            self.drawLineTo(event.pos())
            self.scribbling = False

    def paintEvent(self, event):
        painter = QtGui.QPainter()
        painter.begin(self)
        painter.drawImage(QtCore.QPoint(0, 0), self.image)
        painter.end()

    def resizeEvent(self, event):
        if self.width() > self.image.width() or self.height() > self.image.height():
            newWidth = max(self.width() + 128, self.image.width())
            newHeight = max(self.height() + 128, self.image.height())
            self.resizeImage(self.image, QtCore.QSize(400, 400))
            self.update()

        QtGui.QWidget.resizeEvent(self, event)

    def drawLineTo(self, endPoint):
        painter = QtGui.QPainter()
        painter.begin(self.image)
        painter.setPen(QtGui.QPen(self.myPenColor, self.myPenWidth,
                                  QtCore.Qt.SolidLine, QtCore.Qt.RoundCap,
                                  QtCore.Qt.RoundJoin))
        painter.drawLine(self.lastPoint, endPoint)
        painter.end()
        self.modified = True

        rad = self.myPenWidth / 2 + 2
        self.update(QtCore.QRect(self.lastPoint, endPoint).normalized()
                                         .adjusted(-rad, -rad, +rad, +rad))
        self.lastPoint = QtCore.QPoint(endPoint)

    def resizeImage(self, image, newSize):
        if image.size() == newSize:
            return

        newImage = QtGui.QImage(newSize, QtGui.QImage.Format_RGB32)
        newImage.fill(QtGui.qRgb(255, 255, 255))
        painter = QtGui.QPainter()
        painter.begin(newImage)
        painter.drawImage(QtCore.QPoint(0, 0), image)
        painter.end()
        self.image = newImage

    def prepareImage(self):
        newSize = QtCore.QSize(28, 28)
        retImage = self.image.scaled(newSize, transformMode = QtCore.Qt.SmoothTransformation) #transformMode = QtCore.Qt.SmoothTransformation
        buf = np.ndarray((1, 1, 28, 28))
        for i in xrange(28):
            for j in xrange(28):
                gray = QtGui.qGray(retImage.pixel(i, j))
                retImage.setPixel(i, j, QtGui.QColor(gray, gray, gray).rgb())
                buf[0][0][j][i] = 1-gray/255.
        retImage.save("result.png")
        return buf
    
    def isModified(self):
        return self.modified

    def penColor(self):
        return self.myPenColor

    def penWidth(self):
        return self.myPenWidth

class FormWidget(QtGui.QWidget):
 
    def __init__(self, parent):        
        super(FormWidget, self).__init__(parent)
        self.scribbleArea = ScribbleArea()

        self.layout = QtGui.QVBoxLayout(self)
        self.buttonBox = QtGui.QHBoxLayout(self)

        self.exitBtn = QtGui.QPushButton()
        self.runBtn = QtGui.QPushButton()
        self.helpBtn = QtGui.QPushButton()
        self.fitBtn = QtGui.QPushButton()
        self.clearBtn = QtGui.QPushButton()

        self.exitIcon = QtGui.QIcon('icons/exit.png')
        self.fitIcon = QtGui.QIcon('icons/repeat.png')
        self.runIcon = QtGui.QIcon('icons/play-button.png')
        self.helpIcon = QtGui.QIcon('icons/info.png')
        self.clearIcon = QtGui.QIcon('icons/garbage.png')

        self.exitBtn.setIcon(self.exitIcon)
        self.fitBtn.setIcon(self.fitIcon)
        self.runBtn.setIcon(self.runIcon)
        self.helpBtn.setIcon(self.helpIcon)
        self.clearBtn.setIcon(self.clearIcon)

        self.exitBtn.setToolTip("Exit")
        self.fitBtn.setToolTip("Fit")
        self.runBtn.setToolTip("Run")
        self.helpBtn.setToolTip("Help")
        self.clearBtn.setToolTip("Clear")

        self.ansLabel = QtGui.QLabel("Answer")
        self.sureLabel = QtGui.QLabel("Sure")

        self.buttonBox.addWidget(self.exitBtn)
        self.buttonBox.addWidget(self.fitBtn)
        self.buttonBox.addWidget(self.runBtn)
        self.buttonBox.addWidget(self.helpBtn)
        self.buttonBox.addWidget(self.clearBtn)

        self.layout.addLayout(self.buttonBox)
        self.layout.addWidget(self.scribbleArea)
        self.layout.addWidget(self.ansLabel)
        self.layout.addWidget(self.sureLabel)
        self.layout.setStretch(1, 10)
        self.layout.setStretch(2, 1)
        self.layout.setStretch(3, 1)

class ChangeNetWindow(QtGui.QWidget):
    def __init__(self, network, parent = None):
        QtGui.QWidget.__init__(self, parent)
        self.setWindowTitle("Choose architecture")
        self.layout = QtGui.QVBoxLayout()
        self.prepareButtons(network)
        self.layout.addWidget(QtGui.QLabel("Please, choose architecture of a neural net"))
        self.layout.addWidget(self.dense100Btn)
        self.layout.addWidget(self.dense800Btn)
        self.layout.addWidget(self.convBtn)
        self.setLayout(self.layout)
        self.show()
    
    def prepareButtons(self, network):
        self.btnBox = QtGui.QButtonGroup()
        self.dense800Btn = QtGui.QRadioButton("Dense-800")
        self.dense100Btn = QtGui.QRadioButton("Dense-200-Dense-100-Dense-50")
        self.convBtn = QtGui.QRadioButton("Conv-32-MaxPool-Dense-256")
        self.btnBox.addButton(self.dense100Btn, 0)
        self.btnBox.addButton(self.dense800Btn, 1)
        self.btnBox.addButton(self.convBtn, 2)
        self.dense800Btn.setCheckable(True)
        self.dense100Btn.setCheckable(True)
        self.convBtn.setCheckable(True)
        if network == 0:
            self.dense100Btn.setChecked(True)
        elif network == 1:
            self.dense800Btn.setChecked(True)
        else:
            self.convBtn.setChecked(True)


class AboutWidget(QtGui.QWidget):
    def __init__(self, parent = None):
        QtGui.QWidget.__init__(self, parent)
        self.setWindowTitle("About")

        self.layout = QtGui.QHBoxLayout()
        self.prepareLayout()
        
        self.setLayout(self.layout)
        self.show()

    def createImageLabels(self):
        self.miptPixmap = QtGui.QPixmap("icons/eng.jpg").scaledToWidth(100)
        self.miptImage = QtGui.QLabel()
        self.miptImage.setPixmap(self.miptPixmap)

        self.flaticonPixmap = QtGui.QPixmap("icons/flaticon-logo-footer.svg").scaledToWidth(100)
        self.flaticonImage = QtGui.QLabel()
        self.flaticonImage.setPixmap(self.flaticonPixmap)

        self.qtPixmap = QtGui.QPixmap("icons/PyQt.png").scaledToWidth(100)
        self.qtImage = QtGui.QLabel()
        self.qtImage.setPixmap(self.qtPixmap)

    def createLabels(self):
        self.miptLabel = QtGui.QLabel("""Project of digit-recognition is made by students of Department of Radioengineering 
        and Cybernetics of Moscow Institute of Physics and Technology - Pavel Zakharov and Nikita Mokrov.\n""")
        
        self.flaticonLabel = QtGui.QLabel("""<div>Icons made by <a href="http://www.flaticon.com/authors/madebyoliver" title="Madebyoliver">Madebyoliver</a>
         from <a href="http://www.flaticon.com" title="Flaticon">www.flaticon.com</a> is licensed by <a href="http://creativecommons.org/licenses/by/3.0/" 
         title="Creative Commons BY 3.0" target="_blank">CC 3.0 BY</a></div>\n""")
        self.flaticonLabel.setOpenExternalLinks(True)
        
        self.theanoLabel = QtGui.QLabel("""Neural nets is based on <a href="http://deeplearning.net/software/theano/index.html">Theano</a> and <a href="https://lasagne.readthedocs.io/en/latest/">Lasagne</a> libraries:\n""")
        self.theanoLabel.setOpenExternalLinks(True)
        
        self.qtLabel = QtGui.QLabel("""This project uses PyQt4. More about Qt library in corresponding dialog\n""")

    def prepareLayout(self):
        self.createLabels()
        self.createImageLabels()

        self.leftLayout = QtGui.QVBoxLayout()
        self.rightLayout = QtGui.QVBoxLayout()
    
        self.leftLayout.addWidget(self.miptLabel)
        self.rightLayout.addWidget(self.miptImage)
        
        self.leftLayout.addWidget(self.theanoLabel)
        
        self.leftLayout.addWidget(self.flaticonLabel)
        self.rightLayout.addWidget(self.flaticonImage)
        
        self.leftLayout.addWidget(self.qtLabel)
        self.rightLayout.addWidget(self.qtImage)

        self.layout.addLayout(self.leftLayout)
        self.layout.addLayout(self.rightLayout)

class HelpWidget(QtGui.QWidget):
    def __init__(self, parent = None):
        QtGui.QWidget.__init__(self, parent)
        self.setWindowTitle("Help")

        self.nerNetPixmap = QtGui.QPixmap("icons/NerNet.jpeg").scaledToWidth(400)
        self.nerNetImage = QtGui.QLabel()
        self.nerNetImage.setPixmap(self.nerNetPixmap)
        self.nerNetImage.setAlignment(QtCore.Qt.AlignHCenter)

        self.label1 = QtGui.QLabel("""<b>What am I dealing with?</b><br>
            This is a program that recognizes written number from 0 to 9. It is based on <a href="https://en.wikipedia.org/wiki/Artificial_neural_network">neural net</a>,
            that creates recognizing pattern using pictures with known number.
            This process is called "fitting". Before using a net, it should be fitted.<br>
            <b>Note! </b>Fitting is a really slow process, so don't be afraid to wait for a couple of minutes for the first time. Later program will use acquired data, so fitting will 
            be much faster.""")
        self.label1.setWordWrap(True);
        self.label2 = QtGui.QLabel("""Before using a net, fit it, clicking a corresponding button. After finishing you may draw a number and click "Run".
You will get a predicted number and probability that the prediction is right. (Yes, we cannot predict for sure. Actually, nobody can). "Options" menu enables you to change pen color, width and net architecture.""")
        self.label2.setWordWrap(True);
        self.layout = QtGui.QVBoxLayout()

        self.layout.addWidget(self.label1)
        self.layout.addWidget(self.nerNetImage)
        self.layout.addWidget(self.label2)
        
        self.setLayout(self.layout)
        self.show()

class MainWindow(QtGui.QMainWindow):
    def __init__(self, parent = None):
        QtGui.QMainWindow.__init__(self, parent)
        self.saveAsActs = []
        self.centWidget = FormWidget(self)
        self.setCentralWidget(self.centWidget)
        self.test = None
        self.nerNetArchitecture = 0 #0 is for dense-100, 1 for dense-800, 2 is for conv

        self.createActions()
        self.createMenus()

        self.setWindowTitle(self.tr("Digit recognition v1"))
        self.resize(600, 600)

        self.connect(self.centWidget.exitBtn, QtCore.SIGNAL('clicked()'), self, QtCore.SLOT('close()'))
        self.centWidget.fitBtn.clicked.connect(self.initializeNetwork)
        self.centWidget.runBtn.clicked.connect(self.run)
        self.centWidget.helpBtn.clicked.connect(self.help)
        self.centWidget.clearBtn.clicked.connect(self.centWidget.scribbleArea.clearImage)

        self.networkReady = 0
        self.progBar = QtGui.QProgressBar()
        self.statusBar().addWidget(self.progBar)
        self.labl = QtGui.QLabel()
        self.statusBar().addPermanentWidget(self.labl)
        self.setlabl()

        self.net = NerNet.NerNet()
        self.net.init_data()
        self.net.signalConnect(self.epochEvent)
        #self.initializeNetwork()

    def setlabl(self):
        if self.networkReady == 100:
            self.labl.setText("Ready")
        else:
            self.labl.setText("Initializing..")

    def run(self):
        print "Run"
        self.net.get_result(self.centWidget.scribbleArea.prepareImage())

    def initializeNetwork(self):
        #Fit a net
        #self.net.get_accuracy() - init area
        self.networkReady = 0
        self.centWidget.runBtn.setEnabled(False)
        self.centWidget.fitBtn.setEnabled(False)
        self.net.make_and_check(path = "Weight/Conv")
    
    def epochEvent(self, epoch):
        if epoch == self.net.num_epochs:#10
            self.centWidget.runBtn.setEnabled(True)
            self.centWidget.fitBtn.setEnabled(True)
            self.progBar.setValue(100)
            self.setlabl()
            self.networkReady = 0
            return
        self.networkReady = self.networkReady + 100 / self.net.num_epochs #10
        self.progBar.setValue(self.networkReady)
        self.setlabl()

    def penColor(self):
        newColor = QtGui.QColorDialog.getColor(self.centWidget.scribbleArea.penColor())
        if newColor.isValid():
            self.centWidget.scribbleArea.setPenColor(newColor)

    def penWidth(self):
        newWidth, ok = QtGui.QInputDialog.getInteger(self, self.tr("Width"),
                                               self.tr("Select pen width:"),
                                               self.centWidget.scribbleArea.penWidth(),
                                               1, 50, 1)
        if ok:
            self.centWidget.scribbleArea.setPenWidth(newWidth)

    def about(self):
        self.aboutWindow = AboutWidget()

    def help(self):
        self.helpWindow = HelpWidget()

    def changeNetwork(self):
        self.changeNetWindow = ChangeNetWindow(self.nerNetArchitecture)
        self.changeNetWindow.btnBox.buttonClicked.connect(self.setNetwork)

    def setNetwork(self):
        self.nerNetArchitecture = self.changeNetWindow.btnBox.checkedId()

    def createActions(self):
        self.penColorAct = QtGui.QAction(self.tr("&Pen Color..."), self)
        self.connect(self.penColorAct, QtCore.SIGNAL("triggered()"),
                     self.penColor)

        self.penWidthAct = QtGui.QAction(self.tr("Pen &Width..."), self)
        self.connect(self.penWidthAct, QtCore.SIGNAL("triggered()"),
                     self.penWidth)

        self.clearScreenAct = QtGui.QAction(self.tr("&Clear Screen"), self)
        self.clearScreenAct.setShortcut(self.tr("Ctrl+L"))
        self.connect(self.clearScreenAct, QtCore.SIGNAL("triggered()"),
                     self.centWidget.scribbleArea.clearImage)

        self.aboutAct = QtGui.QAction(self.tr("&About"), self)
        self.connect(self.aboutAct, QtCore.SIGNAL("triggered()"), self.about)

        self.aboutQtAct = QtGui.QAction(self.tr("About &Qt"), self)
        self.connect(self.aboutQtAct, QtCore.SIGNAL("triggered()"),
                     QtGui.qApp, QtCore.SLOT("aboutQt()"))
        self.changeNetworkAct = QtGui.QAction(self.tr("&Change Network"), self)
        self.connect(self.changeNetworkAct, QtCore.SIGNAL("triggered()"), self.changeNetwork)

    def createMenus(self):
        self.optionMenu = QtGui.QMenu(self.tr("&Options"), self)
        self.optionMenu.addAction(self.penColorAct)
        self.optionMenu.addAction(self.penWidthAct)
        self.optionMenu.addAction(self.changeNetworkAct)
        self.optionMenu.addSeparator()
        self.optionMenu.addAction(self.clearScreenAct)

        self.helpMenu = QtGui.QMenu(self.tr("&Help"), self)
        self.helpMenu.addAction(self.aboutAct)
        self.helpMenu.addAction(self.aboutQtAct)

        self.menuBar().addMenu(self.optionMenu)
        self.menuBar().addMenu(self.helpMenu)

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


