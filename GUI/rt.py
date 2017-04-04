#!/home/paul/anaconda2/bin/python2.7

import sys
from PyQt4 import QtCore, QtGui
import NerNet
import numpy as np

class ScribbleArea(QtGui.QWidget):
    def __init__(self, parent = None):
        QtGui.QWidget.__init__(self, parent)

        self.setAttribute(QtCore.Qt.WA_StaticContents)
        self.modified = False
        self.scribbling = False
        self.myPenWidth = 1
        self.myPenColor = QtCore.Qt.blue
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
            self.resizeImage(self.image, QtCore.QSize(newWidth, newHeight))
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
        retImage = self.image.scaled(newSize)
        buf = np.ndarray((28, 28))
        for i in range(28):
            for j in range(28):
                gray = QtGui.qGray(retImage.pixel(i, j))
                retImage.setPixel(i, j, QtGui.QColor(gray, gray, gray).rgb())
                buf[i][j] = 1-gray
        retImage.save("result.png")
        print buf/255
    
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
        self.aboutBtn = QtGui.QPushButton()
        self.fitBtn = QtGui.QPushButton()
        self.clearBtn = QtGui.QPushButton()

        self.exitIcon = QtGui.QIcon('icons/exit.png')
        self.fitIcon = QtGui.QIcon('icons/repeat.png')
        self.runIcon = QtGui.QIcon('icons/play-button.png')
        self.aboutIcon = QtGui.QIcon('icons/info.png')
        self.clearIcon = QtGui.QIcon('icons/garbage.png')

        self.exitBtn.setIcon(self.exitIcon)
        self.fitBtn.setIcon(self.fitIcon)
        self.runBtn.setIcon(self.runIcon)
        self.aboutBtn.setIcon(self.aboutIcon)
        self.clearBtn.setIcon(self.clearIcon)

        self.ansLabel = QtGui.QLabel("Answer")
        self.sureLabel = QtGui.QLabel("Sure")

        self.buttonBox.addWidget(self.exitBtn)
        self.buttonBox.addWidget(self.fitBtn)
        self.buttonBox.addWidget(self.runBtn)
        self.buttonBox.addWidget(self.aboutBtn)
        self.buttonBox.addWidget(self.clearBtn)

        self.layout.addLayout(self.buttonBox)
        self.layout.addWidget(self.scribbleArea)
        self.layout.addWidget(self.ansLabel)
        self.layout.addWidget(self.sureLabel)
        self.layout.setStretch(1, 10)
        self.layout.setStretch(2, 1)
        self.layout.setStretch(3, 1)


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
        self.miptLabel = QtGui.QLabel("""Project of digit-recognition is made by students of Department of Radioengineering and Cybernetics 
of Moscow Institute of Physics and Technology - Pavel Zakharov and Nikita Mokrov.\n""")
        
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

class MainWindow(QtGui.QMainWindow):
    def __init__(self, parent = None):
        QtGui.QMainWindow.__init__(self, parent)
        self.saveAsActs = []
        self.centWidget = FormWidget(self)
        self.setCentralWidget(self.centWidget)

        self.createActions()
        self.createMenus()

        self.setWindowTitle(self.tr("Digit recognition v1"))
        self.resize(500, 500)

        self.connect(self.centWidget.exitBtn, QtCore.SIGNAL('clicked()'), self, QtCore.SLOT('close()'))
        self.centWidget.fitBtn.clicked.connect(self.initializeNetwork)
        self.centWidget.runBtn.clicked.connect(self.run)
        self.centWidget.aboutBtn.clicked.connect(self.about)
        self.centWidget.clearBtn.clicked.connect(self.centWidget.scribbleArea.clearImage)

        self.networkReady = 0
        self.progBar = QtGui.QProgressBar()
        self.statusBar().addWidget(self.progBar)
        self.labl = QtGui.QLabel()
        self.statusBar().addPermanentWidget(self.labl)
        self.setlabl()

        self.timer = QtCore.QBasicTimer()
        self.initializeNetwork()

    def setlabl(self):
        if self.networkReady == 100:
            self.labl.setText("Ready")
        else:
            self.labl.setText("Initializing..")

    def run(self):
        print "Run"
        self.centWidget.scribbleArea.prepareImage()

    def initializeNetwork(self):
        if self.timer.isActive():
            self.timer.stop()
        else:
            self.timer.start(100, self)
            self.centWidget.runBtn.setEnabled(False)
            self.centWidget.fitBtn.setEnabled(False)
    """???????????????????????????????????????????????"""
    def timerEvent(self, event): 
        if self.networkReady >= 100:
            self.timer.stop()
            self.networkReady = 0
            self.centWidget.runBtn.setEnabled(True)
            self.centWidget.fitBtn.setEnabled(True)
            return
        self.networkReady = self.networkReady + 1
        self.progBar.setValue(self.networkReady)
        self.setlabl()

    def open(self):
        if self.maybeSave():
            fileName = QtGui.QFileDialog.getOpenFileName(self,
                                                         self.tr("Open File"),
                                                         QtCore.QDir.currentPath())
            if not fileName.isEmpty():
                self.centWidget.scribbleArea.openImage(fileName)

    def penColor(self):
        newColor = QtGui.QColorDialog.getColor(self.centWidget.scribbleArea.penColor())
        if newColor.isValid():
            self.centWidget.scribbleArea.setPenColor(newColor)

    def penWidth(self):
        newWidth, ok = QtGui.QInputDialog.getInteger(self, self.tr("Scribble"),
                                               self.tr("Select pen width:"),
                                               self.centWidget.scribbleArea.penWidth(),
                                               1, 50, 1)
        if ok:
            self.centWidget.scribbleArea.setPenWidth(newWidth)

    def about(self):
        self.aboutWindow = AboutWidget()

    def createActions(self):
        self.openAct = QtGui.QAction(self.tr("&Open..."), self)
        self.openAct.setShortcut(self.tr("Ctrl+O"))
        self.connect(self.openAct, QtCore.SIGNAL("triggered()"), self.open)
        
        self.exitAct = QtGui.QAction(self.tr("E&xit"), self)
        self.exitAct.setShortcut(self.tr("Ctrl+Q"))
        self.connect(self.exitAct, QtCore.SIGNAL("triggered()"), self, QtCore.SLOT("close()"))

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

    def createMenus(self):
        self.fileMenu = QtGui.QMenu(self.tr("&File"), self)
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)

        self.optionMenu = QtGui.QMenu(self.tr("&Options"), self)
        self.optionMenu.addAction(self.penColorAct)
        self.optionMenu.addAction(self.penWidthAct)
        self.optionMenu.addSeparator()
        self.optionMenu.addAction(self.clearScreenAct)

        self.helpMenu = QtGui.QMenu(self.tr("&Help"), self)
        self.helpMenu.addAction(self.aboutAct)
        self.helpMenu.addAction(self.aboutQtAct)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.optionMenu)
        self.menuBar().addMenu(self.helpMenu)

if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


