from PyQt5 import QtWidgets as qtw
from PyQt5 import QtGui as qtg
from PyQt5 import QtCore as qtc
import sys
import numpy as np
import cv2
from deeplearning import face_mask_prediction



class VideoCapture(qtc.QThread):
    change_pixmap_signal = qtc.pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self.run_flag = True
        
    def run(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)   # to access camera needs also the cv2.CAP_DSHOW in order to close the camera without errors ONLY FOR WINDOWS
        
        while self.run_flag:
            ret , frame = cap.read()  # read the frames from the camera, BGR format
            prediction_img = face_mask_prediction(frame)  # to connect the deeplearning model succesfully with the video capture - application
            
            
            if ret == True:  # if i got the frames we need to emit them
                self.change_pixmap_signal.emit(prediction_img)
                
        prediction_img = 127+np.zeros((450,600,3),dtype=np.uint8)
        self.change_pixmap_signal.emit(prediction_img)
        cap.release()
        
    def stop(self):
        self.run_flag = False       
        
        self.wait()  # is for the QThread



class mainWindow(qtw.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(qtg.QIcon('./images/icon.png'))  # window icon
        self.setWindowTitle('Face Mask Detection - Jan Swiech - 24522')  # window title 
        self.setFixedSize(600,600)  # size of the window
    
    
        # Adding Widgets
        label = qtw.QLabel('<h3>Mask Detection application</h3>') # bigger number at h (h1,h2,h3,...) the smaller the letters become
        self.cameraButton = qtw.QPushButton('Start Camera',clicked=self.cameraButtonClick, checkable=True)
    
    
    
        # screen label where the video will be shown
        self.screen = qtw.QLabel()
        self.img = qtg.QPixmap(600,500)
        self.img.fill(qtg.QColor('light grey'))
        self.screen.setPixmap(self.img)
        
        
        # layout
        layout = qtw.QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.cameraButton)
        layout.addWidget(self.screen)
        
        self.setLayout(layout)
        self.show()  # use it to show the window on the screen
        
        
    def cameraButtonClick(self):
        print('clicked')
        status = self.cameraButton.isChecked()
        
        if status == True:
            self.cameraButton.setText('Close Camera')
            
            # open the camera
            self.capture = VideoCapture()  # initialize the video capture
            self.capture.change_pixmap_signal.connect(self.updateImage)  #every time i receive the signal from the camera the screen should be updated and the updateImage is a slot that receives the signal from video capture 
            self.capture.start()
            
        elif status == False:
            self.cameraButton.setText('Open Camera')
            
            self.capture.stop()
    
    @qtc.pyqtSlot(np.ndarray) # the decorator for the pyqt
    def updateImage(self,image_array):
       rgb_img = cv2.cvtColor(image_array,cv2.COLOR_BGR2RGB)  # convert the BGR image to RGB
       h,w, ch = rgb_img.shape
       bytes_per_line = ch*w
       
       # convert into QImage
       convertedImage = qtg.QImage(rgb_img.data,w,h,bytes_per_line,qtg.QImage.Format_RGB888)
       scaledImage = convertedImage.scaled(600,580,qtc.Qt.KeepAspectRatio)
       qt_img = qtg.QPixmap.fromImage(scaledImage)
       
       
       # update the image to screen every moment 
       self.screen.setPixmap(qt_img) 
       
    
        
# create simple template
        
if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)
    mw = mainWindow()
    sys.exit(app.exec())
    