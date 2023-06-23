#PyQt5 라이브러리
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

#open cv에서는 영상을 프레임단위로 가져오기 때문에 sleep을 통해서 프레임을 연결시켜주어 영상으로 보이게 만드는 것임
import threading
#화면을 윈도우에 띄우기 위해 sys접근
import sys

#open cv 라이브러리
import cv2

#Read / Write INI
import configparser

import pymcprotocol

QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)

file_Path = 'C:\\HSENG\\2023 06 21\\2023 06 21\\00. Project\\01. Lead Vision System\\'
Main_UI = uic.loadUiType(file_Path + "RcutDetection.ui")[0]
Sub_UI = uic.loadUiType(file_Path + "subUI.ui")[0]

"""
# PLC 통신
pymc3e = pymcprotocol.Type3E()
pymc3e.connect("192.0.1.1", 1025)
wordunits_values = pymc3e.batchread_wordunits(headdevice="D100", readsize=1)
pymc3e.batchwrite_wordunits(headdevice="D1000", values=[0, 10, 20, 30, 40])
"""
#======================================================================================= readINI()
def readINI():
    config = configparser.ConfigParser()
    config.read(file_Path + 'System.ini')

    app = config['ROI_1']
    top_1 = app['top']
    left_1 = app['left']
    width_1 = app['width']
    height_1 = app['height']
    val_1 = app['val_1']

    app = config['ROI_2']
    top_2 = app['top']
    left_2 = app['left']
    width_2 = app['width']
    height_2 = app['height']
    val_2 = app['val_2']
    return top_1, left_1, width_1, height_1, val_1, top_2, left_2, width_2, height_2, val_2
#========================================================================================== writeINI()
def writeINI(section, options, values):
    File_Path = file_Path + 'system.ini'
    config = configparser.ConfigParser()
    config.read(File_Path)

    if not config.has_section(section):
        config.add_section(section)

    for i in range(len(options)):
        config.set(section, options[i], str(values[i]))
        print(options[i], str(values[i]))

    with open(File_Path, 'w') as configfile:
        config.write(configfile)

#============================================================================= class Ui_MainWindow(QMainWindow, Main_UI):
class Ui_MainWindow(QMainWindow, Main_UI):
    global isDragging, x0, y0, img
    isDragging = False                         # 마우스 드래그 상태 저장
    x0,y0,w,h = -1,-1,-1,-1                    # 영역 선택 좌표 저장

    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setupUi(self)
        self.initUI()

        self.red, self.blue, self.yellow = (255, 0, 0), (0, 0, 255), (233,233,0)

        # ROI 영역 변수 초기화
        self.roi_start = None
        self.roi_end = None
        self.setMouseTracking(True)

    def initUI(self):
                
        # Button Test EXE
        self.btnSNAP = QPushButton('TEST', self)
        self.btnSNAP.setGeometry(QtCore.QRect(1850, 880, 130, 90))
        self.btnSNAP.setStyleSheet("background-color: rgb(93, 93, 93); font: 75 22pt 'Georgia'; color: rgb(236, 236, 236);")
        self.btnSNAP.clicked.connect(self.btnTEST_Clicked)  # Connect the clicked signal to the btnTEST_Clicked slot

        # Button PLC ON
        self.btnPLCON = QPushButton('PLC ON', self)
        self.btnPLCON.setGeometry(QtCore.QRect(1800, 880, 130, 40))
        self.btnPLCON.setStyleSheet("background-color: rgb(93, 93, 93); font: 75 17pt 'Georgia'; color: rgb(236, 236, 236);")
        self.btnPLCON.clicked.connect(self.btnPLCON_Clicked)  # Connect the clicked signal to the btnPLCON_Clicked slot

        # Check Thread Stop
        self.ckStop = QCheckBox('Thread Stop', self)
        self.ckStop.setGeometry(QtCore.QRect(1800, 880, 130, 40))
        self.ckStop.setStyleSheet("background-color: rgb(93, 93, 93); font: 75 17pt 'Georgia'; color: rgb(236, 236, 236);")
        self.ckStop.clicked.connect(self.ckStop_Clicked)  # Connect the clicked signal to the btnPLCON_Clicked slot

    def btnTEST_Clicked(self):
        self.StartInspection()

    def btnPLCON_Clicked(self):
        print("PLC ON")

    def ckStop_Clicked(self):
        if self.ckStop.isChecked:
            ret, frame = cap.read()
            frame = self.drawSquare(frame)
            #self.video_pause()
        else:
            self.video_resume()
    
    def showTargetImage(self):

        pixmap = QPixmap(file_Path + "Template_1.JPG")
        self.lblTarget_L.setPixmap(QPixmap(pixmap))
        self.lblTarget_L.setAlignment(Qt.AlignCenter)

        pixmap = QPixmap(file_Path + "Template_2.JPG")
        self.lblTarget_R.setPixmap(QPixmap(pixmap))
        self.lblTarget_R.setAlignment(Qt.AlignCenter)

    def insertROIvalue(self):
        global roiX1_L, roiY1_L, roiX2_L, roiY2_L, roiX1_R, roiY1_R, roiX2_R, roiY2_R
        global top_left1, bottom_right1, top_left2, bottom_right2
        global valROI_L, valROI_R

        roiX1_L, roiY1_L, roiX2_L, roiY2_L, valROI_L = 0,0,0,0,0
        roiX1_R, roiY1_R, roiX2_R, roiY2_R, valROI_R = 0,0,0,0,0

        roiX1_L, roiY1_L, roiX2_L, roiY2_L, valROI_L, roiX1_R, roiY1_R, roiX2_R, roiY2_R, valROI_R = readINI()

        self.lblX_1.setText(roiX1_L)
        self.lblY_1.setText(roiY1_L)
        self.lblW_1.setText(roiX2_L)
        self.lblH_1.setText(roiY2_L)

        self.lblSpec_L.setText((str(float(valROI_L) * 100)) + '%')

        self.lblX_2.setText(roiX1_R)
        self.lblY_2.setText(roiY1_R)
        self.lblW_2.setText(roiX2_R)
        self.lblH_2.setText(roiY2_R)

        self.lblSpec_R.setText(((str(float(valROI_R) * 100)) + '%'))

        roiX1_L = int(roiX1_L)
        roiY1_L = int(roiY1_L)
        roiX2_L = int(roiX2_L)
        roiY2_L = int(roiY2_L)
        valROI_L = float(valROI_L)

        top_left1 = (roiX1_L, roiY1_L)
        bottom_right1 = ( roiX2_L, roiY2_L)

        roiX1_R = int(roiX1_R)
        roiY1_R = int(roiY1_R)
        roiX2_R = int(roiX2_R)
        roiY2_R = int(roiY2_R)
        valROI_R = float(valROI_R)

        top_left2 = (roiX1_R, roiY1_R)
        bottom_right2 = (roiX2_R, roiY2_R)
    
    def drawSquare(self, frame):

        cv2.rectangle(frame, top_left1, bottom_right1, (0, 255, 0), 2)
        cv2.rectangle(frame, top_left2, bottom_right2, (0, 255, 0), 2)

        return frame
    
    def Video_to_frame(self, MainWindow):
        self.showTargetImage()
        self.insertROIvalue()
        global cap
        global frame
        
        cap = cv2.VideoCapture(0)
        # 해상도 강제 조정
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break

            frame = self.drawSquare(frame)
            rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgbImage.shape       #(RGB 이미지) 의 높이( h), 너비( w) 및 채널 수( ch) 를 검색
            bytes_per_line = ch * w         # 채널 수에 너비를 곱하여 이미지의 라인당 바이트 수를 계산
            q_image = QImage(rgbImage.data, w, h, bytes_per_line, QImage.Format_RGB888)

            #  QLabel 위젯()의 픽스맵을 lblMainDisp로 표시되는 이미지로 설정
            self.lblMainDisp.setPixmap(QPixmap.fromImage(q_image))      
            #  QLabel 위젯 scaledContents의 속성을 로 설정하여 라벨 크기에 맞게 이미지의 자동 크기 조정을 가능하게
            self.lblMainDisp.setScaledContents(True)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    
    # video_to_frame을 쓰레드로 사용
    #이게 영상 재생 쓰레드 돌리는거 얘를 조작하거나 함수를 생성해서 연속재생 관리해야할듯
    def video_thread(self):
        thread = threading.Thread(target=self.Video_to_frame, args=(self,))
        thread.daemon = True  # 프로그램 종료시 프로세스도 함께 종료 (백그라운드 재생 X)
        thread.start()
    def video_pause(self):
        self.paused = True
        print("Thread paused")
    def video_resume(self):
        self.paused = False
        with self.pause_cond:
            self.pause_cond.notify()
        print("Thread resumed")
    #===========================================================

    def mousePressEvent(self, event):
        print ("mousePressEvent =================================")
        if event.button() == Qt.LeftButton:
            self.isDragging = True
            self.x0, self.y0 = event.x(), (event.y() + 50)
            print (self.x0, self.y0 , "+++++++++++++++++++++++++++++")

    def mouseReleaseEvent(self, event):
        print ("mouseReleaseEvent =================================")
        if event.button() == Qt.LeftButton and self.isDragging:
            self.isDragging = False
            x, y = event.x(), (event.y() + 50)
            print (x, y)
            self.w, self.h = x - self.x0, y - self.y0
            if self.w > 0 and self.h > 0:
                
                if x<1000:      # ROI (L)
                    cv2.rectangle(frame, (self.x0, self.y0), (x, y), self.blue, 2)
                    ret = QMessageBox.question(self, 'Save', "Do you want to save this ROI?", QMessageBox.Yes | QMessageBox.No )
                    if ret == QMessageBox.Yes:
                        roiX1_L, roiY1_L, roiX2_L, roiY2_L = (self.x0 ), (self.y0), (x), (y)
                        
                        self.lblX_1.setText(str(roiX1_L))
                        self.lblY_1.setText(str(roiY1_L))
                        self.lblW_1.setText(str(roiX2_L))
                        self.lblH_1.setText(str(roiY2_L))
                        writeINI('ROI_1', ('top', 'left', 'width', 'height'), (roiX1_L, roiY1_L, roiX2_L, roiY2_L))
                        self.insertROIvalue()
                else:           # ROI (R)
                    cv2.rectangle(frame, (self.x0, self.y0), (x, y), self.red, 2)
                    ret = QMessageBox.question(self, 'Save', "Do you want to save this ROI?", QMessageBox.Yes | QMessageBox.No )
                    if ret == QMessageBox.Yes:
                        roiX1_R, roiY1_R, roiX2_R, roiY2_R = (self.x0), (self.y0), (x), (y)

                        self.lblX_2.setText(str(roiX1_R))
                        self.lblY_2.setText(str(roiY1_R))
                        self.lblW_2.setText(str(roiX2_R))
                        self.lblH_2.setText(str(y))
                        writeINI('ROI_2', ('top', 'left', 'width', 'height'), [roiX1_R, roiY1_R, roiX2_R, roiY2_R])
                        self.insertROIvalue()
                print (self.x0, self.y0, x, y )
            else:
                print('Drag the area from the top left to the bottom right.')
    
    #===================================================================================

    def StartInspection(self):
        # Load input image and convert to grayscale
        #cap = cv2.VideoCapture("rtsp://192.168.1.20:554/stream1")
        #ret, frame = cap.read()
        cv2.imwrite(file_Path + 'Lead1.JPG', frame)
        img = cv2.imread(file_Path + 'Lead1.JPG')  #, cv2.IMREAD_GRAYSCALE)

        # Load template images and convert to grayscale
        template_1 = cv2.imread(file_Path + 'Template_1.JPG')  #, cv2.IMREAD_GRAYSCALE
        th1, tw1 = template_1.shape[:2]

        # Create the ROI by extracting the region from the input image
        print (roiY1_L,roiY2_L, roiX1_L,roiX2_L)
        roi_L = img[roiY1_L:roiY2_L, roiX1_L:roiX2_L]
        
        # Perform template matching for the first template within the ROI
        res1 = cv2.matchTemplate(roi_L, template_1, cv2.TM_CCORR_NORMED)
        min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(res1)

        # Set a threshold for the first match
        threshold = 0.1

        if max_val1 > threshold:
            # Calculate the top-left and bottom-right coordinates of the first matched region within the ROI
            top_left1 = (max_loc1[0] + roiX1_L, max_loc1[1] + roiY1_L)
            bottom_right1 = (top_left1[0] + tw1, top_left1[1] + th1)

            # Display the match value for the first match
            match_val1 = f"{round(max_val1 * 100, 2)}%"
            if max_val1 > valROI_L:
                print("Matching 1 : ", max_val1)
                # Draw a rectangle around the first matched region
                cv2.rectangle(img, top_left1, bottom_right1, (0, 255, 0), 2)
                print(top_left1, bottom_right1)
                cv2.putText(img, 'O K : '+ match_val1, top_left1, cv2.FONT_HERSHEY_PLAIN, 2, (233,233,0), 1, cv2.LINE_AA)

                self.lblResult_L.setStyleSheet("background-color: rgb(93, 93, 93); font: 75 81pt 'Georgia'; color: rgb(0, 255, 0);")
                self.lblResult_L.setText("OK")
                self.lblVar_L.setStyleSheet("background-color: rgb(93, 93, 93); font: 75 33pt 'Georgia'; color: rgb(0, 255, 0);")
                self.lblVar_L.setText(str(match_val1))

                #pymc3e.batchwrite_wordunits(headdevice="D1000", values=[1])
            else:
                print("Mismatching 1 : ", max_val1)
                cv2.rectangle(img, top_left1, bottom_right1, (0, 0, 255), 2)
                cv2.putText(img, 'N G : '+ match_val1, top_left1, cv2.FONT_HERSHEY_PLAIN, 2, (233,233,0), 1, cv2.LINE_AA)
                self.lblResult_L.setStyleSheet("background-color: rgb(93, 93, 93); font: 75 81pt 'Georgia'; color: rgb(255, 0, 0);")
                self.lblResult_L.setText("NG")
                self.lblVar_L.setStyleSheet("background-color: rgb(93, 93, 93); font: 75 33pt 'Georgia'; color: rgb(255, 0, 0);")
                self.lblVar_L.setText(str(match_val1))

                #pymc3e.batchwrite_wordunits(headdevice="D1000", values=[2])

            height, width, channel = img.shape
            bytes_per_line = channel * width
            q_image = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
            self.lblMainDisp.setPixmap(QPixmap.fromImage(q_image))

            # Perform template matching for the second template within the ROI
            template_2 = cv2.imread(file_Path + 'Template_2.JPG')  #, cv2.IMREAD_GRAYSCALE
            th2, tw2 = template_2.shape[:2]

            # Create the ROI by extracting the region from the input image
            roi_R = img[roiY1_R:roiY2_R, roiX1_R:roiX2_R]

            res2 = cv2.matchTemplate(roi_R, template_2, cv2.TM_CCORR_NORMED)
            min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(res2)

            if max_val2 > threshold:
                # Calculate the top-left and bottom-right coordinates of the second matched region within the ROI
                top_left2 = (max_loc2[0] + roiX1_R, max_loc2[1] + roiY1_R)
                bottom_right2 = (top_left2[0] + tw2, top_left2[1] + th2)

                # Display the match value for the second match
                match_val2 = f"{round(max_val2 * 100, 2)}%"

                # Draw a rectangle around the second matched region
                if max_val2 > valROI_R:
                    print("Matching 1 : ", max_val2)
                    cv2.rectangle(img, top_left2, bottom_right2, (0, 255, 0), 2)
                    cv2.putText(img, 'O K' + match_val2, top_left2, cv2.FONT_HERSHEY_PLAIN, 2, (233,233,0), 1, cv2.LINE_AA)
                    self.lblResult_R.setStyleSheet("background-color: rgb(93, 93, 93); font: 75 81pt 'Georgia'; color: rgb(0, 255, 0);")
                    self.lblResult_R.setText("OK")
                    self.lblVar_R.setStyleSheet("background-color: rgb(93, 93, 93); font: 75 33pt 'Georgia'; color: rgb(0, 255, 0);")
                    self.lblVar_R.setText(str(match_val2))

                    #pymc3e.batchwrite_wordunits(headdevice="D1000", values=[1])
                else:
                    print("Mismatching 2 : ", max_val2)
                    cv2.rectangle(img, top_left2, bottom_right2, (0, 0, 255), 2)
                    cv2.putText(img, 'N G' + match_val2, top_left2, cv2.FONT_HERSHEY_PLAIN, 2, (233,233,0), 1, cv2.LINE_AA)
                    self.lblResult_R.setStyleSheet("background-color: rgb(93, 93, 93); font: 75 81pt 'Georgia'; color: rgb(255, 0, 0);")
                    self.lblResult_R.setText("NG")
                    self.lblVar_R.setStyleSheet("background-color: rgb(93, 93, 93); font: 75 33pt 'Georgia'; color: rgb(255, 0, 0);")
                    self.lblVar_R.setText(str(match_val2))

                    #pymc3e.batchwrite_wordunits(headdevice="D1000", values=[2])

                height, width, channel = img.shape
                bytes_per_line = channel * width
                q_image = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                self.lblMainDisp.setPixmap(QPixmap.fromImage(q_image))

        # Display the image with matched regions
        #cv2.imshow('Lead Matching', img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()


# Main
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWin = Ui_MainWindow()

    mainWin.video_thread()
    mainWin.show()

    sys.exit(app.exec_())



   