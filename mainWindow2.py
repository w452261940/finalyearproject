import time
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.font as tkFont

import cv2
import numpy
from PIL import Image, ImageTk, ImageDraw, ImageFont

import argparse
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank


class MainWindow:
    # window
    __mainWindow = tk.Tk()
    __windowWidth = 1280
    __windowHeight = 720 - 20
    __titleText = '标题'

    # camera
    cameraOpened = False
    cameraWidth = 640
    cameraHeight = 480

    # control
    clockLabel = None
    clockFormat = '%Y-%m-%d\n%H:%M:%S'
    toggleCameraBtn = None
    videoCanvas = None
    resultNameLabel = None
    resultTimeLabel = None

    # face
    mtcnn = None
    learner = None
    conf = None
    args = None
    targets = None
    names = None


    def __init__(self):
        self.initFaceVerify()
        self.createWindow()


    def initFaceVerify(self):
        parser = argparse.ArgumentParser(description='for face verification')
        parser.add_argument('-s', '--save', help='whether save', action='store_true')
        parser.add_argument('-th', '--threshold', help='threshold to decide identical faces', default=1.54, type=float)
        parser.add_argument('-u', '--update', help='whether perform update the facebank', action='store_false')
        parser.add_argument('-tta', '--tta', help='whether test time augmentation', action='store_true')
        parser.add_argument('-c', '--score', help='whether show the confidence score', action='store_true')
        self.args = parser.parse_args()

        self.conf = get_config(False)

        self.mtcnn = MTCNN()
        print('mtcnn loadad')

        self.learner = face_learner(self.conf, True)
        self.learner.threshold = self.args.threshold
        if self.conf.device.type == 'cpu':
            self.learner.load_state(self.conf, 'ir_se50.pth', True, True)
        else:
            self.learner.load_state(self.conf, 'ir_se50.pth', True, True)
        self.learner.model.eval()
        print('learner loaded')

        if self.args.update:
            self.targets, self.names = prepare_facebank(self.conf, self.learner.model, self.mtcnn, tta=self.args.tta)
        else:
            self.targets, self.names = load_facebank(self.conf)


    def createWindow(self):
        self.__mainWindow.title(self.__titleText)
        # self.__mainWindow.geometry('{}x{}'.format(self.__windowWidth, self.__windowHeight))
        self.__mainWindow.minsize(self.__windowWidth, self.__windowHeight)
        self.__mainWindow.rowconfigure(0, weight=1)
        self.__mainWindow.columnconfigure(0, weight=1)

        # 主要尺寸
        width7 = self.__windowWidth * 0.7
        width3 = self.__windowWidth * 0.3
        height7 = self.__windowHeight * 0.7
        height3 = self.__windowHeight * 0.3

        bottomFrame = tk.Frame(self.__mainWindow, height=height3)
        bottomFrame.pack_propagate(False)
        bottomFrame.pack(side=tk.BOTTOM, fill=tk.X)
        resultFrame = tk.LabelFrame(self.__mainWindow, width=width3, text='Result')
        resultFrame.pack_propagate(False)
        resultFrame.pack(side=tk.RIGHT, fill=tk.Y)
        cameraFrame = tk.LabelFrame(self.__mainWindow, width=width7, text='Camera')
        cameraFrame.pack_propagate(False)
        cameraFrame.pack(side=tk.LEFT, expand=tk.YES, fill=tk.BOTH)

        self.createVideoContent(cameraFrame)
        self.createResultContent(resultFrame)
        self.createTimeNewsContent(bottomFrame, width7)
        self.createOptContent(bottomFrame, width3)

        self.__mainWindow.mainloop()

    
    def createVideoContent(self, cameraFrame):
        self.videoCanvas = tk.Canvas(cameraFrame)
        self.videoCanvas.pack(expand=tk.YES, fill=tk.BOTH, padx=10, pady=10)


    def createResultContent(self, resultFrame):
        self.resultNameLabel = tk.Label(resultFrame, font='微软雅黑 20', width=15, height=2)
        self.resultNameLabel.pack(side=tk.BOTTOM)

        self.resultTimeLabel = tk.Label(resultFrame, font='微软雅黑 20', width=15, height=2)
        self.resultTimeLabel.pack(side=tk.BOTTOM)

    
    def createTimeNewsContent(self, bottomFrame, width):
        timeNewsFrame = tk.Frame(bottomFrame, width=width)
        timeNewsFrame.pack(side=tk.LEFT, expand=tk.YES, fill=tk.BOTH)

        timeFrame = tk.LabelFrame(timeNewsFrame, width=width/3, text='Time')
        timeFrame.pack_propagate(False)
        timeFrame.pack(side=tk.LEFT, expand=tk.YES, fill=tk.BOTH)
        newsFrame = tk.LabelFrame(timeNewsFrame, width=width - width/3, text='News')
        newsFrame.pack_propagate(False)
        newsFrame.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.BOTH)

        self.createTimeShow(timeFrame)
        self.createNewsShow(newsFrame)
    

    def createTimeShow(self, timeFrame):
        self.clockLabel = tk.Label(timeFrame, font='微软雅黑 20', text=time.strftime(self.clockFormat))
        self.clockLabel.pack(expand=tk.YES)
        self.clockLabel.after(1000, self.clockRefresh)


    def clockRefresh(self):
        self.clockLabel.config(text=time.strftime(self.clockFormat))
        self.__mainWindow.update()
        self.clockLabel.after(1000, self.clockRefresh)

    
    def createNewsShow(self, newsFrame):
        newsLabel = tk.Label(newsFrame, text='news here')
        newsLabel.pack(expand=tk.YES)


    def createOptContent(self, bottomFrame, width):
        optFrame = tk.LabelFrame(bottomFrame, width=width, text='Operation')
        optFrame.pack_propagate(False)
        optFrame.pack(side=tk.RIGHT, fill=tk.BOTH)

        self.toggleCameraBtn = tk.Button(optFrame, text='打开摄像头', font='微软雅黑 16', command=self.toggleCamera)
        self.toggleCameraBtn.pack(expand=tk.YES)


    def toggleCamera(self):
        self.cameraOpened = not self.cameraOpened
        self.toggleCameraBtn.config(text='打开摄像头' if not self.cameraOpened else '关闭摄像头')
        self.__mainWindow.update()
        if(self.cameraOpened):
            self.initCamera()
        else:
            self.videoCanvas.delete(tk.ALL)


    def initCamera(self):
        cap = cv2.VideoCapture(0)
        while self.cameraOpened and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                continue

            resultNames = []
            try:
                frame = cv2.flip(frame, 1)
                image2recog = Image.fromarray(frame)
                bboxes, faces = self.mtcnn.align_multi(image2recog, self.conf.face_limit, self.conf.min_face_size)
                bboxes = bboxes[:,:-1]
                bboxes = bboxes.astype(int)
                bboxes = bboxes + [-1,-1,1,1]                
                results, score = self.learner.infer(self.conf, faces, self.targets, self.args.tta)
               
                
                for idx, bbox in enumerate(bboxes):
                    if self.args.score:
                        name = self.names[results[idx] + 1] + '_{:.2f}'.format(score[idx])
                    else:
                        name = self.names[results[idx] + 1]

                    resultNames.append(name)
                    frame = self.drawResultBox(frame, bbox, name)
            except:
                resultNames = []
            
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.showResultText(resultNames)
            self.showResultImage(image)
                
        cap.release()


    def drawResultBox(self, frame, bbox, name):
        frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 6)

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        fontStyle = ImageFont.truetype("font/simsun.ttc", 20, encoding="utf-8")
        draw.text((bbox[0], bbox[1]), name, (0, 255, 0), font=fontStyle)

        frame = cv2.cvtColor(numpy.asarray(image), cv2.COLOR_RGB2BGR)
        return frame


    def showResultImage(self, image):
        canvasWidth = self.videoCanvas.winfo_width()
        canvasHeight = self.videoCanvas.winfo_height()
        imgWidth, imgHeight = self.getImgSize(canvasWidth, canvasHeight)

        image = image.resize((imgWidth, imgHeight), Image.ANTIALIAS)
        image2show = ImageTk.PhotoImage(image)

        self.videoCanvas.create_image(canvasWidth / 2, canvasHeight / 2, anchor=tk.CENTER, image=image2show)

        self.__mainWindow.update_idletasks()
        self.__mainWindow.update()


    # 使图片尺寸适应窗口大小
    def getImgSize(self, canvasWidth, canvasHeight):
        prop = self.cameraWidth / self.cameraHeight
        if (canvasWidth >= canvasHeight):
            imgHeight = canvasHeight - 10
            imgWidth = int(imgHeight * prop)
            return imgWidth, imgHeight
        else:
            imgWidth = canvasWidth - 10
            imgHeight = int(imgWidth / prop)
            return imgWidth, imgHeight

    
    def showResultText(self, names):
        if (len(names)):
            self.resultNameLabel.config(text=names)
            self.resultTimeLabel.config(text=time.strftime(self.clockFormat))
        else:
            self.resultNameLabel.config(text='')
            self.resultTimeLabel.config(text='')


mainWindow = MainWindow()

