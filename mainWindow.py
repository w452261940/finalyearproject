import time
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.font as tkFont
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont
from face_verify_init import initial as faceInitial


class MainWindiw:
    __mainWindow = tk.Tk()
    __windowWidth = 1280
    __windowHeight = 720 - 20
    __titleText = 'Face recognize for access control'

    cameraOpened = False
    cameraWidth = 640
    cameraHeight = 480

    videoCanvas = None
    logText = None

    mtcnn = None
    learner = None
    mtcnnConf = None
    targets = None
    names = None
    args = None


    def __init__(self):
        self.initFaceVerify()
        self.createWindow()


    def initFaceVerify(self):
        self.mtcnn, self.learner, self.mtcnnConf, self.args, self.targets, self.names = faceInitial()


    # 窗口
    def createWindow(self):
        self.__mainWindow.title(self.__titleText)
        self.__mainWindow.geometry('{}x{}'.format(self.__windowWidth, self.__windowHeight))
        self.__mainWindow.minsize(self.__windowWidth, self.__windowHeight)

        titleHeight = self.__windowHeight / 15
        self.createTitle(titleHeight)
        self.createMainContent(titleHeight)

        self.__mainWindow.mainloop()


    # 上方标题栏
    def createTitle(self, titleHeight):
        titleFrame = tk.Frame(self.__mainWindow, width=self.__windowWidth, height=titleHeight, bg='black')
        titleFrame.pack_propagate(False)
        titleFrame.pack(side=tk.TOP, expand=tk.NO, fill=tk.X)

        titleLabel = tk.Label(titleFrame, text=self.__titleText, bg='black', fg='white', font=('Consolas', '20', tkFont.BOLD))
        titleLabel.pack(side=tk.LEFT, expand=tk.NO, fill=tk.BOTH)


    # 下方内容容器
    def createMainContent(self, titleHeight):
        mainFrame = tk.Frame(self.__mainWindow, width=self.__windowWidth, height=self.__windowHeight - titleHeight, bg='#f5f5f5')
        mainFrame.pack_propagate(False)
        mainFrame.pack(side=tk.BOTTOM, expand=tk.YES, fill=tk.BOTH)

        self.createVideoContent(mainFrame, 0.75)
        self.createControlContent(mainFrame, 0.25)
    

    # 内容容器左侧视频显示区
    def createVideoContent(self, mainFrame, scale):
        self.videoCanvas = tk.Canvas(mainFrame, width=self.__windowWidth * scale, bg='pink')
        self.videoCanvas.pack_propagate(False)
        self.videoCanvas.pack(side=tk.LEFT, expand=tk.YES, fill=tk.BOTH)


    # 内容容器右侧操作和日志显示区
    def createControlContent(self, mainFrame, scale):
        controlFrame = tk.Frame(mainFrame, width=self.__windowWidth * scale)
        controlFrame.pack_propagate(False)
        controlFrame.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.BOTH)

        self.createRecognizeLogContent(controlFrame)
        self.createTopControlContent(controlFrame)


    # 右下角识别日志区，总高度1/3
    def createRecognizeLogContent(self, controlFrame):
        logFrame = tk.Frame(controlFrame, height=self.__windowHeight / 3)
        logFrame.pack_propagate(False)
        logFrame.pack(side=tk.BOTTOM, expand=tk.YES, fill=tk.BOTH)

        scrollBar = tk.Scrollbar(logFrame)
        self.logText = tk.Text(logFrame)

        scrollBar.config(command=self.logText.yview)
        self.logText.config(yscrollcommand=scrollBar.set)

        scrollBar.pack(side=tk.RIGHT, fill=tk.Y)
        self.logText.pack(side=tk.LEFT, expand=tk.YES, fill=tk.BOTH)


    # 右上角操作区，总高度2/3
    def createTopControlContent(self, controlFrame):
        operationFrame = tk.Frame(controlFrame, height=self.__windowHeight * 2 / 3)
        operationFrame.pack_propagate(False)
        operationFrame.pack(side=tk.TOP, expand=tk.YES, fill=tk.BOTH)

        self.createCameraControlButton(operationFrame)


    # 打开/关闭摄像头按钮
    def createCameraControlButton(self, operationFrame):
        cameraButton = tk.Button(operationFrame, text='toggle camera', command=self.toggltCamera)
        cameraButton.pack_propagate(False)
        cameraButton.pack(side=tk.TOP, expand=tk.YES, fill=tk.X)


    # 打开或关闭电脑摄像头
    def toggltCamera(self):
        self.cameraOpened = not self.cameraOpened
        if (self.cameraOpened):
            self.initCamera()
        else:
            self.videoCanvas.delete(tk.ALL)


    # Face Recognize
    def initCamera(self):
        cap = cv2.VideoCapture(0)
        cap.set(3, self.cameraWidth)
        cap.set(4, self.cameraHeight)
        # cap.set(5, 24) # 设置图像采集帧速
        while self.cameraOpened and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                try:
                    frame = cv2.flip(frame, 1)
                    image2recog = Image.fromarray(frame)
                    bboxes, faces = self.mtcnn.align_multi(image2recog, self.mtcnnConf.face_limit, self.mtcnnConf.min_face_size)
                    bboxes = bboxes[:,:-1]
                    bboxes = bboxes.astype(int)
                    bboxes = bboxes + [-1,-1,1,1]                
                    results, score = self.learner.infer(self.mtcnnConf, faces, self.targets, self.args.tta)
                    for idx, bbox in enumerate(bboxes):
                        if self.args.score:
                            image, name = self.drawResultBox(frame, bbox, self.names[results[idx] + 1] + '_{:.2f}'.format(score[idx]))
                        else:
                            image, name = self.drawResultBox(frame, bbox, self.names[results[idx] + 1])
                except:
                    name = None
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                self.showResultImg(image)

                if name != None:
                    self.showRsultText(name)

        cap.release()


    # 绘制识别结果，中文支持
    def drawResultBox(self, frame, bbox, name):
        frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 6)

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        fontStyle = ImageFont.truetype("font/simsun.ttc", 20, encoding="utf-8")
        draw.text((bbox[0], bbox[1]), name, (0, 255, 0), font=fontStyle)
        return image, name


    def showResultImg(self, image):
        canvasWidth = self.videoCanvas.winfo_width()
        canvasHeight = self.videoCanvas.winfo_height()
        imgWidth, imgHeight = self.getImgSize(canvasWidth, canvasHeight)

        image = image.resize((imgWidth, imgHeight), Image.ANTIALIAS)
        image2show = ImageTk.PhotoImage(image)

        self.videoCanvas.create_image(canvasWidth / 2, canvasHeight / 2, anchor=tk.CENTER, image=image2show)

        self.__mainWindow.update_idletasks()
        self.__mainWindow.update()


    def showRsultText(self, name):
        self.logText.insert('end', '{} {}'.format(time.strftime('%Y-%m-%d %H:%M:%S'), name))
        self.logText.insert(tk.INSERT, '\n')
        self.logText.see(tk.END)


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
  

mainWindiw = MainWindiw()
