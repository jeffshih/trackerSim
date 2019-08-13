import json 
import os 
import numpy as np 
import cv2 


class detector(object):
    
    def loadLocalData(self):
        for root, subdir, files in os.walk(self.pathToDetection):
            subdir.sort()
            files.sort()
            for fileName in files:
                filePath = os.path.join(root, fileName)
                self.jobList.append(filePath)

    def __init__(self,pathToDetection=""):
        self.currentFrame = 0
        self.detections = [] 
        self.jobList = []
        self.clipPath = ""
        self.pathToDetection = pathToDetection
        if (os.path.isdir(self.pathToDetection)):
            self.loadWhole()
            self.frameLen = len(self.jobList)
            self.clipCount = int(self.frameLen/1800)+1
        else:
            self.frameLen = 1800
            self.clipCount = 1

    def getCurrentHoldDets(self):
        self.curDets = {}
        for idx in range(self.clipCount):
            if (idx != self.clipCount-1):
                self.curDets[idx+1]=self.jobList[idx*1800:(idx+1)*1800]
            else:
                self.curDets[idx+1]=self.jobList[idx*1800:]
        return self.curDets


    def loadClip(self,pathToClip):
        self.clipPath = pathToClip
        self.jobList = []
        with open(pathToClip) as f:
            self.jobList = f.readlines()
            f.close()

    def loadWhole(self):
        self.loadLocalData()
            
    def getClipDetections(self):
        self.detections = []
        for job in self.jobList:
            j = self.parseJobFile(job)
            self.detections.append(j)
        return self.detections  
    
    def getNextFrame(self):
        job = self.jobList[self.currentFrame]
        print(job)
        self.currentFrame += 1
        return self.parseJobFile(job)

    def parseJobFile(self, jobFile):
        jobFile = jobFile.strip()
        with open(jobFile) as f:
            reader = json.load(f)
            return reader
 

def createClip(path):
    mockDetector = detector(path)
    res = mockDetector.getCurrentHoldDets()
    
    clips = res.keys()
    print(clips)
    
    for idx in clips:
        data = res.get(idx)
        txtName = path.split('/')[2]+'_%d.txt'%(idx)
        txtPath = os.path.join(path, 'clip', txtName)
        print(txtPath)

        with open(txtPath, 'w+') as f:
            for line in data:
                f.write(line+'\n')
            f.close()

def splitVideo(path):
    md = detector(path)
    videoPath = './SourceVideo/{}.mp4'.format(path.split("/")[2])
    #videoPath = './SourceVideo/190224_063422-17m10s.mp4'
    res = md.getCurrentHoldDets()
    clips = res.keys()
    print(clips) 
    #for idx in clips:
    #print(videoPath1)
    #print(videoPath)
    cap = cv2.VideoCapture(videoPath)
    #fourcc = cv2.VideoWriter_fourcc(0x7634706d/'mp4v')
    fileNames = ['./SplitedVideo/{}_{}.mp4'.format(path.split("/")[2],i) for i in clips]
    writers = [cv2.VideoWriter(fileName,0x7634706d,30,(1920,1080)) for fileName in fileNames]
    frameCnt = 0
    writerIndex = 0
    while(cap.isOpened()):
        ret,frame = cap.read()
        if ret == True:
            frameCnt += 1
            if (frameCnt%1800==0):
                print("now release writer : ",writerIndex)
                writers[writerIndex].release()
                writerIndex+=1
                print("start write video using writer",writerIndex)
            writers[writerIndex].write(frame)
        else:
            break
    cap.release()

def splitVideoToImg(path):
    md = detector(path)
    imgSavePath = './VideoImage'
    videoPath = './SourceVideo/{}.mp4'.format(path.split("/")[2])
    res = md.getCurrentHoldDets()
    clips = res.keys()
    print(clips) 
    cap = cv2.VideoCapture(videoPath)
    frameCnt = 0
    clipIndex = 0
    

    while(cap.isOpened()):
        ret,frame = cap.read()
        if ret == True:
            if (frameCnt%1800==0):
                print("now release writer : ", clipIndex)
                clipIndex +=1
            frameCnt += 1
            imgSubPath = '{}/{}'.format(imgSavePath,clipIndex)
            if (os.path.isdir(imgSubPath)!= True):
                os.makedirs(imgSubPath)
            imgPostFix = '{}_{}.jpg'.format(path.split("/")[2],frameCnt)
            imgName = os.path.join(imgSubPath,imgPostFix)
            cv2.imwrite(imgName,frame)
        else:
            break
    cap.release()




if __name__ == '__main__':
  

    """
    Path setting and config 
    """

    clipLength = 1800

    videoList = ['../Result/190330_065958-17m29s']

    """ 
    Create video clip  
    """
    for p in videoList:
         createClip(p)
         #splitVideo(p)
