import numpy as np 
import detector 
import MultiTracker 
import cv2 
import os 
import sys 
import argparse 
import glob
import re
import logging
loggingLevel = logging.ERROR
logging.basicConfig(level = loggingLevel,filename='./stdoutLog')

class PeopleCounter(object):

    COLORTXT = (0,0,255)
    COLORDET = (0,0,0)
    COLORTRK = (0,255,255)

    def __init__(self,path = ".",tracker=None):
        self.detector = detector.detector()
        self.w = 1920
        self.h = 1080
        self.clipDets =[]
        if tracker is None:
            self.tracker = {"Vs": [2000],"Vxy":[100]}
        else:
            self.tracker = tracker
        self.abnormal = np.array([0,0,0,0,0])
        self.totalTracker = 0

    def renderBox(self,res,img,dets = False):
        if(dets == True):
            COLOR = self.COLORDET 
        else:
            COLOR = self.COLORTRK

        for key,det in res.items():
            idx = key 
            x,y,w,h = (int(i) for i in det) 
            cv2.rectangle(img, (x, y), (x+w, y+h), COLOR, 3)
            cv2.putText(img, str(idx), (x+w, y+h), cv2.FONT_HERSHEY_PLAIN,
                        1, self.COLORTXT, 2, cv2.LINE_AA)
        return img

    def count(self,path,mode="blank"):
        self.detector.loadClip(path)
        self.clipDets = self.detector.getClipDetections() 
        pattern = re.compile('[0-9]{6}_[0-9]{6}-\d+m\d+s_\d+')
        target = pattern.search(path).group()
        self.processVideoFileName = './SplitedVideo/{}.mp4'.format(target)
        if mode == "blank":
            self.genBlank()
        elif mode == "img":
            self.genImg()
        elif mode == "video":
            self.genVideo() 
        elif mode == "track":
            self.genTrack()


    def countAll(self, clips,testCnt):
        for key, val in clips.items():
            root = val+'/*.txt'
            target= glob.glob(root)
            target.sort()
            for idx,path in enumerate(target):
                index = int(re.search('s_\d+', path).group().split('_')[1]) 
                vidName = '{}-{}-res{}.mp4'.format(key,index,testCnt)
                self.vidpath = os.path.join('./result',key,vidName)
                logName = '{}-{}-res{}.log'.format(key,index,testCnt) 
                self.logpath = os.path.join('./result',key,logName)
                if os.path.isfile(self.logpath):
                    continue             
                logger = self.setLogger()
                self.MamaTracker = MultiTracker.MultiTracker(self.tracker,logger)
                #print("now doing tracking on {}".format(vidName))
                #default_stdout = sys.stdout
                #sys.stdout = open(self.logpath,'w')
                self.count(path,'track')
                logger.warning("abnormal count = \n {}".format(self.MamaTracker.abnormal))
                #sys.stdout = default_stdout
                self.abnormal += np.array(self.MamaTracker.abnormal)
                self.totalTracker += self.MamaTracker.totalTracker
                del self.MamaTracker
        print("abnormal count = {}".format(self.abnormal))
        print("total tracker cnt is {}".format(self.totalTracker))

    def countClip(self, clips, clipCnt , testCnt, mode='video'):
        key,val = clips.popitem() 
        detPath = val+"/"+val.split('/')[2]+"_"+str(clipCnt)+'.txt' 
        vidName = '{}-{}-res{}.mp4'.format(key, clipCnt, testCnt)
        self.vidpath = os.path.join('./result',key,vidName)
        logName = '{}-{}-res{}.log'.format(key, clipCnt, testCnt) 
        self.logpath = os.path.join('./result',key,logName)
        if os.path.isfile(self.logpath):
            return False            
        logger = self.setLogger()
        self.MamaTracker = MultiTracker.MultiTracker(self.tracker,logger)
#        default_stdout = sys.stdout
#        sys.stdout = open(self.logpath,'w')
        self.count(detPath,mode)
#        sys.stdout = default_stdout
        return True 

    def setLogger(self):
        logger = logging.getLogger(self.logpath)
        #logger.addHandler(logging.FileHandler(self.logpath))
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(loggingLevel) 
        #logger.propagate = False
        return logger


    def genTrack(self):
        maxFrame = len(self.clipDets)
        currentFrame = 0 
        while(currentFrame < maxFrame):
            clipDet = self.clipDets[currentFrame]
            self.MamaTracker.doTracking(clipDet)
            res = self.MamaTracker.predict() 
            currentFrame += 1


    def genBlank(self):
        maxFrame = len(self.clipDets) 
        currentFrame = 0 
        while(currentFrame < maxFrame):
            clipDet = self.clipDets[currentFrame]
            self.MamaTracker.doTracking(clipDet)
            res = self.MamaTracker.predict() 
            img = np.zeros((self.h,1436,3))
            img[:,:] = (255,255,255) 
            img = self.renderBox(res,img)
            imgPath = '{}_{}.jpg'.format(self.processVideoFileName.split('.')[0],self.MamaTracker.frameNum)
            cv2.imwrite(os.path.join('./output',imgPath), img)
            currentFrame += 1

    def genVideo(self):
        writers = cv2.VideoWriter(self.vidpath,0x7634706d,30,(1436,1080)) 
        cap = cv2.VideoCapture(self.processVideoFileName)
        currentFrame = 0
        while(cap.isOpened()):
            ret,frame = cap.read()
            if ret == True:
                detectBoxes = {}
                clipDet = self.clipDets[currentFrame]
                self.MamaTracker.doTracking(clipDet)
                res = self.MamaTracker.predict()
                currentFrame +=1
                frame = frame[0:1080,242:1678,:]
                cv2.putText(frame,str(currentFrame),(200,200),cv2.FONT_HERSHEY_SIMPLEX,3,self.COLORTXT,2)
                if(len(clipDet['Detections'])>0):
                    for idx,det in enumerate(clipDet['Detections']):
                        dx = det['x']
                        dy = det['y']
                        dw = det['w']
                        dh = det['h']
                        bbox = np.array([dx, dy, dw, dh])
                        detectBoxes[idx] = bbox
                    frame = self.renderBox(detectBoxes,frame,dets=True) 
                frame = self.renderBox(res,frame)
                writers.write(frame) 
            else:
                break
        writers.release()
        cap.release()
 

    def genImg(self):
        cap = cv2.VideoCapture(self.processVideoPath)
        currentFrame = 0
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                clipDet = self.clipDets[currentFrame]
                self.MamaTracker.doTracking(clipDet)
                res = self.MamaTracker.predict()
                currentFrame +=1
                frame = frame[0:1080,242:1678,:]
                img = self.renderBox(res,frame)
                imgPath = '{}_{}.jpg'.format(self.processVideoFileName.split('.')[0],self.MamaTracker.frameNum)
                cv2.imwrite(os.path.join('./output',imgPath), img)
            else:
                break 
        cap.release()



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--Video","-v",dest='video',type=str,help="video path")
    parser.add_argument("--Tracker","-t",dest='config',type=str,help="tracker config") 
    parser.add_argument("--Mode","-m",dest='mode',type=str,default='all',help='mode')
    parser.add_argument("--Clip",'-c',dest='clip',type=str,help="video clip")
    parser.add_argument("--Round",'-r',dest='cnt',type=str,help="test round",default='0')
    parser.add_argument("--level", "-l", dest='level',type=str,help="logging level",default="WARNING")
    args = parser.parse_args()

    videoList = {'Clip-C': '../Result/190224_063422-17m10s/clip',
                 'Clip-D': '../Result/190301_073457-19m38s/clip',
                 'Clip-A': '../Result/190330_073124-17m35s/clip',
                 'Clip-B': '../Result/190414_123559-17m47s/clip',
                 'Clip-E': '../Result/190330_065958-17m29s/clip'}

    p = re.compile('\d{6}_\d{6}-\d+m\d+s')
    target = dict()
    if (args.video==None):
        target = videoList 
    else:
        target[args.video] = videoList.get(args.video) 

    videoFlag = False
    imgFlag = False 
#    Vs = [i for i in range(500,4000,100)]
#    Vxy = [i for i in range(30,380,10)]
    
#    tracker = {}
#    for i in range(len(Vs)):
#        tracker["Vs"] = Vs[i]
#        tracker["Vxy"] = Vxy[i]
#        pc = PeopleCounter(tracker=tracker) 
#        pc.countAll(target,0)
    pc = PeopleCounter()
    mode = args.mode 
    if mode == 'all':
        pc.countAll(target,args.cnt)
    else:
        pc.countClip(target, args.clip, args.cnt, mode)
