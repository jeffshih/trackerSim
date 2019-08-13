import cv2
import math
from util import *
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
from filterpy.kalman import KalmanFilter
import logging


class TYPE:
    UNKNOWN = 0
    PREDICT = 1
    HIT = 2


class TrackHistory():

    def __init__(self):
        self.box = [0, 0, 0, 0] 
        self.type = TYPE.UNKNOWN


class STATE(): 
    NONE = 0
    INITED = 1
    CANDIDATE = 2 
    COUNTED = 3
    JUMPED = 4
    ENDED = 5



class Tracker(object):

    MAXWIDTH = 700
    MAXHEIGHT = 1000
    
   
    def checkBBoxVar(self):
        p = self.kf.x 
        flag = False
        S = p[2]/90
        #if(p[6]>self.Vsv):
        if(p[6] > S):
            self.abnormal1 +=1
            #self.VsFlag=True
            flag = True

        #if(p[4] > self.Vxyv or p[5] > self.Vxyv):
        if p[2] > 0:
            lf = np.sqrt(p[2])/3 
        else:
            lf = 30
        if(p[4]>lf or p[5]>lf):
            self.abnormal2 +=1 
            #self.VxyFlag =True
            flag = True
        return flag



    def __init__(self, bbox, ids, logger, cfg):
       
        self.abnormal1 = 0
        self.abnormal2 = 0
        
        self.Vsv = cfg["Vs"]
        self.Vxyv = cfg["Vxy"]

        self.logger = logger
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])
        
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0]])
        
        self.kf.R[2:, 2:] *= 10
        self.kf.Q[-1, -1] *= 0.01 
        self.kf.Q = np.array([[0.01, 0, 0, 0, 0.01, 0, 0],
                              [0, 0.01, 0, 0, 0, 0.01, 0],
                              [0, 0, 0.01, 0, 0, 0, 0.01],
                              [0, 0, 0, 0.01, 0, 0, 0],
                              [0.01, 0, 0, 0, 0.1, 0, 0],
                              [0, 0.01, 0, 0, 0, 0.1, 0],
                              [0, 0, 0.01, 0, 0, 0, 0.1]])
        #self.kf.Q[6, 6] = 1
        #self.kf.P *= 10
        
        self.kf.x[:4] = Tracker.get_xysr_rect(bbox) 
        self.ids = ids 
        self.state = STATE.INITED

        self.trackHistory = []

        self.timeSinceUpdate = 0
        self.hits = 1
        self.age = 0
        self.countRect = np.array([0, 0, 0, 0])
        self.VsFlag = False
        self.VxyFlag = False

    def get_rect_xysr(self,p):
      
        cx, cy, s, r = p[0], p[1], p[2], p[3]
        w = math.sqrt(abs(s*r))
        h = abs(float(s/w))
        x = float(cx-w/2.)
        y = float(cy-h/2.)

        if (x < 0 and cx > 0):
            x = 0
        
        if (y < 0 and cy > 0):
            y = 0

        return np.array([x, y, w, h])

    def predict(self):
        
        self.kf.predict()
        self.logger.info("current trk id is {} and predicted state is :".format(self.ids)) 
        self.logger.info(self.kf.x)
        abnormal = self.checkBBoxVar()
        self.age += 1
        self.timeSinceUpdate += 1 
        bInvalid = False

        if (self.kf.x[2,0]>0):
            predictBox = self.getState()
        else:
            self.logger.info("=================== current tracker {} invalid ===============".format(self.ids))
            bInvalid = True
            predictBox = np.array([0, 100, 100, 100])
 

        th = TrackHistory()
        th.box = predictBox
        th.type = TYPE.PREDICT
        self.trackHistory.append(th)

        if(len(self.trackHistory) > 30):
            self.trackHistory.pop(0)

        nextEstimatedRect = predictBox
        
        if(bInvalid):
            self.logger.info("=================== current tracker {} invalid ===============".format(self.ids))
            self.state = STATE.ENDED
        elif(nextEstimatedRect[2] > Tracker.MAXWIDTH or nextEstimatedRect[3] > Tracker.MAXHEIGHT):
            self.state = STATE.ENDED
            self.logger.info("######################## bbox too big {}#####################".format(self.ids))
        elif(self.timeSinceUpdate > 25):
            self.state = STATE.ENDED
            self.logger.info("************************ too long un hit {}*******************".format(self.ids))
        elif(self.age > 15 and self.timeSinceUpdate > self.hits * 3):
            self.state = STATE.ENDED
            self.logger.info("------------------------- bad update quality{} ----------------".format(self.ids))
        #elif(self.age > 15 and float(self.timeSinceUpdate)/self.age > 0.8):
        #    self.state = STATE.ENDED

        return nextEstimatedRect 


    def get_xysr_rect(bbox):
        cx = bbox[0]+bbox[2]/2. 
        cy = bbox[1]+bbox[3]/2. 
        s = bbox[2]*bbox[3]
        r = float(bbox[2])/bbox[3]
        
        return np.array([cx, cy, s, r])

    def update(self, bbox):

        self.timeSinceUpdate = 0
        self.hits += 1
        frameNow = len(self.trackHistory)

        if frameNow <=1:
            speed = 0 
        else:
            checkSpeedNum = 5
            pathLen = 0

            if (frameNow<5):
                checkSpeedNum = frameNow-1

            for i in range(1,checkSpeedNum):
                b1 = self.trackHistory[frameNow-i].box 
                b2 = self.trackHistory[frameNow-i-1].box 
                pntN = centerOfTop(b1)
                pntP = centerOfTop(b2)
                pathLen += distBtwPoint(pntN,pntP)
                
            
            speed = pathLen/checkSpeedNum

        def sigmoid(x):
            s = 1/(1+np.exp(-x))
            return s
        
        S = self.kf.x[2]/90
        if self.VsFlag:
            self.kf.x[6] = S*sigmoid((self.kf.x[6]/S)-1)
            self.VsFlag = False 
        
        lf = np.sqrt(abs(self.kf.x[2]))/3

        if self.VxyFlag:
            if self.kf.x[5] > self.kf.x[4]:
                self.kf.x[5] = lf*sigmoid((self.kf.x[5]/lf)-1)
            else:
                self.kf.x[4] = lf*sigmoid((self.kf.x[4]/lf)-1)
            self.VxyFlag = False
        
        measurement = Tracker.get_xysr_rect(bbox)
        self.kf.update(measurement)
        predictBox = self.getState()
        th = TrackHistory()
        th.box = predictBox
        th.type = TYPE.HIT
        self.trackHistory.pop()
        self.trackHistory.append(th)

        self.lastUpdateRect = bbox

        '''Logging'''

        self.logger.info("=========== current tracker is =========")
        self.logger.info(self.ids)
        self.logger.info("=========== current residual is ========")
        self.logger.info(self.kf.y)
        self.logger.info('==========current updated status is =======')
        self.logger.info(self.kf.x)
        self.logger.info('==========current kalman gain is ==========')
        self.logger.info(self.kf.K)
        
    def getState(self):
        return self.get_rect_xysr(self.kf.x)


if __name__ == '__main__':

    bbox = np.array([[666], [162], [83], [301]])

    xysr = Tracker.get_xysr_rect(bbox)
    rect = Tracker.get_rect_xysr(xysr)

    self.logger.info(bbox.shape)
    trk = Tracker(bbox, 0)
    self.logger.info("xysr: ", xysr)
    self.logger.info("rect: ", rect)

    measure = np.array([[666],[161],[82],[303]])

    nextRect = trk.predict()
    trk.update(measure)
    
    for _ in range(20):
        nextRect = trk.predict()
        self.logger.info(trk.state) 

    self.logger.info("trk history: ", trk.trackHistory)
    self.logger.info("after update:", nextRect)

