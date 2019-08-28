import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment 
#from scipy.optimize.linear_sum_assignment import linear_sum_assignment
#import tracker as tracker
import tTracker as tracker
import detector as Detector
import math
from util import *
import logging
import warnings
from numpy.linalg import norm

def warn(*args, **kwargs):
    pass

warnings.warn = warn


class MultiTracker(object):

    def __init__(self,trackerCfg,logger):
        self.activeTracker = []
        self.frameNum = 0
        self.trackerCount = 0
        self.trackerID = 0
        self.abnormal = [0,0,0,0,0]
        self.logger = logger
        self.abnormal = [0,0,0,0,0]
        self.abnormal3 = 0
        self.abnormal4 = 0
        self.abnormal5 = 0
        self.cfg = trackerCfg
        self.totalTracker = 0
        self.PRECISION = 100
        self.affinityThreshold = 0.3

    def initTracker(self,boxes, logger):
        bbox = boxes[:,np.newaxis]
        self.trackerID +=1
        trk = tracker.Tracker(bbox,self.trackerID,logger,self.cfg)
        self.activeTracker.append(trk)
        self.trackerCount = len(self.activeTracker)
        self.logger.info("init tracker number {} with {} holds".format(self.trackerID,self.trackerCount))


    def getTrackerCount(self):
        return self.trackerCount

    def getMatchingCost(self,detBoxes,predBoxes):
        detCnt = len(detBoxes)
        predCnt = len(predBoxes) 
        iouMatrix = np.zeros((detCnt,predCnt))
        costMatrix = np.zeros((detCnt,predCnt))
        for i, det in enumerate(detBoxes):
            for j,pred in enumerate(predBoxes):
                trkHist = self.activeTracker[j].trackHistory
                trackerInstance = self.activeTracker[j]
                iou = self.costIOU(det,pred)
                linPosArea = self.costLinPosArea(det,pred)
                expPosArea = self.costExpPosArea(det,pred)
                costM = self.costMaha(det,pred,trackerInstance)
                #speed = self.costSpeed(trkHist,det)
                cost = self.PRECISION*iou
                #cost = expPosArea 
                #cost *= self.PRECISION
                #if(iou<0.3 and costExpPosArea < 0.3):
                #    cost = 0.5*iou + 0.5*costExpPosArea 
                #elif(iou < 0.5 or costExpPosArea < 0.5):
                #    cost = 0.25*iou + 0.25*area + 0.5*speed
                #    intersect = self.getArea(self.intersection(det,pred))
                #    minArea = min(self.getArea(det),self.getArea(pred))
                #    if((intersect/minArea) > 0.75):
                #        cost = 0.5*iou + 0.5*area + speed 
                #    else:
                #        ndw = (det[2]-pred[2])/(det[2]+pred[2])
                #        ndh = (det[3]-pred[3])/(det[3]+pred[3])
                #        costB = math.exp(-0.5*(ndw*ndw+ndh*ndh))
                #        if (costB > 0.75):
                #            cost = 100
                #        else:
                #            cost = 0.5*iou + 0.5*area + speed*1.5
                #print("for matching cost speed is:", speed)
                costMatrix[i, j] = cost
                iouMatrix[i, j] = costM
        return costMatrix, iouMatrix

    def doTracking(self, res, img=""):
        #if(len(self.activeTracker)==0):
             
        predictBoxes = []
        

        self.logger.info("=============== now do tracking on frame {} ==============".format(self.frameNum))
        for i,trk in enumerate(self.activeTracker):
            if (trk.state == 5):
                self.abnormal[0] += self.activeTracker[i].abnormal1 
                self.abnormal[1] += self.activeTracker[i].abnormal2
                if(len(self.activeTracker[i].trackHistory)<10):
                    self.abnormal4 +=1
                    self.abnormal[3] +=1
                if(self.activeTracker[i].hits<3):
                    self.abnormal5 +=1
                    self.abnormal[4] +=1
                del(self.activeTracker[i])

                self.logger.info("aaaaaaaaaaaaaaaaaaaaa tracker{} were deleted aaaaaaaaaaaaaaaa".format(trk.ids))
            else:
                predictBoxes.append(trk.getState())
        
        self.trackerCount = len(self.activeTracker)
        detectBoxes = []
 
        if(len(res['Detections'])>0):
            for det in res['Detections']:
                x = det['x']
                y = det['y']
                w = det['w']
                h = det['h']
                bbox = np.array([x, y, w, h])
                detectBoxes.append(bbox)
        
        costMatrix, iouMatrix = self.getMatchingCost(detectBoxes,predictBoxes)              

        unmatchedDetections = []
        unmatchedTrackers = []
        match_indice = linear_assignment(costMatrix)


        for d, dets in enumerate(detectBoxes):
            if (d not in match_indice[:,0]):
                unmatchedDetections.append(d) 
        for t, trks in enumerate(predictBoxes):
            if (t not in match_indice[:,1]):
                unmatchedTrackers.append(t)

        #filter out low IOU 
        matches = []
        for m in match_indice:
            if(iouMatrix[m[0],m[1]]  > 9.4877): #self.affinityThreshold*self.PRECISION):
            #if(iouMatrix[m[0],m[1]] > self.affinityThreshold):
                unmatchedDetections.append(m[0])
                unmatchedTrackers.append(m[1])
            else:
                matches.append(m.reshape(1,2))
        if (len(matches)==0):
            matches = np.empty((0,2),dtype=int)
        else:
            matches = np.concatenate(matches, axis = 0)
        #matches = match_indice

        """
        Logging
        """
        if(self.trackerCount>0 or len(detectBoxes)>0):
            self.logger.info("doTracking detectBoxes : \n{}".format(detectBoxes))
            self.logger.info("doTracking predictBox : \n{}".format(predictBoxes))
            self.logger.info("current costMatrix is :\n".format(costMatrix))
            self.logger.info("with det count : %3d and predCount : %3d"%(len(detectBoxes),self.trackerCount))
            self.logger.info("matched index is : \n{}".format(matches))
            self.logger.info("unmatchedDetections :\n{}".format(unmatchedDetections))
            self.logger.info("unmatchedTrackers :\n{}".format(unmatchedTrackers))
 
        for pair in matches:
            t = pair[1]
            d = pair[0]
            trk = self.activeTracker[t]
            measurement = detectBoxes[d]
            trk.update(measurement)

        for idx in unmatchedDetections:
            self.initTracker(detectBoxes[idx], self.logger)
            self.totalTracker +=1 
        if(self.trackerCount > 2.5*len(detectBoxes)):
            self.logger.info('abnormal3, too many trackers')
            self.abnormal3 +=1
            self.abnormal[2] +=1

        self.frameNum += 1        

    def predict(self):
        res = {}
        if(self.trackerCount==0):
            return res
        for trk in self.activeTracker:
            if (trk.state != 5):
                tracklet = trk.predict()
                res[trk.ids] = tracklet
            else:
                self.logger.info("======================= trk {} is ended=============".format(trk.ids))

        self.logger.info("prection result with {} trackers".format(len(res)))
        return res 

    def getArea(self,bbox):
    #    self.logger.info("getArea : ",bbox)
        return bbox[2]*bbox[3]

    def intersection(self, a, b):
    #    self.logger.info("intersection a :", a)
    #    self.logger.info("intersection b :", b)
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[0]+a[2], b[0]+b[2])
        y2 = min(a[1]+a[3], b[1]+b[3])

    #    self.logger.info("intesect : %3d, %3d, %3d, %3d"%(x1, x2, y1, y2))
        if x1<x2 and y1<y2:
            return [x1, y1, x2-x1, y2-y1]
        else:
            return [0,0,0,0]

    def costIOU(self, det,pred):
        
        intersect = self.getArea(self.intersection(det,pred))
        union = self.getArea(det)+self.getArea(pred)-intersect 
        IOU = intersect/union 
        cost = 1.0-IOU
        return cost 

    def costExpPosArea(sef, det,pred):
        dcx = det[0] + det[2]/2
        dcy = det[1] + det[3]/2
        pcx = pred[0] + pred[2]/2
        pcy = pred[1] + pred[3]/2

        ndx = (dcx-pcx)/det[2]
        ndy = (dcy-pcy)/det[3]
        costA = math.exp(-0.5*(ndx**2+ndy**2))
        ndw = abs(det[2]-pred[2])/(det[2]+pred[2])
        ndh = abs(det[3]-pred[3])/(det[3]+pred[3])
        costB = math.exp(-0.5*(ndw+ndh))

        cost = 1.0-(costA*costB)

        return cost

    def costMaha(self, det, pred, trackerInstance):
        
        deltaY = np.array([det[i]-pred[i] for i in range(4)]) 
        mahalanobis = math.sqrt(float(np.dot(np.dot(deltaY.T, trackerInstance.kf.SI),deltaY)))
        return mahalanobis        

    def costLinPosArea(self, det, pred):
        
        dcx = det[0] + det[2]/2
        dcy = det[1] + det[3]/2
        pcx = pred[0] + pred[2]/2
        pcy = pred[1] + pred[3]/2

        posCost = np.sqrt(pow(dcx-pcx,2) + pow(dcy-pcy,2))/1797
        shapeCost = np.sqrt(pow(det[2]-pred[2],2)+pow(det[3]-pred[3],2))/3228496

        return posCost*shapeCost

    def costSpeed(self, trkHist,det):
        frameNow = len(trkHist)
        if (frameNow <= 1):
            return 0
        lastHit = -1
        pathLen = 0
        cost = 100

        for i in range(len(trkHist)-1,0, -1):
            th = trkHist[i]
            if(th.type == 2):
                lastHit = i 
                break
        checkSpeedNum = 5
        if (frameNow<5):
            checkSpeedNum = frameNow-1

        for i in range(1,checkSpeedNum):
            pntN = centerOfTop(trkHist[frameNow-i].box)
            pntP = centerOfTop(trkHist[frameNow-i-1].box)
            pathLen += distBtwPoint(pntN,pntP)


        speed = pathLen/checkSpeedNum
  #      self.logger.info("calculat speed :" , speed, ) 
        if(lastHit>-1):
            cntOfTopDet = centerOfTop(det)
            cntOfTopLH = centerOfTop(trkHist[lastHit].box)
            deltaDist = distBtwPoint(cntOfTopDet,cntOfTopLH)
            angleBtwDet = getAngle(cntOfTopDet,cntOfTopLH)
            
            pntN = centerOfTop(trkHist[frameNow-1].box)
            vecDP = (pntN[0]-cntOfTopLH[0],pntN[1]-cntOfTopLH[1])
            vecDD = (cntOfTopDet[0]-cntOfTopLH[0],cntOfTopDet[1]-cntOfTopLH[1])
            predictAngle = getAngle(pntN,cntOfTopLH) 
            
            direction = (predictAngle-angleBtwDet)

            predDist = speed*(frameNow-lastHit-1)
            
            if((deltaDist < predDist) or (((deltaDist < 2*predDist) and speed < 30) or (deltaDist < 30 and speed < 10))):
                cost = 0
            else:
                Lef = math.pow(det[2]*det[3]*trkHist[lastHit].box[2]*trkHist[lastHit].box[3], 0.25)
                if(deltaDist > Lef*(frameNow-lastHit-1)):
                    return cost 
                else:
                    cost = np.dot(vecDP,vecDD)/(norm(vecDP)*norm(vecDD))
                    
                    #cost = 1.0-(0.5*(math.exp(deltaDist*-0.1)) - 0.5*((direction/2*np.pi)**2))
                    #print("cost speed is ", cost)
                    #print("detlaDist is ", deltaDist)
                    #print("Lef is", Lef*(frameNow-lastHit-1))
                    #print("direction is ", direction)
                    #cost = 1.0 - 0.5*(math.exp(deltaDist*-1/30)) - 0.5*((abs(direction)/2*np.pi))
                    #cost = 1.0-math.exp(deltaDist*-1/10)
        return cost

    def getHistory(self):
        ths = []
        for trk in self.activeTracker:
            ths.append(trk.trackHistory)
        return ths

    def matching(self):
        return 0
    
    def getState(self):
        preds = []
        for trk in self.activeTracker:
            preds.append(trk.getState())

        return preds

if __name__ == '__main__':
  

    """
    Path setting and config 
    """

    clipLength = 1800

    videoList = ['../Result/190224_063422-17m10s',
                 '../Result/190301_073457-19m38s',
                 '../Result/190330_073124-17m35s',
                 '../Result/190414_123559-17m47s']
    path = '../Result/190414_123559-17m47s'
    clipPath = '../Result/190224_063422-17m10s/clip/190224_063422-17m10s_0_.txt'
  

    """ 
    Create video clip  
    """
   # for p in videoList:
   #      createClip(p)
  
    """
    use mock detector to get result
    """
    
    mockDetector = Detector.detector(path)
    mockDetector.loadClip(clipPath)
    clipDets = mockDetector.getClipDetections()
    #self.logger.info(clipDets[1])
    
    MamaTracker = MultiTracker()
    
#    for data in clipDets:
#        frameNum = data['FrameIndex']
#        dets = data['Detections']
#        self.logger.info(len(dets))
#        for det in dets:
#            x = det['x']
#            y = det['y']
#            w = det['w']
#            h = det['h']
#            bbox = np.array([[x], [y], [w], [h]])
            #MamaTracker.initTracker(bbox)
#        break
#    self.logger.info("After init mama tacker has %3d trackers"%(MamaTracker.getTrackerCount()))
    currentFrame = 0
    maxFrame = len(clipDets)
    while(True):
        if(currentFrame == maxFrame):
            break 
        clipDet = clipDets[currentFrame]
        MamaTracker.doTracking(clipDet)
        currentFrame+=1
    #iter1 = MamaTracker.predict()
#    ths = MamaTracker.getHistory()

#    for thList in ths:
#        for idx,th in enumerate(thList):
#            self.logger.info("track history at : %d :"%(idx), th.box)
    
#    MamaTracker.doTracking(clipDets[2])
 
#    self.logger.info(MamaTracker.getTrackerCount())
#    for _ in MamaTracker.activeTracker:
#        self.logger.info("Tracker current state :", _.getState())
