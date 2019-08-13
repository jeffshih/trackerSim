import numpy as np 
import cv2 
import math

def rectOverLap(A,B):
    
    if(A[0] > (B[0]+B[2]) or B[0] > (A[0]+A[2])):
        return False

    if(A[1] > (B[1]+B[3]) or B[1] > (A[1]+A[3])):
        return False

    return True

def centerOfRect(B):
    
    return (B[0]+B[2]/2., B[1]+B[3]/2.)

def centerOfBottom(B):
    
    return (B[0]+B[2]/2., B[1]+B[3])

def centerOfTop(B):
    
    return (B[0]+B[2]/2., B[1])

def distBtwPoint(p1, p2):

    return math.sqrt(pow(p1[0]-p2[0], 2) + pow(p1[1]-p2[1],2))

def getAngle(p1, p2):
    r = (p2[0]-p1[0])
    i = (p2[1]-p1[1])
    val = complex(r, i)
    angle = np.angle(val)
    return angle








    
