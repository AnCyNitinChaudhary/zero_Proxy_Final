from cvzone.FaceDetectionModule import FaceDetector
from time import time
import cvzone
import cv2
classID=1 #0 is fake and 1 is real 
outputFolderPath = 'Dataset/DataCollect'
confidence=0.8
save=True
blurThreshold = 35
debug=False
offsetPercentageW = 10
offsetPercentageH = 20
camWidth, camHeight = 640, 480
floatingPoint=6
cap = cv2.VideoCapture(0)
cap.set(3,camWidth)
cap.set(4,camHeight)
detector= FaceDetector()
while True:
    success,img=cap.read()
    imgOut=img.copy()
    img,bboxs=detector.findFaces(img)
    listBlur = []  # True False values indicating if the faces are blur or not
    listInfo = []  # The normalized values and the class name for the label txt file

    if bboxs:
       # center=bboxs[0]["center"]
        #cv2.circle(img,center,5,(255,0,255),cv2.FILLED)
        for bbox in bboxs:
            x,y,w,h=bbox["bbox"]
            score=bbox["score"][0]
            print(x,y,w,h)
            if score>confidence:
                offsetW = (offsetPercentageW / 100) * w
                x = int(x - offsetW)
                w = int(w + offsetW * 2)
                offsetH = (offsetPercentageH / 100) * h
                y = int(y - offsetH * 3)
                h = int(h + offsetH * 3.5)
                # ------  To avoid values below 0 --------
                if x < 0: x = 0
                if y < 0: y = 0
                if w < 0: w = 0
                if h < 0: h = 0
                #cv2.rectangle(img,(x,y,w,h),(255,0,0),3)
                
                #finding blurness
                imgFace = img[y:y + h, x:x + w]
                cv2.imshow("Face", imgFace)
                blurValue = int(cv2.Laplacian(imgFace, cv2.CV_64F).var())
                if blurValue > blurThreshold:
                    listBlur.append(True)
                else:
                    listBlur.append(False)



                #Normalize
                ih, iw, _ = img.shape
                xc, yc = x + w / 2, y + h / 2
                
                print(xc,yc)
                xcn, ycn = round(xc / iw, floatingPoint), round(yc / ih, floatingPoint)
                wn, hn = round(w / iw, floatingPoint), round(h / ih, floatingPoint)
                if xcn > 1: xcn = 1
                if ycn > 1: ycn = 1
                if wn > 1: wn = 1
                if hn > 1: hn = 1
                listInfo.append(f"{classID} {xcn} {ycn} {wn} {hn}\n")
               
                #drawing
                cv2.rectangle(imgOut, (x, y, w, h), (255, 0, 0), 3)
                cvzone.putTextRect(imgOut, f'Score: {int(score * 100)}% Blur: {blurValue}', (x, y - 0),scale=2, thickness=3)
                if debug:
                    cv2.rectangle(img, (x, y, w, h), (255, 0, 0), 3)
                    cvzone.putTextRect(img, f'Score: {int(score * 100)}% Blur: {blurValue}', (x, y - 0),
                                       scale=2, thickness=3)
        if save:
             if all(listBlur) and listBlur != []:
                # ------  Save Image  --------
                timeNow = time()
                timeNow = str(timeNow).split('.')
                timeNow = timeNow[0] + timeNow[1]
                cv2.imwrite(f"{outputFolderPath}/{timeNow}.jpg", img)
                #save level text file 
                for info in listInfo:
                    f = open(f"{outputFolderPath}/{timeNow}.txt", 'a')
                    f.write(info)
                    f.close()


                

    cv2.imshow("Image",imgOut)
    cv2.waitKey(1)