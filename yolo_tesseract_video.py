import cv2
import numpy as np
from PIL import Image
import pytesseract
import re
import math

cap = cv2.VideoCapture('test.mp4')
if cap.isOpened():
    w = int(cap.get(3))
    h = int(cap.get(4))
    f = int(cap.get(5))
    
    print('width: {}, height : {}, frame : {}'.format(w,h,f))
    
    
    # ratio=416/w
    # w = 416
    # h = int(h*ratio)
    # print(h)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    outVideo = cv2.VideoWriter('output.avi', fourcc, f, (w,h))

    net = cv2.dnn.readNet("obj_60000.weights", "obj.cfg")
    classes = []
    with open("obj.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

   
    while True:
        ret, img = cap.read()
        if ret:
            
            #img = cv2.resize(img, None, fx=ratio, fy=ratio)
            height, width, channels = img.shape
            # print(img.shape)

            # Detecting objects
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            # 정보를 화면에 표시
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    if class_id != 7: ## only bus number 
                        continue

                    confidence = scores[class_id]
                    if confidence > 0.3:
                        
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        # 좌표
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            font = cv2.FONT_HERSHEY_PLAIN
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]

                    label = str(classes[class_ids[i]])

                    
                    paddingx = int(w * 0.3)
                    paddingy = int(h * 0.05)
                    px1 = x - paddingx if x-paddingx >= 0 else 0
                    px2 = x + w + paddingx if x + w + paddingx <= width else width
                    py1 = y - paddingy if y-paddingy >= 0 else 0
                    py2 = y + h + paddingy if y+h+paddingy <= height else height
                    

                    roi = img[py1:py2, px1:px2]

                    src = roi
                    dst = roi.copy()
                    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
                    canny = cv2.Canny(gray, 5000, 1500, apertureSize = 5, L2gradient = True)
                    cv2.imshow("canny",canny)
                    lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 50, minLineLength = 5, maxLineGap = 15)
                    angle = 0
                    maxdim = 0
                    if not (lines is None):
                        for i in lines:
                            xdim = i[0][2] - i[0][0]
                            ydim = i[0][3] - i[0][1]
                            iangle = math.atan2(ydim, xdim)*180/np.pi
                            dim = math.sqrt((xdim * xdim) + (ydim * ydim))
                            if abs(angle) < 40 and maxdim < dim:
                                maxdim = dim
                                angle =iangle

                    cv2.imshow('dst',dst)
                    
                    
                    roih, roiw, roic = roi.shape
                    matrix = cv2.getRotationMatrix2D((roiw/2, roih/2), angle, 1)
                    roi = cv2.warpAffine(roi, matrix, (roiw, roih))
                    
                    # roi = cv2.GaussianBlur(roi,(5,5), 1)
                    # roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    # roi = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 2)

                    # kernel = np.ones((3, 3), np.uint8)
                    # roi = cv2.dilate(roi, kernel, iterations = 1)
                    # roi = cv2.erode(roi, kernel, iterations = 1)
                    # roi = cv2.erode(roi, kernel, iterations = 1)
                    # roi = cv2.dilate(roi, kernel, iterations = 1)
                    

                    

                    # roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
                    #roi = cv2.resize(roi, None, fx=1.5, fy=1.5)

                    r = pytesseract.image_to_string(roi, lang='Hangul')
                    rNumberlist= re.findall("\d+",r)
                    #print(rNumberlist)
                    carnum = ""
                    for num in rNumberlist:
                        if len(num) >= 4:
                            carnum=num[-4:]


                    cv2.imshow('roi',roi)
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 2)
                    cv2.putText(img, label+"  "+carnum, (x, y + 30), font, 3, (0,0,255), 3)
            cv2.imshow("Image", img)
            outVideo.write(img)
        else :
            print("end")
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
else:
    print('error')

#cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
