import numpy as np
import cv2
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
template = cv2.imread('template.jpg')
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame,1)   #Gambar di flip 1:horiz, 0:vert
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Konversi jadi citra grays

        ordo = template_gray.shape
        w = ordo[0]
        h = ordo[1]

        # metode yg bisa dipakai
        metode = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

        #proses matching
        res = cv2.matchTemplate(gray,template_gray,metode[0])
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        #bounding box
        cv2.rectangle(frame,max_loc,(max_loc[0]+w, max_loc[1]+h),(0,0,255), 2)
                             
        if cv2.waitKey(1) & 0xFF == ord('c'):
            cv2.rectangle(frame,(0,0),(200,200),(0,255,0), 2)
            template = gray[:200,:200]
            cv2.imwrite('template.jpg', template)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.imshow('frame',frame)    #menampilkan di layar
        
    else:
        break
# Release semua objek, baik kamera maupun file
cap.release()
cv2.destroyAllWindows()
