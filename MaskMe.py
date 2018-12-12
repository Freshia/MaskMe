import cv2
faceFinder=cv2.CascadeClassifier('F:/Python/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
cam=cv2.VideoCapture(0)
img3=cv2.imread('DarthVader.jpg',cv2.IMREAD_COLOR)

while(True):
    _,frame=cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = faceFinder.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=3)
    #draw rectangle over all faces
    for (x, y, w, h) in faces:
        if h > 0 and w > 0:

            #h, w = int(0.4 * h), int( * w)
            w = w+20
            h = h+20
            startx = x-15
            starty = y-30
#Darth starty = y-30,startx = x-15,w=w+20,y=y+20
            frame_roi = frame[starty:(starty + h),startx:startx + w]

            #face_mask_small = imutils.resize(img3, width=w, height=h)
            face_mask_small = cv2.resize(img3, (w, h), interpolation=cv2.INTER_AREA)


            #face_mask_small = cv2.resize(img3, (w, h),interpolation=cv2.INTER_AREA)
            cv2.imshow('DarthVader', face_mask_small)
            #cv2.imshow('ROI', frame_roi)
            img3gray = cv2.cvtColor(face_mask_small, cv2.COLOR_RGB2GRAY)
            ret, mask = cv2.threshold(img3gray, 220, 255, cv2.THRESH_BINARY_INV)

            mask_inv = cv2.bitwise_not(mask)

            # Use the mask to extract the face mask region of interest
            masked_face = cv2.bitwise_and(face_mask_small, face_mask_small,mask=mask)
            # Use the inverse mask to get the remaining part of the image
            rows,cols = frame_roi.shape[:2]
            rows1, cols1 = mask_inv.shape[:2]
            if(rows==rows1 & cols==cols1):
                masked_frame = cv2.bitwise_and(frame_roi, frame_roi,mask=mask_inv)

                # add the two images to get the final output
                frame[starty:starty+ h, startx:startx+ w] = cv2.add(masked_face, masked_frame)
            else:
                print("Adjust face")

    cv2.imshow('EvilMe!!', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()

