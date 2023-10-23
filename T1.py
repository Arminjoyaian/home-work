import cv2
first_frame = None

our_video = cv2.VideoCapture(0)

while True:
    check, frame = our_video.read()
    our_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    our_frame_gray = cv2.GaussianBlur(our_frame_gray,(21,21),0)
    if first_frame is None:
        first_frame=our_frame_gray
        continue
    delta_frame = cv2.absdiff(first_frame, our_frame_gray)
    threshold_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    threshold_frame = cv2.dilate(threshold_frame, None, iterations=2)

    cnts, hierarchy = cv2.findContours(threshold_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    for contour in cnts:
        if cv2.contourArea(contour) < 500:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
    cv2.imshow("First Frame", first_frame)
    cv2.imshow("Capturing first ", our_frame_gray)
    cv2.imshow("Delta Frame", delta_frame)
    cv2.imshow("Threshold Frame", threshold_frame)
    cv2.imshow("Detecting objects", frame)

    key = cv2.waitKey(1)
    if(key == ord('q')):
        break

our_video.release()
cv2.destroyAllWindows