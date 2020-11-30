import cv2 
import numpy as np
  

cap = cv2.VideoCapture('C:/Users/user/Desktop/detection/test.mp4')
  

car_cascade = cv2.CascadeClassifier('C:/Users/user/Desktop/detection/haarcascade_car.xml')
  

while True: 
     
    ret, color_img = cap.read() 
    if ret == False:
        break  
    
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY) 
      
  
    
    cars = car_cascade.detectMultiScale(gray, 1.1, 4) 
      
    
    for (x,y,w,h) in cars: 
        cv2.rectangle(color_img,(x,y),(x+w,y+h),(0,255,0),2) 
  
   
    cv2.imshow('img',color_img)

    # Wait for Esc key to stop 
    if cv2.waitKey(33) == 27: 
        break
  
# De-allocate any associated memory usage 
cv2.destroyAllWindows() 