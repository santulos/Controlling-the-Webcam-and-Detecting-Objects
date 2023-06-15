import cv2, time
#este codigo captura la primera parte de los 10 seg de camara, si queres capturar un video necesitas un loop
from datetime import datetime

first_frame =None
video = cv2.VideoCapture(0)

status_list= [None,None]
times= []




while True:
	
        check, frame = video.read()

        status=0
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray,(21,21),0)    #difumina la imagen, la hace borrosa y permite mas precision en la escla de grises
        
        if first_frame is None:
            first_frame=gray
            continue # esto significa que continua al principio del loop

        
        
        delta_frame = cv2.absdiff(first_frame, gray) #genera otra imagen

        thresh_frame=cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1] # genera una tercera imagen en blancos y negros bien definidos a partir del balor 30

        thresh_frame=cv2.dilate(thresh_frame,None,iterations=2) # suaviza la imagen y elimina los aguros negros que se ven en la anterior, cuando aparece el objeto se estabiliza la imaen
        
        (cnts,_)=cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        for contour in cnts:
                if cv2.contourArea(contour) < 1000: #esto mide el area del contorno, si es mayor de 1000 pix 
                    continue                            # sigue al siguiente y si no sale del for
                status=1

                (x, y , w, h)= cv2.boundingRect(contour)
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
        
        status_list.append(status)
        if status_list[-1]==1 and status_list[-2]==0:
              times.append(datetime.now())               #esta parte del codio registra los cambios en 0 y 1, cambio de mov y registra el tiempo
        if status_list[-1]==0 and status_list[-2]==1:
              times.append(datetime.now())
              

        cv2.imshow("Capturing", gray)
        cv2.imshow("Delta", delta_frame)
        cv2.imshow("Thresh", thresh_frame)
        cv2.imshow("Color",frame)

        
        
        cv2.waitKey(1) 
        
        key = cv2.waitKey(1)
        if key == ord('q'):
                break
        
print(status_list)

video.release()
cv2.destroyAllWindows
