import cv2

classifier1_file="classifiers\car_classifier.xml"
classifier2_file="classifiers\human_classifier.xml"
classifier3_file="classifiers\Face_classifier.xml"
classifier4_file="classifiers\smile_classifier.xml"




print("Welcome To Ai_Detection")
print("Enter Cam To Use Camera(or)Enter Vid Name To Use a Video")
print("Red=Car;Yellow=Human;Green=Face;Blue=Smile")
print("To Exit press e")

x=input("")

if x=="Cam":
    print("Thanks For Using Camera")
    print("enter 0 if you want to use webcam")
    print("enter 1 if you want to use droidcam")
    cam=int(input(""))
    video=cv2.VideoCapture(cam)
else:
    if x=="Vid":
        print("Thanks For Using a Video")
        print("Video Name")
        Vid=str(input(""))
        video=cv2.VideoCapture(Vid)



car_tracker=cv2.CascadeClassifier(classifier1_file)
human_tracker=cv2.CascadeClassifier(classifier2_file)
face_tracker=cv2.CascadeClassifier(classifier3_file)
smile_tracker=cv2.CascadeClassifier(classifier4_file)



while True:
    (read_sucessfull,frame)=video.read()
    if read_sucessfull:
        grayscaled_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    else:
        break

    
    cars=car_tracker.detectMultiScale(grayscaled_frame)
    humans=human_tracker.detectMultiScale(grayscaled_frame)
    faces=face_tracker.detectMultiScale(grayscaled_frame)
    smiles=smile_tracker.detectMultiScale(grayscaled_frame)

    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,225),2)
    for(x,y,w,h) in humans:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,225,225),2)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,128,0),2)
    for(x,y,w,h) in smiles:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,165,0),2)

    cv2.imshow("Ai Detection",frame) 
    key=cv2.waitKey(1)

    if key==101:
        break
video.release()   
            

