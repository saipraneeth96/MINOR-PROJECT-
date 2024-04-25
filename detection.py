import numpy as np
import time
import cv2
import os
import numpy as np
from playsound import playsound
import threading
import smtplib
import imghdr
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from pydub import AudioSegment
from pydub.playback import play
from email.message import EmailMessage

def detect_animal(animal):
    # Function to detect if the animal is wild or not
    wild_animals = ["lion", "tiger", "bear"]  # Example list of wild animals

    if animal.lower() in wild_animals:
        return True  # Animal is wild
    else:
        return False  # Animal is not wild

def sound_alarm(animal_detected):
    if detect_animal(animal_detected):
        print("Wild animal detected! Sounding alarm...")
        # Code to produce alarm sound (implementation depends on your environment)
        # Example: You can use a library like playsound to play an alarm sound
        threading.Thread(target=playsound, args=(r'C:\Users\hemanth gadde\Downloads\Animal-Intrusion-Alarm-main (1)\Animal-Intrusion-Alarm-main\alarm.wav',), daemon=True).start()
        # Example: playsound('alarm_sound.mp3')
    else:
        print("No alarm needed. Animal is not wild.")

def send_email(label):
    Sender_Email = "vtu19044@veltech.edu.in"
    Reciever_Email = "hemanthgadde04@gmail.com"
    # Password = input('Enter your email account password: ')
    appPassword ="vwml ckjr mahq vzpc"   #ENTER GOOGLE APP PASSWORD HERE

    newMessage = EmailMessage()    #creating an object of EmailMessage class
    newMessage['Subject'] = "Animal Detected" #Defining email subject
    newMessage['From'] = Sender_Email  #Defining sender email
    newMessage['To'] = Reciever_Email  #Defining reciever email
    newMessage.set_content('An animal has been detected') #Defining email body

    with open(r'C:\Users\hemanth gadde\Downloads\Animal-Intrusion-Alarm-main (1)\Animal-Intrusion-Alarm-main\images\\' + label + '.png', 'rb') as f:
        image_data = f.read()
        image_type = imghdr.what(f.name)
        image_name = f.name[7:]

    newMessage.add_attachment(image_data, maintype='image', subtype=image_type, filename=image_name)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(Sender_Email, appPassword) #Login to SMTP server
        smtp.send_message(newMessage)      #Sending email using send_message method by passing EmailMessage object

def async_email(label):
    threading.Thread(target=send_email, args=(label,), daemon=True).start()

args = {"confidence":0.5, "threshold":0.3}
flag = False

labelsPath = r"C:\Users\hemanth gadde\Downloads\Animal-Intrusion-Alarm-main (1)\Animal-Intrusion-Alarm-main\yolo-coco\coco.names"
LABELS = open(labelsPath).read().strip().split("\n")
final_classes = ['bird', 'cat', 'dog', 'sheep', 'horse', 'cow', 'elephant', 'zebra', 'bear', 'giraffe']

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
    dtype="uint8")

weightsPath = os.path.abspath(r"C:\Users\hemanth gadde\Downloads\Animal-Intrusion-Alarm-main (1)\Animal-Intrusion-Alarm-main\yolo-coco\yolov3-tiny.weights")
configPath = os.path.abspath(r"C:\Users\hemanth gadde\Downloads\Animal-Intrusion-Alarm-main (1)\Animal-Intrusion-Alarm-main\yolo-coco\yolov3-tiny.cfg")

# print(configPath, "\n", weightsPath)
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

flag=True

# Start webcam capture
vs = cv2.VideoCapture(0)
(W, H) = (None, None)

while True:
    # read the next frame from the webcam
    (grabbed, frame) = vs.read()

    # if the frame was not grabbed, then we have reached the end
    # of the stream
    if not grabbed:
        break

    # if the frame dimensions are empty, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # initialize our lists of detected bounding boxes, confidences,
    # and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability)
            # of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > args["confidence"]:
                # scale the bounding box coordinates back relative to
                # the size of the image, keeping in mind that YOLO
                # actually returns the center (x, y)-coordinates of
                # the bounding box followed by the boxes' width and
                # height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top
                # and and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates,
                # confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping
    # bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
        args["threshold"])

    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            if(LABELS[classIDs[i]] in final_classes):
                # playsound('alarm.wav')
                if(flag):
                    sound_alarm(LABELS[classIDs[i]])
                    flag=False
                    async_email(LABELS[classIDs[i]])
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                    confidences[i])
                cv2.putText(frame, text, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    else:
        flag=True

    cv2.imshow("Output", frame)

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
vs.release()
cv2.destroyAllWindows()
