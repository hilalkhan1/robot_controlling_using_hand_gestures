import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import pybullet as p

model = load_model("model.h5")  
label = np.load("labels.npy")

holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils  

cap = cv2.VideoCapture(0)

threshold = 0.5  # Adjust the threshold as per your requirements

# Set up PyBullet simulation
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")
robotId = p.loadURDF("hrobot.urdf", [0, 0, 0.1])

# Define gesture control mappings
GESTURES = {
    'MoveForward': [0, 0.1],     # Move forward
    'MoveBackWard': [0, -0.1],  # Move backward
    'Move Right': [0.1, 0],    # Move left
    'Move Left': [-0.1, 0],    # Move right
}

while True:
    lst = []
    
    success, img = cap.read()
    img = cv2.flip(img, 1)
    
    res = holis.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if res.left_hand_landmarks:
        for i in res.left_hand_landmarks.landmark:
            lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
            lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
    else:
        for i in range(42):
            lst.append(0.0)
            
    if res.right_hand_landmarks:
        for i in res.right_hand_landmarks.landmark:
            lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
            lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
    else:
        for i in range(42):
            lst.append(0.0)
    
    lst = np.array(lst).reshape(1, -1)
    lst = lst / np.max(np.abs(lst))  # Normalize the input data
    
    predictions = model.predict(lst)
    pred_index = np.argmax(predictions)
    pred_prob = predictions[0][pred_index]
    
    if pred_prob > threshold:
        pred = label[pred_index]
    else:
        pred = "None"
    
    dx, dy = GESTURES.get(pred, [0, 0])
    if dx != 0 or dy != 0:
        pos, orn = p.getBasePositionAndOrientation(robotId)
        new_pos = [pos[0] + dx, pos[1] + dy, pos[2]]
        p.resetBasePositionAndOrientation(robotId, new_pos, orn)
    
    cv2.putText(img, f"Prediction: {pred}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    
    drawing.draw_landmarks(img, res.left_hand_landmarks, holistic.HAND_CONNECTIONS) 
    drawing.draw_landmarks(img, res.right_hand_landmarks, holistic.HAND_CONNECTIONS)    
    
    cv2.imshow("Image", img)
    
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        cap.release()
