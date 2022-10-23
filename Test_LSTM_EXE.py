#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras


# In[2]:


model = keras.models.load_model('saved_model_latest/LSTM')

# Check its architecture
model.summary()


# In[3]:


import cv2
import mediapipe as mp
import numpy as np


# In[4]:


mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils # For drawing keypoints
points = mpPose.PoseLandmark # Landmarks


# In[5]:


def getarray(landmark): 
  r=[]
  for i in landmark:
    r.append([i.x,i.y])
  r=np.array(r)
  r=r.flatten()
  return r


# In[6]:


def test(path):
  cap = cv2.VideoCapture(path)
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  print(total_frames)
  arr=[]
  for fno in range(0, total_frames,1):
    print(f"frame number is{fno}")
    #print(fno)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
    _, image = cap.read()
    img = image.copy()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    landmark=results.pose_landmarks.landmark
    r=getarray(landmark)
    #print(r)
    if(fno<14):
      arr.append(r)
    elif(fno==14):
      arr.append(r)
      k=np.array(arr).reshape(-1,15,66)
      print(model.predict(k)[0])
    else:
      del arr[0]
      arr.append(r)
      k=np.array(arr).reshape(-1,15,66)
      p=model.predict(k)[0][0]
#       if((fno+1)%30==0):
      p=round(p,2)
      p=str(p)
      print(p)
      cv2.putText(image, p, (500,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness = 3)
      cv2.imshow('test',image)
      if cv2.waitKey(10) & 0xFF==ord('q'):
          break
    #out.write(image)

  cap.release()
  cv2.destroyAllWindows()


# In[9]:


test("D:/Capstone/test/preetam saha not crossing the road.mp4")


# In[11]:


# Import the required Libraries
from tkinter import *
from tkinter import ttk, filedialog
from tkinter.filedialog import askopenfile
import os

# Create an instance of tkinter frame
win = Tk()

# Set the geometry of tkinter frame
win.geometry("700x350")

def open_file():
   file = filedialog.askopenfile()
   if file:
      filepath = os.path.abspath(file.name)
      test(filepath)

# Add a Label widget
label = Label(win, text="Click the Button to browse the Files", font=('Georgia 13'))
label.pack(pady=10)

# Create a Button
ttk.Button(win, text="Browse", command=open_file).pack(pady=20)

win.mainloop()


# In[12]:




# In[ ]:




