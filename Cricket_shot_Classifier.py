import mediapipe as mp # Import mediapipe
import cv2
import numpy as np
import pickle 
import pandas as pd

with open('models\Stance.pkl', 'rb') as f:
   Stancemodel = pickle.load(f)

with open('models\FootEngagement.pkl', 'rb') as f:
   Foot_engagementmodel = pickle.load(f) 
   
with open('models\Defence_shot.pkl', 'rb') as f:
    Defencemodel = pickle.load(f)   

correctColor = (0, 256, 0)
inCorrectColor = (0, 0, 256)

mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions
cap = cv2.VideoCapture(0)

count = 0


def runDefencemodel():
   
    # Initiate holistic model

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():

            ret, frame = cap.read()
            
            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False        
            
            # Make Detections
            results = holistic.process(image)

            # Recolor image back to BGR for rendering
            image.flags.writeable = True   
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                # Extract Pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                        
                # Concate rows
                row = pose_row
        
                # Make Detections
                X = pd.DataFrame([row])

                shotDetection_class = Defencemodel.predict(X)[0]
                shotDetection_prob = Defencemodel.predict_proba(X)[0][1]
            
                if  shotDetection_prob < 0.80 or shotDetection_class == "Incorrect":

                    color = inCorrectColor
                    count = 0
                
                else:
                    color = correctColor
                    count += 1
                    
                if color == inCorrectColor:

                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)
                                        )
                    
                else:
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)
                                        )
                
                # count down
                cv2.circle(image, (image.shape[1]-75, 75), 50, (255, 0, 0), 5) # Outer circle
                cv2.ellipse(image, (image.shape[1]-75, 75), (50, 50), 0, 0, count*6, (0, 255, 0), 3) # Arc that represents count
                cv2.putText(image, str(count), (image.shape[1]-90, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA) # Write count inside

                cv2.rectangle(image, (0,0), (250, 60), (0, 250, 0), -1) 
                    
                    # Display Class
                cv2.putText(image, 'CLASS', (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, shotDetection_class.split(' ')[0], (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                            
                    # Display Probability
                cv2.putText(image, 'PROB', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(round( shotDetection_prob,2)), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.circle(image, (0, 100), 15, (0, 255, 0), -1)
                cv2.putText(image, '1', (3, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.circle(image, (0, 160), 15, (0, 255, 0), -1)
                cv2.putText(image, '2', (3, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.circle(image, (0, 220), 15,  (0, 255, 255), -1)
                cv2.putText(image, '3', (3, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                
                # Add text labels for each circle
                cv2.putText(image, 'Stance', (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(image, 'Foot', (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(image, 'Bat', (20, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (0, 255, 255), 1, cv2.LINE_AA)

            except:
                pass

            cv2.imshow('Live Webcam Feed', image)
            
            if cv2.waitKey(10) & 0xFF == ord('q' or 'Q') or count == 60 :   
                break
                    
        cap.release()
        cv2.destroyAllWindows()

def runFoot_engagementmodel():
    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():

            ret, frame = cap.read()
            
            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False        
            
            # Make Detections
            results = holistic.process(image)

            # Recolor image back to BGR for rendering
            image.flags.writeable = True   
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                # Extract Pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                        
                # Concate rows
                row = pose_row
        
                # Make Detections
                X = pd.DataFrame([row])

                shotDetection_class = Foot_engagementmodel.predict(X)[0]
                shotDetection_prob = Foot_engagementmodel.predict_proba(X)[0][1]
            
                if  shotDetection_prob < 0.40 or shotDetection_class == "Incorrect":

                    color = inCorrectColor
                    count = 0
                
                else:
                    color = correctColor
                    count += 1
                    
                if color == inCorrectColor:

                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)
                                        )
                    
                else:
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)
                                        )
                
                # count down
                cv2.circle(image, (image.shape[1]-75, 75), 50, (255, 0, 0), 5) # Outer circle
                cv2.ellipse(image, (image.shape[1]-75, 75), (50, 50), 0, 0, count*6, (0, 255, 0), 3) # Arc that represents count
                cv2.putText(image, str(count), (image.shape[1]-90, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA) # Write count inside

                cv2.rectangle(image, (0,0), (250, 60), (0, 250, 0), -1)
                    
                    # Display Class
                cv2.putText(image, 'CLASS', (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, shotDetection_class.split(' ')[0], (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                            
                    # Display Probability
                cv2.putText(image, 'PROB', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(round( shotDetection_prob,2)), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                cv2.circle(image, (0, 100), 15, (0, 255, 0), -1)
                cv2.putText(image, '1', (3, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.circle(image, (0, 160), 15,  (0, 255, 255), -1)
                cv2.putText(image, '2', (3, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.circle(image, (0, 220), 15, (0, 0, 255), -1)
                cv2.putText(image, '3', (3, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                
                # Add text labels for each circle
                cv2.putText(image, 'Stance', (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(image, 'Foot', (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image, 'Bat', (20, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            except:
                pass

            cv2.imshow('Live Webcam Feed', image)
            
            if cv2.waitKey(10) & 0xFF == ord('q' or 'Q') or count == 60 :
                runDefencemodel()
                break
        cap.release()
        cv2.destroyAllWindows()

def stance():  
    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

        while cap.isOpened():

            ret, frame = cap.read()
            
            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False        
            
            # Make Detections
            results = holistic.process(image)

            # Recolor image back to BGR for rendering
            image.flags.writeable = True   
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                # Extract Pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                        
                # Concate rows
                row = pose_row
        
                # Make Detections
                X = pd.DataFrame([row])

                shotDetection_class = Stancemodel.predict(X)[0]
                shotDetection_prob = Stancemodel.predict_proba(X)[0][1]
            
                if  shotDetection_prob < 0.80 or shotDetection_class == "Incorrect":

                    color = inCorrectColor
                    count = 0
                
                else:
                    color = correctColor
                    count += 1
                    
                if color == inCorrectColor:

                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)
                                        )
                    
                else:
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=4),
                                        mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2)
                                        )
                
                # count down
                cv2.circle(image, (image.shape[1]-75, 75), 50, (255, 0, 0), 5) # Outer circle
                cv2.ellipse(image, (image.shape[1]-75, 75), (50, 50), 0, 0, count*6, (0, 255, 0), 3) # Arc that represents count
                cv2.putText(image, str(count), (image.shape[1]-90, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA) # Write count inside

                cv2.rectangle(image, (0,0), (250, 60), (0, 250, 0), -1) 
                    
                # Display Class
                cv2.putText(image, 'CLASS', (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, shotDetection_class.split(' ')[0], (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                            
                # Display Probability
                cv2.putText(image, 'PROB', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(image, str(round( shotDetection_prob,2)), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                

                cv2.circle(image, (0, 100), 15, (0, 255, 255), -1)
                cv2.putText(image, '1', (3, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.circle(image, (0, 160), 15, (0, 0, 255), -1)
                cv2.putText(image, '2', (3, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.circle(image, (0, 220), 15, (0, 0, 255), -1)
                cv2.putText(image, '3', (3, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                
                # Add text labels for each circle
                cv2.putText(image, 'Stance', (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(image, 'Foot', (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(image, 'Bat', (20, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
  
            except:
                pass
            cv2.imshow('Live Webcam Feed', image)          
            if cv2.waitKey(10) & 0xFF == ord('q' or 'Q') or count == 60 :
                runFoot_engagementmodel()
                break
        cap.release()
        cv2.destroyAllWindows()
         


