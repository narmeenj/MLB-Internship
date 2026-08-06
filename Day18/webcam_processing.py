import os 
import cv2
import sys
import numpy as np

if not os.path.exists("Day18/output_videos/webcam_videos"):
    os.makedirs("Day18/output_videos/webcam_videos")
    
while True:
    option = int(input("""
   -----Webcam Processing Menu-----
    1. Live Mirrored Webcam Feed
    2. Real-Time Edge Detection (Side-by-Side)
    3. Record Live Webcam Video
    4. Exit
    
    Enter your choice: """))

    match option:
        case 1:
            capture = cv2.VideoCapture(0)
            window_name = "Mirrored Webcam Feed"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)
            
            while True:
                ret, frame = capture.read()
                if not ret:
                    print("Error: Could not read webcam frame.")
                    break
                
                mirrored_frame = cv2.flip(frame, 1)
                cv2.imshow(window_name, mirrored_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                
            capture.release()
            cv2.destroyAllWindows()        

        case 2:
            capture = cv2.VideoCapture(0)
            window_name = "Real-Time Pipeline: Original vs Canny Edges"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1200, 500)
            
            while True:
                ret, frame = capture.read()
                if not ret:
                    break
                
                mirrored_frame = cv2.flip(frame, 1)
                
                gray = cv2.cvtColor(mirrored_frame, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                edges = cv2.Canny(blurred, 50, 150)
                
                edges_3channel = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                
                combined_view = np.hstack((mirrored_frame, edges_3channel))
                cv2.imshow(window_name, combined_view)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                    
            capture.release()
            cv2.destroyAllWindows()

        case 3:
            capture = cv2.VideoCapture(0)
            
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = 20
            
            writer = cv2.VideoWriter(
                "Day18/output_videos/webcam_videos/webcam_recording.mp4",
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps, (width, height)
            )
            
            window_name = "Recording Webcam Feed... Press 'q' to Save"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)
            
            print("\nRecording started! Recording file saving to Day18/output_videos/webcam_videos/webcam_recording.mp4...")
            
            while True:
                ret, frame = capture.read()
                if not ret:
                    break
                
                mirrored_frame = cv2.flip(frame, 1)
                writer.write(mirrored_frame)
                cv2.imshow(window_name, mirrored_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            
            capture.release()
            writer.release()
            cv2.destroyAllWindows()
            print("Recording saved successfully!")

        case 4:
            print("System Exited!")
            sys.exit()
            
        case _:
            print("Invalid Option!")
