import cv2
import sys
import os
import numpy as np

if not os.path.exists("Day18/output_videos/processed_video"):
    os.makedirs("Day18/output_videos/processed_video")
    
while True:
    option = int(input("""
   -----Video Processing Menu-----
    1. Read and Display Video
    2. Show Properties of Video
    3. Convert Video to Grayscale
    4. Apply Canny Edge Detection
    5. Save Processed Video
    6. Live Webcam
    7. Exit
    
    Enter your choice: """))

    match option:
        case 1:
            capture = cv2.VideoCapture("Day18/input_videos/sample1.mp4")
            window_name = "Original Video"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)
            
            while capture.isOpened():
                ret, frame = capture.read()
                if not ret:
                    print("\nVideo ended successfully.")
                    break
                
                cv2.imshow(window_name, frame)
                
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                
            capture.release()
            cv2.destroyAllWindows()

        case 2:
            capture = cv2.VideoCapture("Day18/input_videos/sample1.mp4")
            fps = capture.get(cv2.CAP_PROP_FPS)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print("\n--- Properties ---")
            print("FPS:", fps)
            print("Width:", width)
            print("Height:", height)
            print("Total Frames:", total_frames)
            print("------------------")
            capture.release() 
           
        case 3:
            capture = cv2.VideoCapture("Day18/input_videos/sample1.mp4")
            window_name = "Grayscale"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)
            
            while True:
                ret, frame = capture.read()
                if not ret:
                    print("\nVideo ended successfully.")
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cv2.imshow(window_name, gray)
                
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                
            capture.release()
            cv2.destroyAllWindows()    
                                        
        case 4:
            capture = cv2.VideoCapture("Day18/input_videos/sample1.mp4")
            window_name = "Comparison: Grayscale vs Canny Edges"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1200, 500)
            
            while True:
                ret, frame = capture.read()
                if not ret:
                    print("\nVideo ended successfully.")
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                
                combined_view = np.hstack((gray, edges))
                cv2.imshow(window_name, combined_view)
                
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                    
            capture.release()
            cv2.destroyAllWindows() 
                            
        case 5:
            capture = cv2.VideoCapture("Day18/input_videos/sample1.mp4")
            fps = capture.get(cv2.CAP_PROP_FPS)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))   
            
            writer = cv2.VideoWriter(
                "Day18/output_videos/processed_video/processed_video.mp4",
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps, (width, height), False
            )    
            
            window_name = "Processed Edge Output Video"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)
            
            while True:
                ret, frame = capture.read()
                if not ret:
                    print("\nVideo saved and processing complete.")
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                writer.write(edges)
                cv2.imshow(window_name, edges)
                
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                
            capture.release()
            writer.release()
            cv2.destroyAllWindows()

        case 6:
            capture = cv2.VideoCapture(0)
            window_name = "Mirrored Webcam Feed"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 600)
            
            while True:
                ret, frame = capture.read()
                if not ret:
                    break
                
                mirrored_frame = cv2.flip(frame, 1)
                cv2.imshow(window_name, mirrored_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                
            capture.release()
            cv2.destroyAllWindows()        
            
        case 7:
            print("System Exited!")
            sys.exit()
         
        case _:
            print("Invalid Option!")
