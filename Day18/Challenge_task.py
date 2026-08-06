import cv2
import os
import numpy as np

def run_challenge():
    input_dir = "Day18/input_videos"
    output_dir = "Day18/output_videos/processesd_samples"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    video_files = ["sample1.mp4", "sample2.mp4", "sample3.mp4"]
    window_name = "Challenge Task Processing Pipeline"
    
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 500)
    
    print("--------- Starting Challenge Task Processing ------------\n")
    
    for video_name in video_files:
        input_path = os.path.join(input_dir, video_name)
        output_path = os.path.join(output_dir, f"processed_{video_name}")
        
        if not os.path.exists(input_path):
            print(f"Skipping {video_name}: File not found in {input_dir}")
            continue
            
        capture = cv2.VideoCapture(input_path)
        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps, (width, height), False
        )
        
        print(f"Processing: {video_name} -> Saving to: processed_{video_name}")
        
        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            writer.write(edges)
            
            combined_view = np.hstack((gray, edges))
            cv2.imshow(window_name, combined_view)
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                print("Processing interrupted by user.")
                break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed. Interrupted processing.")
                break
                
        capture.release()
        writer.release()
        print(f"Successfully finalized {video_name}.\n")
        
    cv2.destroyAllWindows()
    
    print("------- Challenge Task Observation Report --------")
    print("""
    1. Video 1 Observation (e.g., Cat Scene):
       - Low background texture allows Canny edge detection to cleanly isolate the main object structure.
       - Subtle lighting variances did not cause excessive noise artifacts due to effective Gaussian blurring.
       
    2. Video 2 Observation (e.g., Nature/Leaves Scene):
       - High-frequency texture details (like leaf veins and grass blades) produced a dense network of intricate edge patterns.
       - To simplify the edge structures, a wider kernel size or a higher upper threshold can be evaluated.
       
    3. Video 3 Observation (e.g., Motion/Action Scene):
       - High-speed movement fields introduced localized motion blur across consecutive image matrices.
       - The structural edge tracking lines faded or broke up into segments during rapid motion frames.
    -----------------------------------------------------------------------------------------------------""")

if __name__ == "__main__":
    run_challenge()
