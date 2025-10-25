import cv2
import numpy as np

# --- Video Parameters ---
FILENAME = "test_video.mp4"
WIDTH = 400
HEIGHT = 300
DURATION_SECONDS = 5
FPS = 24  # Frames per second

# --- Object Parameters ---
SQUARE_SIZE = 50

def create_video():
    """Generates a simple video file using OpenCV."""

    # Define the codec and create a VideoWriter object
    # 'mp4v' is a common codec for .mp4 files.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(FILENAME, fourcc, FPS, (WIDTH, HEIGHT))

    if not out.isOpened():
        print("Error: Could not open video writer.")
        return

    total_frames = DURATION_SECONDS * FPS

    print(f"Generating video: '{FILENAME}'...")
    print(f"Dimensions: {WIDTH}x{HEIGHT}, Duration: {DURATION_SECONDS}s, FPS: {FPS}")

    for i in range(total_frames):
        # Create a black background frame
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

        # Calculate the square's position (moves from left to right)
        start_x = int((i / total_frames) * (WIDTH - SQUARE_SIZE))
        end_x = start_x + SQUARE_SIZE
        
        start_y = (HEIGHT // 2) - (SQUARE_SIZE // 2)
        end_y = start_y + SQUARE_SIZE

        # Draw a white rectangle on the frame
        cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255, 255, 255), -1)

        # Write the frame to the video file
        out.write(frame)
        
        # Optional: Print progress
        if (i + 1) % FPS == 0:
            print(f"  ... Wrote second { (i + 1) // FPS } / {DURATION_SECONDS}")


    # Release everything when the job is done
    out.release()
    print(f"\n✅ Video '{FILENAME}' created successfully!")
    print("You can now use this file to test your web application.")

if __name__ == '__main__':
    create_video()