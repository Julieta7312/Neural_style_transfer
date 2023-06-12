import cv2
import os, time

VIDEO_DIR = './sources/RosaLinnArmeniaEurovision2022.mp4' # File to capture
IMAGES_PATH = './to_yolo/dataset/images' # Dir to save captured images


def filter_frames(img_num, mod=50):
    if img_num % mod == 0:
        return True
    else:
        return False


def del_imgs(mod=50):
    frames = os.listdir(IMAGES_PATH)
    for fr in frames:
        rem_fr = fr.replace('.jpg', '')
        if rem_fr.isnumeric() and (int(rem_fr) % mod != 0):
            if os.path.exists(f"{IMAGES_PATH}/{fr}"):
                os.remove(f"{IMAGES_PATH}/{fr}")


def capture_frames(fps=1/60):
    
    cap = cv2.VideoCapture(VIDEO_DIR) 
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2) 
    fps_ms = int(fps * 1000)
    
    img_num = 0
    while True:
        if cap.isOpened():        
            ret, frame = cap.read()
            imgname = os.path.join(IMAGES_PATH, f"{img_num}.jpg")
            if ret:                
                cv2.imshow('Video Capture', frame)
                if filter_frames(img_num):
                    cv2.imwrite(imgname, frame)
            else:
                break
            if cv2.waitKey(fps_ms) & 0xFF == ord('q'):
                break
            img_num += 1
            time.sleep(fps)
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":

    capture_frames()
    
