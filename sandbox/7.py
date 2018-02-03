import time
import cv2

video = cv2.VideoCapture('../input/GOPR0384.MP4')

while video.isOpened():
    success, frame = video.read()
    if success:
        cv2.rectangle(frame, (10, 10), (200, 200), (255, 0, 0), 20)
        cv2.imshow('Frame', frame)

        # cv2.imwrite("frame-%d.jpg" % int(time.time()), frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

# When everything done, release the video capture object
video.release()

# Closes all the frames
cv2.destroyAllWindows()
