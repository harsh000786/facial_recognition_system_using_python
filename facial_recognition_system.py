import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

image1 = face_recognition.load_image_file("faces/image1.png")
image1_encoding = face_recognition.face_encodings(image1)[0]
image2 = face_recognition.load_image_file("faces/image2.jpg")
image2_encoding = face_recognition.face_encodings(image2)[0]

known_face_encodings = [image1_encoding, image2_encoding]
known_face_names = ["img1", "img2"]

students = known_face_names.copy()

face_locations = []
face_encodings = []

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

attendance_marked = {}

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Failed to grab frame")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")

            if name not in attendance_marked:
                lnwriter.writerow([name, current_time])
                attendance_marked[name] = current_time

            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 100)
            fontScale = 1.5
            fontColor = (255, 0, 0)
            thickness = 3
            lineType = 2
            cv2.putText(frame, name + " Present", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

    cv2.imshow("Attendance", frame)

    if cv2.waitKey(10) & 0xFF == ord("q"):
        print("Exiting...")
        break

video_capture.release()
f.close()
cv2.destroyAllWindows()

