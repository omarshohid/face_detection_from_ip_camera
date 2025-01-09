import cv2
import face_recognition as fr
import numpy as np
import os

# Path to the images folder
images_path = "images"

# Load all known face images and encodings
known_face_encodings = []
known_face_names = []

for file_name in os.listdir(images_path):
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(images_path, file_name)
        try:
            # Load the image and encode it
            image = fr.load_image_file(image_path)
            face_encoding = fr.face_encodings(image)[0]
            known_face_encodings.append(face_encoding)
            # Use the file name (without extension) as the person's name
            known_face_names.append(os.path.splitext(file_name)[0])
        except IndexError:
            print(f"Warning: No face found in the image '{file_name}'. Skipping it.")

if not known_face_encodings:
    print("Error: No valid faces found in the images folder.")
    exit()

# Start video capture
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Error: Could not open video capture.")
    exit()

print("Press 'q' to exit the application.")

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert frame from BGR to RGB for face_recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face locations and encodings
    fc_locations = fr.face_locations(rgb_frame)
    fc_encodings = fr.face_encodings(rgb_frame, fc_locations)

    # Process each face detected in the frame
    for (top, right, bottom, left), face_encoding in zip(fc_locations, fc_encodings):
        # Compare face encodings with the known faces
        matches = fr.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Use the face with the smallest distance as the best match
        face_distances = fr.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a rectangle around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw a label with the name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.75, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('Face Recognition System', frame)

    # Exit the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
video_capture.release()
cv2.destroyAllWindows()
