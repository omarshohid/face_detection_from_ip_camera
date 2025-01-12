import imouapi.api
import aiohttp
import asyncio
import sys
import cv2
import numpy as np
import face_recognition as fr
import os
from dotenv import load_dotenv
import threading

load_dotenv()

# Check if running on Windows and adjust the event loop policy
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Retrieve the app_id and app_secret from environment variables
app_id = os.getenv("APP_ID")
app_secret = os.getenv("APP_SECRET")

# Check if app_id and app_secret were loaded successfully
if not app_id or not app_secret:
    print("Error: Please ensure APP_ID and APP_SECRET are set in the .env file.")
    sys.exit(1)


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


# Create an aiohttp session object
async def create_session():
    return await aiohttp.ClientSession().__aenter__()

# Create an Imou API client with the session
async def get_all_devices():
    try:
        # Create a session using aiohttp
        async with await create_session() as session:
            # Initialize the API client with the session
            api_client = imouapi.api.ImouAPIClient(app_id, app_secret, session)

            # Synchronously connect to the API
            await api_client.async_connect()
            print("Connected to API")

            # Get all device info
            devices_info = await api_client.async_api_deviceBaseList()

            if devices_info:
                print("List of Devices:")
                for device in devices_info['deviceList']:
                    print(f"Device ID: {device['deviceId']}")
                    print(f"Device Channels: {[channel['channelName'] for channel in device['channels']]}")

                    # Get live video stream for the device
                    for channel in device['channels']:
                        video_url = await get_video_stream(api_client, device['deviceId'], channel['channelId'])
                        if video_url:
                            print(f"Live Stream URL for Channel {channel['channelName']}: {video_url}")
                            # Display the video in a cv2 window
                            # display_video(video_url)
                            # Start a separate thread to display video
                            threading.Thread(target=display_video, args=(video_url,)).start()
                        else:
                            print(f"No live stream URL found for Channel {channel['channelName']}.")
                    print('-' * 50)
            else:
                print("No devices found.")

    except Exception as e:
        print(f"Error retrieving devices: {e}")

# Fetch the live video stream URL for a given device and channel
async def get_video_stream(api_client, device_id, channel_id):
    try:
        # Fetch the live stream URL using the Imou API
        stream_url_info = await api_client.async_api_getLiveStreamInfo(device_id)
        print(f"Stream URL Info for Device {device_id}, Channel {channel_id}: {stream_url_info}")

        if stream_url_info and 'streams' in stream_url_info:
            # Extracting stream URLs for the channels
            for stream in stream_url_info['streams']:
                if stream['status'] == '1':  # Checking if the stream is active
                    return stream['hls']  # Return the live stream URL
        return None
    except Exception as e:
        print(f"Error fetching stream URL for device {device_id}, channel {channel_id}: {e}")
        return None

# Display the live stream video in a cv2 window
def display_video(video_url):
    cap = cv2.VideoCapture(video_url)  # Open the video stream using the URL

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        ret, frame = cap.read()  # Read a frame from the stream
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
        # if not ret:
        #     print("Failed to grab frame.")
        #     break

        # # Display the frame
        # cv2.imshow("Live Stream", frame)

        # # Wait for the user to press 'q' to exit the video window
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release()
    cv2.destroyAllWindows()

# Run the function to get all devices and their live video streams
def main():
    try:
        asyncio.run(get_all_devices())  # Use asyncio.run() instead of event loop
    except Exception as e:
        print(f"Error in main: {e}")

# Run the main function
if __name__ == "__main__":
    main()
