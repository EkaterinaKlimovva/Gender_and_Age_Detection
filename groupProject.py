import cv2
import os
import numpy as np
import concurrent.futures
import time

# Path to the directory with images
image_dir = 'dataset'

# Paths to the models and configuration files
face_prototxt = "weight/opencv_face_detector.pbtxt"
face_model = "weight/opencv_face_detector_uint8.pb"
age_prototxt = "weight/age_deploy.prototxt"
age_model = "weight/age_net.caffemodel"
gender_prototxt = "weight/gender_deploy.prototxt"
gender_model = "weight/gender_net.caffemodel"

# Mean values for image preprocessing
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Categories of age and gender
age_categories = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_categories = ['Male', 'Female']

# Loading the models
face_net = cv2.dnn.readNet(face_model, face_prototxt)
age_net = cv2.dnn.readNet(age_model, age_prototxt)
gender_net = cv2.dnn.readNet(gender_model, gender_prototxt)

# Function to process individual image
def process_image(image_path):
    start_time = time.time()
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    fr_cv = cv2.resize(image, (720, 640))
    
    # Face detection
    fr_h, fr_w, _ = fr_cv.shape
    blob = cv2.dnn.blobFromImage(fr_cv, 1.0, (300, 300), [104, 117, 123], True, False)
    face_net.setInput(blob)
    detections = face_net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detections[0, 0, i, 3] * fr_w)
            y1 = int(detections[0, 0, i, 4] * fr_h)
            x2 = int(detections[0, 0, i, 5] * fr_w)
            y2 = int(detections[0, 0, i, 6] * fr_h)
            faceBoxes.append([x1, y1, x2, y2])
    
    if not faceBoxes:
        return None

    # Selecting the detected face with the largest bounding box area
    faceBox = max(faceBoxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))

    # Face preprocessing for gender prediction
    face = fr_cv[max(0, faceBox[1]-15): min(faceBox[3]+15, fr_cv.shape[0]-1),
                 max(0, faceBox[0]-15): min(faceBox[2]+15, fr_cv.shape[1]-1)]
    if face.size == 0:
        return None
    
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    gender_net.setInput(blob)
    genderPreds = gender_net.forward()
    gender = gender_categories[genderPreds[0].argmax()]

    # Face preprocessing for age prediction
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    age_net.setInput(blob)
    agePreds = age_net.forward()
    age = age_categories[agePreds[0].argmax()]

    print(f"Age: {age}")
    print(f"Gender: {gender}")

    # Return the processing time
    return time.time() - start_time

# # Sequential Image Processing
# print("Sequential Image Processing...")
# sequential_start_time = time.time()
# sequential_total_processing_time = 0
# for image_name in os.listdir(image_dir):
#     image_path = os.path.join(image_dir, image_name)
#     sequential_processing_time = process_image(image_path)
#     if sequential_processing_time is not None:
#         sequential_total_processing_time += sequential_processing_time

# sequential_total_time = time.time() - sequential_start_time

# Parallel Image Processing
print("\nParallel Image Processing...")
start_time = time.time()
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = []
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        future = executor.submit(process_image, image_path)
        futures.append(future)

    total_processing_time = 0
    for future in concurrent.futures.as_completed(futures):
        processing_time = future.result()
        if processing_time is not None:
            total_processing_time += processing_time
    

# Printing the total processing time
parallel_total_time = time.time() - start_time

# print(f"Sequential Processing Time: {sequential_total_time} seconds")
# print(f"Total Sequential Processing Time: {sequential_total_processing_time} seconds")

print(f"Parallel Processing Time:  seconds")
print(f"Total Parallel Processing Time: {total_processing_time} seconds")
