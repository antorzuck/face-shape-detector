#importing the libraries
import numpy as np
import cv2
import dlib
from sklearn.cluster import KMeans
import math
from math import degrees

# Load the image
imagepath = "face.jpg"
face_cascade_path = "haarcascade_frontalface_default.xml"
predictor_path = "shape_predictor_68_face_landmarks.dat"

faceCascade = cv2.CascadeClassifier(face_cascade_path)
predictor = dlib.shape_predictor(predictor_path)

image = cv2.imread(imagepath)
image = cv2.resize(image, (500, 500))
original = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gauss = cv2.GaussianBlur(gray, (3, 3), 0)
faces = faceCascade.detectMultiScale(
    gauss,
    scaleFactor=1.05,
    minNeighbors=5,
    minSize=(100, 100),
    flags=cv2.CASCADE_SCALE_IMAGE
)

print("found {0} faces!".format(len(faces)))

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    dlib_rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
    detected_landmarks = predictor(image, dlib_rect).parts()
    landmarks = np.matrix([[p.x, p.y] for p in detected_landmarks])

results = original.copy()

for (x, y, w, h) in faces:
    cv2.rectangle(results, (x, y), (x + w, y + h), (0, 255, 0), 2)
    temp = original.copy()
    forehead = temp[y:y + int(0.25 * h), x:x + w]
    rows, cols, bands = forehead.shape
    X = forehead.reshape(rows * cols, bands)

    kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)
    y_kmeans = kmeans.fit_predict(X)

    for i in range(0, rows):
        for j in range(0, cols):
            if y_kmeans[i * cols + j] == True:
                forehead[i][j] = [255, 255, 255]
            if y_kmeans[i * cols + j] == False:
                forehead[i][j] = [0, 0, 0]

    forehead_mid = [int(cols / 2), int(rows / 2)]
    lef = 0
    pixel_value = forehead[forehead_mid[1], forehead_mid[0]]

    for i in range(0, cols):
        if forehead_mid[0] - i >= 0:
            if not np.array_equal(forehead[forehead_mid[1], forehead_mid[0] - i], pixel_value):
                lef = forehead_mid[0] - i
                break
    left = [lef, forehead_mid[1]]

    rig = 0
    for i in range(0, cols):
        if forehead_mid[0] + i < cols:
            if not np.array_equal(forehead[forehead_mid[1], forehead_mid[0] + i], pixel_value):
                rig = forehead_mid[0] + i
                break
    right = [rig, forehead_mid[1]]

    line1 = np.subtract(right + y, left + x)[0]
    cv2.line(results, tuple(x + left), tuple(y + right), color=(0, 255, 0), thickness=2)

    linepointleft = (landmarks[1, 0], landmarks[1, 1])
    linepointright = (landmarks[15, 0], landmarks[15, 1])
    linepointtop = (landmarks[8, 0], y)
    linepointbottom = (landmarks[8, 0], landmarks[8, 1])

    line2 = np.subtract(linepointright, linepointleft)[0]
    line3 = np.subtract(linepointbottom, linepointtop)[1]
    line4 = np.subtract(linepointbottom, linepointtop)[1]

    similarity = np.std([line1, line2, line3])
    ovalsimilarity = np.std([line2, line4])

    ax, ay = landmarks[3, 0], landmarks[3, 1]
    bx, by = landmarks[4, 0], landmarks[4, 1]
    cx, cy = landmarks[5, 0], landmarks[5, 1]
    dx, dy = landmarks[6, 0], landmarks[6, 1]

    alpha0 = math.atan2(cy - ay, cx - ax)
    alpha1 = math.atan2(dy - by, dx - bx)
    alpha = alpha1 - alpha0
    angle = abs(degrees(alpha))
    angle = 180 - angle

    for i in range(1):
        if similarity < 10:
            if angle < 160:
                print('squared shape. Jawlines are more angular')
                break
            else:
                print('round shape. Jawlines are not that angular')
                break
        if line3 > line1:
            if angle < 160:
                print('triangle shape. Forehead is more wider')
                break
        if ovalsimilarity < 10:
            print('diamond shape. Line 2 & Line 4 are similar and Line 2 is slightly larger')
            break
        if line4 > line2:
            if angle < 160:
                print('rectangular. Face length is largest and jawlines are angular')
                break
            else:
                print('oblong. Face length is largest and jawlines are not angular')
                break
        print("Unable to determine face shape")

output = np.concatenate((original, results), axis=1)
cv2.imwrite('out.jpg', output)
