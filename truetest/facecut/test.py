import torch
import cv2
from torchvision import transforms
import PIL.Image as Image
from resnet import ResNetFaceModel
import numpy as np
from facenet_pytorch import MTCNN
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
img_size = 100
threshold = 30  # Maybe you need adjust this threshold
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

model = ResNetFaceModel()
model.load_state_dict(torch.load('model_41.pt', map_location=torch.device('cpu')))
model = model.to(device)
model.eval()
mtcnn = MTCNN()
def crop_face(img):
    boxes, _ = mtcnn.detect(img)
    if boxes is not None:
        for box in boxes:
            x, y, w, h = box
            x = int(x)
            y = int(y)
            w = int(w - x)
            h = int(h - y)
            img = img[y:y+h, x:x+w]
    return img

def predict(image):
    global model
    image = np.array(image)
    image = image[:, :, ::-1].copy()
    image = crop_face(image)
    image = Image.fromarray(image)
    preprocessed_face = preprocess(image)
    preprocessed_face = torch.stack([preprocessed_face]).to(device)
    if preprocessed_face.nelement() > 0:
        output = model(preprocessed_face)
    else:
        output = None
    return output

def preprocess(image):
    preprocess_pipeline = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    preprocessed_image = preprocess_pipeline(image)
    return preprocessed_image

prev_x, prev_y, prev_w, prev_h = 0, 0, 0, 0
alpha = 0.5


def draw_rectangle_and_crop_face(img):
    clone = img.copy()
    boxes, _ = mtcnn.detect(img)
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)
        clone = clone[int(y):int(h), int(x):int(w)]
    return clone, img

cap = cv2.VideoCapture(0)
reference_image = None
start_prediction = False  # 这里定义 start_prediction

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        reference_image, _ = draw_rectangle_and_crop_face(frame.copy())
        print('q pressed')
        cv2.putText(frame, 'Reference image set', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_4)
        continue

    if cv2.waitKey(1) & 0xFF == ord('s'):
        print('s pressed')
        start_prediction = True

    cropped_frame, frame = draw_rectangle_and_crop_face(frame)  # 注意这里我们使用了新的函数处理帧

    if reference_image is not None and start_prediction:
        output1 = predict(reference_image)
        output2 = predict(cropped_frame)  # 注意这里我们将处理过的帧输入到 predict 函数

        distance = torch.dist(output1, output2)
        print('distance', distance)
        if distance < threshold:
            cv2.putText(frame, 'The same person', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_4)
        else:
            cv2.putText(frame, 'Different people', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_4)

    cv2.imshow('Live Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

cap.release()
cv2.destroyAllWindows()