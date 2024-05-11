import cv2
import torch
from torchvision import transforms
import PIL
from resnet import  ResNetFaceModel
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
img_size = 100
threshold = 0.1
capture_reference = False
start_prediction = False

# 加载模型
model = ResNetFaceModel(num_classes=1)
model.load_state_dict(torch.load('model_41.pt'))
model = model.to(device)
model.eval()

def preprocess(image):
    preprocess_pipeline = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])
    image = PIL.Image.fromarray(image)
    preprocessed_image = preprocess_pipeline(image)
    return preprocessed_image

def predict(image):
    global model
    pred_image = preprocess(image)
    pred_image = torch.tensor(pred_image).unsqueeze(0).to(device)
    output = model(pred_image)
    return output

cap = cv2.VideoCapture(0)
reference_image = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 检查是否按下 'q' 键，如果按下 'q' 键，设置 reference_image
    if cv2.waitKey(1) & 0xFF == ord('q'):
        reference_image = frame
        print('q pressed')
        cv2.putText(frame, 'Reference image set', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_4)
        continue

    # 检查是否按下 's' 键，如果按下 's' 键，开始预测
    if cv2.waitKey(1) & 0xFF == ord('s'):
        print('s pressed')
        start_prediction = True

    if reference_image is not None and start_prediction:
        output1 = predict(reference_image)
        output2 = predict(frame)
        distance = torch.dist(output1, output2)
        print('distance',distance)
        if distance < threshold:
            cv2.putText(frame, 'The same person', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_4)
        else:
            cv2.putText(frame, 'Different people', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_4)

    cv2.imshow('Live Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

cap.release()
cv2.destroyAllWindows()