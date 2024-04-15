import cv2
import numpy as np

from tflite_runtime.interpreter import Interpreter


def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}


def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
  set_input_tensor(interpreter, image)
  interpreter.invoke()
  output_details = interpreter.get_output_details()[0]
  output = np.squeeze(interpreter.get_tensor(output_details['index']))

  if output_details['dtype'] == np.uint8:
    scale, zero_point = output_details['quantization']
    output = scale * (output - zero_point)

  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]


labels = load_labels("labels.txt")
interpreter = Interpreter("model.tflite")

interpreter.allocate_tensors()
_, height, width, _ = interpreter.get_input_details()[0]['shape']

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224) # Capture width 預設為 224
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224) # Capture height 預設為 224

cv2.namedWindow('JCCSSYL', cv2.WINDOW_NORMAL) # 創建一個視窗
cv2.resizeWindow('JCCSSYL', 320, 240) # 調整視窗大小以符合解析度

while True:
  ret,image_src = cap.read()

  frame_width = image_src.shape[1]
  frame_height = image_src.shape[0]

  cut_d = int((frame_width - frame_height) / 2)
  crop_img = image_src[0:frame_height, cut_d:(cut_d + frame_height)]

  image = cv2.resize(crop_img, (224,224), interpolation = cv2.INTER_AREA)

  results = classify_image(interpreter, image)
  label_id, prob = results[0]

  # print(labels[label_id],prob) # 顯示辨認度, 用後可 Remark

  cv2.putText(image_src, labels[label_id], (5,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA) # 顯示辨識到的 Label

  cv2.imshow('JCCSSYL', image_src) # 顯示運算結果

  if cv2.waitKey(1) & 0xFF == ord('q'): break # 當按下 Q 鍵就結束

cap.release() # 釋放系統資源
cv2.destroyAllWindows() # 關閉所有程式產生的視窗
