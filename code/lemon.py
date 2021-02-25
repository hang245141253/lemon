import os
import cv2
import numpy as np
from PIL import Image
from paddlelite.lite import *


def preprocess(img):
    '''
    预测图片预处理
    '''
    #resize
    img = img.resize((224, 224), Image.BILINEAR) #Image.BILINEAR双线性插值
    img = np.array(img).astype('float32')
    
    # HWC to CHW 
    img = img.transpose((2, 0, 1))
    
    #Normalize
    img = img / 255         #像素值归一化
    mean = [0.31169346, 0.25506335, 0.12432463]   
    std = [0.34042713, 0.29819837, 0.1375536]
    img[0] = (img[0] - mean[0]) / std[0]
    img[1] = (img[1] - mean[1]) / std[1]
    img[2] = (img[2] - mean[2]) / std[2]
    
    return img

def run(image, predictor):
    '''
    执行预测
    '''
    image_data = np.array(preprocess(image)).flatten().tolist()

    input_tensor = predictor.get_input(0)
    input_tensor.resize([1, 3, 224, 224])
    input_tensor.set_float_data(image_data)

    predictor.run()

    output_tensor = predictor.get_output(0)
    
    lab = np.argmax(output_tensor.numpy())  #argmax():返回最大数的索引
    print("result: {}".format(label_list[lab]))


if __name__ == '__main__':
    
    config = MobileConfig()
    config.set_model_from_file('./models/model.nb')
    predictor = create_paddle_predictor(config)
    label_list = ['0:優良', '1:良', '2:加工品', '3:規格外']
    
    cap=cv2.VideoCapture(-1)
    cap.set(3,224)
    cap.set(4,224)
    if cap.isOpened() != True:
        print("Error: Please check the camera")
        exit(-1)
    while(1):
        _ , Vshow = cap.read()
        img = Image.fromarray(cv2.cvtColor(Vshow, cv2.COLOR_BGRA2RGB))#PIL图像和cv2图像转化
        run(img, predictor)
        
        cv2.imshow('Capture', Vshow)
        if cv2.waitKey(1)==ord('q'):
            print('完成')
            break

    cap.release()
    cv2.destroyAllWindows()
