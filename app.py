import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps  
import numpy as np

# 파일 업로더를 만듭니다.
file = st.file_uploader('이미지를 업로드해주세요', type=['jpg','png'])

np.set_printoptions(suppress=True)

# 모델을 가져옵니다.
model = load_model("keras_Model.h5", compile=False)

# 라벨을 읽습니다.
class_names = open("labels.txt", "r", encoding='UTF-8').readlines()

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

if file is None:
  st.text('이미지를 업로드하세요')
else:   
    # 파일을 엽니다.
    image = Image.open(file).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    image_array = np.asarray(image)  
    
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
   
    data[0] = normalized_image_array
    st.image(image, use_column_width = True)
    prediction = model.predict(data)
    index = np.argmax(prediction)
    # class_name는 1 고양이와 같이 문자열입니다.
    class_name = class_names[index]
    confidence_score = prediction[0][index]  
    confidence_score = round(confidence_score*100, 2)
    st.text(f'{class_name.split(" " )[1]} {confidence_score}%')

