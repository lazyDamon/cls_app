import torch
from torchvision import models, transforms

from PIL import Image

import streamlit as st


def Inference(imgPath):
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    model.eval()

    preprocess = weights.transforms()

    img = Image.open(imgPath)
    batch = preprocess(img).unsqueeze(0)
    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]

    return category_name, score


st.title('分类任务')
file = st.file_uploder("加载一张图片", type='jpg')
if file is not None:
    img = Image.open(file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    st.write("请稍后...")
    category, score = Inference(file)
    st.success('successful prediction')
    st.write("类别： ", category, "   ", "得分： ", score)

