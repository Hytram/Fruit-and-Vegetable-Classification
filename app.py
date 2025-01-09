from fastapi import FastAPI, File, UploadFile
import numpy as np
from PIL import Image, UnidentifiedImageError
import tensorflow as tf
import logging

labels = {
    0: 'apple', 
    1: 'banana', 
    2: 'beetroot', 
    3: 'bell pepper', 
    4: 'cabbage', 
    5: 'capsicum', 
    6: 'carrot', 
    7: 'cauliflower', 
    8: 'chilli pepper', 
    9: 'corn', 
    10: 'cucumber', 
    11: 'eggplant', 
    12: 'garlic', 
    13: 'ginger', 
    14: 'grapes', 
    15: 'jalepeno', 
    16: 'kiwi', 
    17: 'lemon', 
    18: 'lettuce', 
    19: 'mango', 
    20: 'onion', 
    21: 'orange', 
    22: 'paprika', 
    23: 'pear', 
    24: 'peas', 
    25: 'pineapple', 
    26: 'pomegranate', 
    27: 'potato', 
    28: 'raddish', 
    29: 'soy beans', 
    30: 'spinach', 
    31: 'sweetcorn', 
    32: 'sweetpotato', 
    33: 'tomato', 
    34: 'turnip', 
    35: 'watermelon'
}

# Cấu hình logging
logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Tải mô hình đã lưu
try:
    model = tf.keras.models.load_model("model.h5")
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    model = None  # Để tránh lỗi nếu mô hình không tải được


def preprocess_image(image: Image.Image):
    """
    Tiền xử lý ảnh để phù hợp với đầu vào của mô hình
    """
    image = image.resize((224, 224))  # Resize ảnh về kích thước phù hợp với mô hình
    image = np.array(image) / 255.0  # Chuẩn hóa ảnh (scale về [0, 1])
    image = np.expand_dims(image, axis=0)  # Thêm batch dimension
    return image


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Đảm bảo mô hình đã được tải
        if model is None:
            return {"error": "Model not loaded. Please check server configuration."}

        # Đọc file ảnh
        try:
            image = Image.open(file.file).convert("RGB")
        except UnidentifiedImageError:
            return {"error": "Invalid image format"}
        except Exception as e:
            logging.error(f"Unexpected error when reading image: {e}")
            return {"error": str(e)}

        # Tiền xử lý ảnh
        input_data = preprocess_image(image)
        logging.info("Image preprocessed successfully.")

        # Dự đoán với mô hình
        predictions = model.predict(input_data)
        predicted_class = np.argmax(predictions, axis=1)
        logging.info(f"Prediction: {predictions}")

        return {"predicted_class": labels[int(predicted_class[0])]}
    except Exception as e:
        logging.error(f"Error occurred during prediction: {e}")
        return {"error": str(e)}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Fruit and Vegetable Classification API. Use /predict/ to classify an image."}
