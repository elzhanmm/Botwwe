from keras.models import load_model  # Для работы с библиотекой Keras нужен TensorFlow
from PIL import Image, ImageOps  # Нужно установить pillow или PIL
import numpy as np

# Отключаем функции нотации для удобства
np.set_printoptions(suppress=True)

# Загружаем модель
model = load_model("keras_model.h5", compile=False)

# Загружаем метки классов чтобы прописывал к какому классу принадлежит картинка
class_names = open("labels.txt", "r").readlines()

# Создаём массив нужной формы для подачи в модель Keras
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Путь к вашему изображению (вместо <IMAGE_PATH> поставить путь к вашему изображению)
image = Image.open("<IMAGE_PATH>").convert("RGB")

# Изменяем размер изображения до 224x224 и затем обрезаем с центра чтобы ваша модель понимала
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# Преобразуем изображение в numpy массив
image_array = np.asarray(image)

# Делаем изображение чётким
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Загружаем изображение в массив
data[0] = normalized_image_array

# Делаем моделью предсказание чтобы с помощью модели он распознал и написал из какого оно класса
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Выводим результат
print("Class:", class_name[2:], end="")
print("Confidence Score:", confidence_score)
