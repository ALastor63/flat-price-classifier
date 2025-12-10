# Классификатор цен на квартиры

Нейросетевая модель для классификации цен на квартиры на три категории: дёшево, средне, дорого.

## Описание

Проект использует искусственную нейронную сеть (Keras/TensorFlow) для классификации цен на квартиры на основе следующих признаков:
- Площадь квартиры (м²)
- Количество комнат (1, 2 или 3)
- Этаж (1-25)

Модель классифицирует квартиры на три ценовых категории на основе квантилей цены.

## Технологии

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- TensorFlow/Keras

## Установка

1. Клонируйте репозиторий:

```bash
git clone https://github.com/Alastor63/flat-price-classifier.git
```

2. Перейдите в директорию проекта:

```bash
cd flat-price-classifier
```

3. Создайте и активируйте виртуальное окружение:

```bash
python -m venv venv
source venv/bin/activate  # Для Windows: venv\Scripts\activate
```

4. Установите зависимости:

```bash
pip install pandas numpy scikit-learn tensorflow
```

## Использование

Запустите скрипт:

```bash
python simple_flat_classifier.py
```

Скрипт выполнит следующие действия:
1. Сгенерирует искусственный датасет из 1000 квартир
2. Разделит данные на обучающую и тестовую выборки
3. Обучит нейронную сеть
4. Сохранит модель (`flat_price_model.h5`) и scaler (`scaler.pkl`)
5. Выполнит предсказания на примерах новых квартир

## Структура модели

Нейронная сеть состоит из:
- Входной слой (3 признака)
- Скрытый слой: 16 нейронов с активацией ReLU
- Скрытый слой: 8 нейронов с активацией ReLU
- Выходной слой: 3 нейрона с активацией softmax (для классификации)

## Результаты

После обучения модель выводит:
- Точность на тестовой выборке
- Предсказания для новых квартир с вероятностями для каждого класса

## Сохраненные файлы

После выполнения скрипта создаются:
- `flat_price_model.h5` - обученная модель
- `scaler.pkl` - нормализатор данных

## Пример использования сохраненной модели

```python
from tensorflow.keras.models import load_model
import pickle
import numpy as np

model = load_model('flat_price_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

new_flat = np.array([[50.0, 2, 10]])
new_flat_scaled = scaler.transform(new_flat)
prediction = model.predict(new_flat_scaled, verbose=0)
predicted_class = np.argmax(prediction, axis=1)[0]

class_names = {0: 'дёшево', 1: 'средне', 2: 'дорого'}
print(f'Предсказанный класс: {class_names[predicted_class]}')
```

