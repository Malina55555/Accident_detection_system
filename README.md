# Система детекции Дорожно-Транспортных Происшествий 

Проект разработанной системы детекции и анализа дорожно-транспортных происшествий посредством методов машинного обучения

Подготовлен студентом 4ого курса группы БВТ2203

Несмачной А.А.

## Установка

### 1. Клонирование репозитория

```bash
git clone <repository-url>
cd project_translator
```

### 2. Установка зависимостей

```bash
pip3 install -r /detector_service/requirements.txt
pip3 install -r /verification_service/requirements.txt
pip3 install -r /frontend_service/requirements.txt
```

### 3. Замена путей к файлам/папкам в следующих файлах:

#### /detector_service/config.py:
- YOLO_MODEL_PATH

#### /frontend_service/config.py:
- REPORTS_DIR

#### /verification_service/config.py:
- RTDETR_MODEL_PATH
- REPORT_SAVE_PATH
- FONTS_PATH

## Старт

Переходим по адресу http://127.0.0.1:5000/ в любом браузере

Запускаем автоматическую детекцию с камеры, нажав кнопку "Start Webcum"

либо

Запускаем автоматическую детекцию с видеофайла, выбрав файл, а затем нажав на кнопку "Start Video"

## Предпоказ
[видео из презентации.webm](https://github.com/user-attachments/assets/d17678c1-1f40-44c5-b668-31e724b7aade)


