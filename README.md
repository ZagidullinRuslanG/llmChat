# LLM интерфейс чат-бота с технологией RAG

Данный проект является прототипом чат-бота с имплементацией технологии RAG.

## Установка

1. Установка необходимой версии CUDA.

На тестовом компьютере были установлены следующие версии:
```bash
nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Tue_Feb_27_16:28:36_Pacific_Standard_Time_2024
Cuda compilation tools, release 12.4, V12.4.99
Build cuda_12.4.r12.4/compiler.33961263_0
```
```bash
nvidia-smi
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 552.44                 Driver Version: 552.44         CUDA Version: 12.4     |
+-----------------------------------------+------------------------+----------------------+
```

2. Установка необходимых библиотек:
```bash
pip install -r requirements.txt
```

3. Доустановить torch для работы с GPU (вариант для драйверов версии отличной от 12.4: [pytorch](https://pytorch.org/)):
```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

3. Переустановка llama-cpp на работу с GPU:
```bash
pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir -C cmake.args="-DGGML_CUDA=on"
```

### В случае установки не через llama-cpp, а Ollama
4. Установка [Ollama](https://ollama.com/) 

## Структура файлов проекта

- data - папка с данными, включая веса моделей
- ollama_RAG и llama-cpp_RAG - устаревший вариант проекта с экспериментальными ноутбуками.
- gradio_ui - файлы по текущему проекту.

## Конфигурация

В файле *gradio_ui/config.py* можно выставить необходимые настройки.

В качестве базовой модели предлагается использование llama3.1 8b:
```python
START_MODEL_NAME = 'llama3.1:latest'
```

Скачивание модели в ollama:

```bash
ollama pull llama3.1:latest
```

## Запуск

Для запуска моделей Ollama:
```bash
ollama serve
```

Запуск интерфейса пользователя происходит через файл *gradio_ui/main.py*




