# Примеры использования

## Быстрый старт

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/yourusername/TorchCNNBuilder.git
   cd TorchCNNBuilder
   
2. Установка в режиме разработки (рекомендуется):
    ```bash
   pip install -e .

**Это позволит**: установить все необходимые зависимости, сделать пакет доступным системе, 
видеть изменения в коде без переустановки.

3. Только для запуска примеров (без разработки):

    ```bash
    pip install numpy torch matplotlib jupyter
   
4. ⚠️ Важно! Примеры используют относительные пути к данным. 
Всегда запускайте ноутбуки из корня репозитория:
    ```bash
    jupyter notebook examples/имя_примера.ipynb

## Примеры работы с компонентами библиотеки
Примеры обращения в API для каждого подмодуля расположены в соответствующих файлах `ipynb`:
- [`torchcnnbuilder`](usage_examples/main_examples_ru.ipynb) - основные переменные и мат. аппартат сверток
- [`torchcnnbuilder.builder`](usage_examples/builder_examples_ru.ipynb) - API класса-строителя для создания последовательностей сверток
- [`torchcnnbuilder.models`](usage_examples/model_examples_ru.ipynb) - примеры моделей, созданных с помощью API класса-строителя 
- [`torchcnnbuilder.preprocess`](usage_examples/preprocess_examples_ru.ipynb) - функции подготовки данных

## Прикладные легковесные примеры 

Примеры сборки и обучения моделей:
- [`synthetic_noise_examples`](synthetic_noise_examples) - Эксперимент по подбору архитектуры и оценки устойчивости к шуму
- [`anime_example`](anime_example_ru.ipynb) - Пример на данных с медиа-контентом
- [`ice concentration`](ice_concentration) - Пример на данных о концентрации льда
- [`moving mnist example`](moving_mnist_example_ru.ipynb) - Пример для датасета MovingMnist


[`speed device test`](speed_device_test.py) - демонстрирует скорость сборки модели и использование CPU и CUDA

Пошаговое описание примеров представлено в ячейках ноутбуков и в виде комментариев к коду.

