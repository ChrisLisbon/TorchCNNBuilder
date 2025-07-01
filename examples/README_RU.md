# Примеры использования

## Примеры работы с компонентами библиотеки
Примеры обращения в API для каждого подмодуля расположены в соответствующих файлах `ipynb`:
- [`torchcnnbuilder`](usage_examples/main_examples_ru.ipynb) - основные переменные и мат. аппартат сверток
- [`torchcnnbuilder.builder`](usage_examples/builder_examples_ru.ipynb) - API класса-строителя для создания последовательностей сверток
- [`torchcnnbuilder.models`](usage_examples/model_examples_ru.ipynb) - примеры моделей, созданных с помощью API класса-строителя 
- [`torchcnnbuilder.preprocess`](usage_examples/preprocess_examples_ru.ipynb) - функции подготовки данных

## Прикладные легковесные примеры 

Примеры сборки и обучения моделей:
- [`synthetic_noise_examples`](synthetic_noise_examples) - Эксперимент по подбору архитектуры и оценки устойчивости к шуму
- [`anime_example`](anime_example.ipynb) - Пример на данных с медиа-контентом
- [`moving mnist example`](moving_mnist_example.ipynb) - Пример для датасета MovingMnist


[`speed device test`](speed_device_test.py) - демонстрирует скорость сборки модели и использование cpu и CUDA

Пошаговое описание примеров представлено в ячейках ноутбуков и в виде комментариев к коду.

