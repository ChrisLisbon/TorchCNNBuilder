# Примеры использования

## Быстрый старт

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/ChrisLisbon/TorchCNNBuilder.git
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

## Типовые проблемы

1. В случае возникнования ошибки ModuleNotFoundError: No module named 'torch':
- проверьте что команда pip list | grep подтверждает, что torch установлен
- в случае, когда torch присутствует в выдачу команды, а ошибка сохраняется - проверьте, 
что виртуальное окружение python (venv) создано и используется как для запуска зависимостей, так и при запуске кода.

Виртуальное окружение может быть создано следующим образом:

```
python -m venv venv
source venv/bin/activate
```

При сохранении ошибки необходимо пересоздать виртуальное окружения с нуля и повторить установки TorchCNNBulder и его зависимостей.

2. Ошибка вида "Minimum and Maximum cuda capability supported by this version of PyTorch is (7.0) - (12.0)" может быть 
вызвана использование неподдерживаемой видеокарты. TorchCNNBuidler поддерживает работы только с видеокартами, 
обеспечивающими работу с версиями CUDA >=7. В иных случаях возможен только запуск в CPU-режиме.

Для запуска примеров необходимо заменить строку device=”cuda” на device=”cpu”.