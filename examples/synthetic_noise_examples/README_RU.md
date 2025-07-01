### Эксперимент по подбору архитектуры и оценки устойчивости к шуму

На изображениях представлены примеры зашумленных синтетическиз данных с уровнями шума 1%, 3%, 5%, 10%, 25% 50%:

<p float="left">
<img src="noise_examples/noise_0.gif" style="width:150px;">
<img src="noise_examples/noise_1.gif" style="width:150px;">
<img src="noise_examples/noise_3.gif" style="width:150px;">
<img src="noise_examples/noise_5.gif" style="width:150px;">
<img src="noise_examples/noise_10.gif" style="width:150px;">
<img src="noise_examples/noise_25.gif" style="width:150px;">
<img src="noise_examples/noise_50.gif" style="width:150px;">
</p>

- Для запуска примера требуется запустить сетку экспериментов 
с обучением различных архитектур, собранных с помощью библиотеки -  [``` grid_train.py```](grid_train.py).

- Обратите внимание, что вспомогательные функции, реализующие генерацию 
синтетических данных и логирование реализованы в файле [``` tools.py```](tools.py).

- Для оценки результатов обучения архитектур реализован блокнот с комментариями 
к ячейкам [``` results.ipynb```](results.ipynb).
