N-граммная модель

Предлагается следующий простой и эффективный метод: n-граммная языковая модель.


Будем называть префиксом несколько подряд идущих слов. По большому тексту для каждого префикса выбранного размера (например, 1 или 2 слова) составляется список слов, которые могут идти после него. В ML это называется обучением. Хранить эту информацию можно в виде словаря: 

(предыдущее слово префикса 1, ..., предыдущее слово префикса N) : 
[(следующее слово, вероятность), (другое следующее слово, вероятность), ...] 

Например для префикса (“мама”, “мыла”) вероятное следующее слово будет “раму”.


При самой генерации следует выбрать начальные слова предложения. Все следующие слова последовательно случайно выбираются из списка слов, идущих после префикса. Выбор из этого словаря можно делать через `np.random.choice`. В ML это называется сэмплированием.

Интерфейс

Основной код должен быть разбит на две части: обучение и генерация.


Обучение:

    Считать входные данные из файлов.
    Очистить тексты: выкидывать неалфавитные символы, приводить к lowercase.
    Разбить тексты на слова (в ML это называется токенизацией).
    Сохранить модель в каком-нибудь формате, который позволяет восстановить её в утилите генерации.


Параметры `train.py`:

    `--input-dir`  путь к директории, в которой лежит коллекция документов. Если данный аргумент не задан, считать, что тексты вводятся из stdin.
    `--model`  путь к файлу, в который сохраняется модель.


Генерация:

    Загрузить модель.
    Инициализировать её каким-нибудь сидом.
    Сгенерировать последовательность нужной длины.
    Вывести её на экран.


Параметры `generate.py`:

    `--model`  путь к файлу, из которого загружается модель.
    `--prefix`  необязательный аргумент. Начало предложения (одно или несколько слов). Если не указано, выбираем начальное слово случайно из всех слов.
    `--length`  длина генерируемой последовательности.


Детали реализации:

    Удобно реализовать консольный интерфейс через `argparse`.
    Для работы с текстами может пригодиться библиотека регулярных выражений `re`.
    Соблюдайте, пожалуйста, pep8. Пишите хороший код. Следуй те принципам ООП. Оберните модель в класс, у которого будет методы `fit` и `generate` .
    Для сохранения модели удобно использовать `pickle` или `dill` .
    Создайте только `train.py`, `generate.py`, папку `data` и, возможно, обученную модель, как `model.pkl`.
    Обучать модель можно на чем угодно: от «Войны и мира» и «Гарри Поттера» до текстов ATL и Pyrokinesis.
    Можно разрабатывать и тестировать алгоритм на малом корпусе текстов. Финальный алгоритм желательно обучить на большом корпусе для лучшего качества и большего разнообразия сгенерированных текстов.
    Нельзя использовать внешние библиотеки. Пользуйтесь тем, что есть по умолчанию в Python, и тем, что указано в секциях выше.