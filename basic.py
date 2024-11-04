import numpy as np
import pandas as pd
import time
import re
import collections
from datetime import timedelta
from tqdm import tqdm

import threading
from concurrent.futures import ThreadPoolExecutor

def test_slice():
    a = [1, 2, 3, 4, 5]
    print(a[:-1])       # [1, 2, 3, 4]
    print(a[:5:2])      # [1, 3, 5]
    print(a[::-1])      # [5, 4, 3, 2, 1]
    print(a[3::-1])     # [4, 3, 2, 1]
    print(a[3:0:-1])    # [4, 3, 2]

    a = np.array(a)
    print(a[...])

    print(i for i in range(5))
    print([i for i in range(5)])

def test_zip():
    data = [(1, 'a'), (2, 'b'), (3, 'c')]
    print(*data)

    l1, l2 = zip(*data)
    print(l1)   # (1, 2, 3)
    print(l2)   # ('a', 'b', 'c')

    for x, y in zip(l1, l2):
        print(x, y)

def test_time():
    start = time.time()
    time.sleep(2)
    elapse = timedelta(seconds=int(time.time() - start))
    print(elapse)

    print(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))

def test_tqdm():
    for i in tqdm(range(10)):
        time.sleep(0.5)

    for i, item in enumerate(tqdm(range(10))):
        time.sleep(0.5)    

def test_collections():
    # Counter
    counter = collections.Counter()
    counter.update(["test", "test", 'me'])
    print(counter.items())

    # OrderedDict
    map = collections.OrderedDict()
    map['id'] = 100
    map['name'] = 'Tom'
    map['birth'] = '1999'
    map['address'] = 'xxxx'

    for k, v in map.items():
        print(k, v)

def test_regex():
    str = '<img src="http://www.image.com/test.jpg" alt="alt" title="title">'
    
    # greedy match
    match = re.search(r'img src="(.*)"', str)
    print(match.group(0))

    # lazy match
    match = re.search(r'img src="(.*?)"', str)
    print(match.group(0))

def test_pandas():
    # DataFrame Series
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['a', 'b', 'c'])
    print(df.iloc[1, :])
    print(df.loc[:, ['a', 'b']])

    print(type(df.iloc[1, :]))          # Series
    print(type(df.loc[:, ['a', 'b']]))  # DataFrame

    # read_json
    df = pd.read_json('res/site.json')
    print(df)

    # read_csv
    df = pd.read_csv("res/report.csv", delimiter=";")
    df = df.loc[:, ['reportTitle','reporter', 'reportTime', 'reportLocation']]

    with open('report.sql', mode='w', encoding="utf-8") as f:
        for i in range(20):
            report = df.iloc[i].values
            sql = "insert into report(title, speaker, time, location) values('{}', '{}', '{}', '{}');\n"\
                .format(report[0], report[1], report[2], report[3])
            f.write(sql)

def test_thread():

    class Task(threading.Thread):
        def __init__(self, name, lock) -> None:
            super().__init__()
            self.name = name
            self.lock = lock

        def run(self) -> None:
            self.lock.acquire()
            for i in range(5):
                print("{}: {}".format(self.name, i))
                time.sleep(0.5)
            self.lock.release()

    lock = threading.Lock()
    thread1 = Task("A", lock)
    thread2 = Task("B", lock)
    thread1.start()
    thread2.start()

if __name__ == '__main__':
    test_regex()