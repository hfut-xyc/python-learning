import threading
import time
from concurrent.futures import ThreadPoolExecutor

'''
thread mutex
'''
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