import random
import syslog
import time

from celery_conf import add

while True:
    x = random.randint(1, 10)
    y = random.randint(1, 10)
    res = add.delay(x, y)
    time.sleep(5)
    if res.ready():
        res.get()
