FROM redis:2.8
RUN apt-get update
RUN apt-get install -yy -q python

COPY redis-master.conf /redis-master/redis.conf
COPY redis-slave.conf /redis-slave/redis.conf
COPY run.sh /run.sh
COPY sentinel.py /sentinel.py

CMD [ "/run.sh" ]
ENTRYPOINT [ "sh", "-c" ]
