FROM ubuntu:14.04

RUN apt-get update && apt-get install libcap2-bin mumble-server -y

ADD ./mumble-server.ini /etc/mumble-server.ini

CMD /usr/sbin/murmurd
