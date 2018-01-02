FROM ubuntu:14.04

RUN apt-get update && apt-get install wget -y
RUN wget http://s3.amazonaws.com/influxdb/influxdb_latest_amd64.deb
RUN dpkg -i influxdb_latest_amd64.deb
RUN rm -r /opt/influxdb/shared

VOLUME /opt/influxdb/shared

CMD /usr/bin/influxdb --pidfile /var/run/influxdb.pid -config /opt/influxdb/shared/config.toml

EXPOSE 8083
EXPOSE 8086
EXPOSE 8090
EXPOSE 8099
