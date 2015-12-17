FROM busybox:ubuntu-14.04

MAINTAINER Jason Wilder "<jason@influxdb.com>"

# admin, http, udp, cluster, graphite, opentsdb, collectd
EXPOSE 8083 8086 8086/udp 8088 2003 4242 25826

WORKDIR /app

# copy binary into image
COPY influxd /app/

# Add influxd to the PATH
ENV PATH=/app:$PATH

# Generate a default config
RUN influxd config > /etc/influxdb.toml

# Use /data for all disk storage
RUN sed -i 's/dir = "\/.*influxdb/dir = "\/data/' /etc/influxdb.toml

VOLUME ["/data"]

ENTRYPOINT ["influxd", "--config", "/etc/influxdb.toml"]
