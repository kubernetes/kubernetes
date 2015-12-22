#!/bin/sh

# Copy existing configuration if pre-existing installation is found
test -f /etc/opt/influxdb/influxdb.conf
if [ $? -eq 0 ]; then
    mkdir /etc/influxdb
    cp -a /etc/opt/influxdb/* /etc/influxdb/
fi

exit
