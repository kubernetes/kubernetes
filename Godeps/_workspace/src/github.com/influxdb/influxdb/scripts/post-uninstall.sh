#!/bin/sh
rm -f /etc/default/influxdb

# Systemd
if which systemctl > /dev/null 2>&1 ; then
    systemctl disable influxdb
    rm -f /lib/systemd/system/influxdb.service
# Sysv
else
    if which update-rc.d > /dev/null 2>&1 ; then
        update-rc.d -f influxdb remove
    else
        chkconfig --del influxdb
    fi
    rm -f /etc/init.d/influxdb
fi

