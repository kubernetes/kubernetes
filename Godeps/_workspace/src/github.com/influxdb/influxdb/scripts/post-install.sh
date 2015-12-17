#!/bin/sh
BIN_DIR=/usr/bin
DATA_DIR=/var/lib/influxdb
LOG_DIR=/var/log/influxdb
SCRIPT_DIR=/usr/lib/influxdb/scripts
LOGROTATE_DIR=/etc/logrotate.d

if ! id influxdb >/dev/null 2>&1; then
        useradd --system -U -M influxdb -s /bin/false -d $DATA_DIR
fi
chmod a+rX $BIN_DIR/influx*

mkdir -p $LOG_DIR
chown -R -L influxdb:influxdb $LOG_DIR
mkdir -p $DATA_DIR
chown -R -L influxdb:influxdb $DATA_DIR

test -f /etc/default/influxdb || touch /etc/default/influxdb

# Remove legacy logrotate file
test -f $LOGROTATE_DIR/influxd && rm -f $LOGROTATE_DIR/influxd

# Remove legacy symlink
test -h /etc/init.d/influxdb && rm -f /etc/init.d/influxdb

# Systemd
if which systemctl > /dev/null 2>&1 ; then
    cp $SCRIPT_DIR/influxdb.service /lib/systemd/system/influxdb.service
    systemctl enable influxdb

# Sysv
else
    cp -f $SCRIPT_DIR/init.sh /etc/init.d/influxdb
    chmod +x /etc/init.d/influxdb
    if which update-rc.d > /dev/null 2>&1 ; then
        update-rc.d -f influxdb remove
        update-rc.d influxdb defaults
    else
        chkconfig --add influxdb
    fi
fi

exit
