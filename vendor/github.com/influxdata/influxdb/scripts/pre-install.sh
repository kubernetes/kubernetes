#!/bin/bash

if [[ -d /etc/opt/influxdb ]]; then
    # Legacy configuration found
    if [[ ! -d /etc/influxdb ]]; then
	# New configuration does not exist, move legacy configuration to new location
	echo -e "Please note, InfluxDB's configuration is now located at '/etc/influxdb' (previously '/etc/opt/influxdb')."
	mv -vn /etc/opt/influxdb /etc/influxdb

	if [[ -f /etc/influxdb/influxdb.conf ]]; then
	    backup_name="influxdb.conf.$(date +%s).backup"
	    echo "A backup of your current configuration can be found at: /etc/influxdb/$backup_name"
	    cp -a /etc/influxdb/influxdb.conf /etc/influxdb/$backup_name
	fi
    fi
fi
