#!/bin/bash

BIN_DIR=/usr/bin
DATA_DIR=/var/lib/influxdb
LOG_DIR=/var/log/influxdb
SCRIPT_DIR=/usr/lib/influxdb/scripts
LOGROTATE_DIR=/etc/logrotate.d

function install_init {
    cp -f $SCRIPT_DIR/init.sh /etc/init.d/influxdb
    chmod +x /etc/init.d/influxdb
}

function install_systemd {
    cp -f $SCRIPT_DIR/influxdb.service /lib/systemd/system/influxdb.service
    systemctl enable influxdb
}

function install_update_rcd {
    update-rc.d influxdb defaults
}

function install_chkconfig {
    chkconfig --add influxdb
}

id influxdb &>/dev/null
if [[ $? -ne 0 ]]; then
    useradd --system -U -M influxdb -s /bin/false -d $DATA_DIR
fi

chown -R -L influxdb:influxdb $DATA_DIR
chown -R -L influxdb:influxdb $LOG_DIR

# Add defaults file, if it doesn't exist
if [[ ! -f /etc/default/influxdb ]]; then
    touch /etc/default/influxdb
fi

# Remove legacy symlink, if it exists
if [[ -L /etc/init.d/influxdb ]]; then
    rm -f /etc/init.d/influxdb
fi

# Distribution-specific logic
if [[ -f /etc/redhat-release ]]; then
    # RHEL-variant logic
    which systemctl &>/dev/null
    if [[ $? -eq 0 ]]; then
	install_systemd
    else
	# Assuming sysv
	install_init
	install_chkconfig
    fi
elif [[ -f /etc/debian_version ]]; then
    # Debian/Ubuntu logic
    which systemctl &>/dev/null
    if [[ $? -eq 0 ]]; then
	install_systemd
    else
	# Assuming sysv
	install_init
	install_update_rcd
    fi
elif [[ -f /etc/os-release ]]; then
    source /etc/os-release
    if [[ $ID = "amzn" ]]; then
	# Amazon Linux logic
	install_init
	install_chkconfig
    fi
fi
