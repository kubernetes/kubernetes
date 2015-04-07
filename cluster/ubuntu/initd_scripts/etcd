#!/bin/sh
set -e

### BEGIN INIT INFO
# Provides:           etcd
# Required-Start:     $docker
# Required-Stop:      
# Should-Start:       
# Should-Stop:        
# Default-Start:      
# Default-Stop:       
# Short-Description:  Start distrubted key/value pair service
# Description:
#  http://www.github.com/coreos/etcd
### END INIT INFO

export PATH=/sbin:/bin:/usr/sbin:/usr/bin:/usr/local/sbin:/usr/local/bin:/opt/bin:

BASE=$(basename $0)

# modify these in /etc/default/$BASE (/etc/default/etcd)
ETCD=/opt/bin/$BASE
# This is the pid file managed by etcd itself
ETCD_PIDFILE=/var/run/$BASE.pid
ETCD_LOGFILE=/var/log/$BASE.log
ETCD_OPTS=""
ETCD_DESC="Etcd"

# Get lsb functions
. /lib/lsb/init-functions

if [ -f /etc/default/$BASE ]; then
	. /etc/default/$BASE
fi

# see also init_is_upstart in /lib/lsb/init-functions (which isn't available in Ubuntu 12.04, or we'd use it)
if false && [ -x /sbin/initctl ] && /sbin/initctl version 2>/dev/null | grep -q upstart; then
	log_failure_msg "$ETCD_DESC is managed via upstart, try using service $BASE $1"
	exit 1
fi

# Check etcd is present
if [ ! -x $ETCD ]; then
	log_failure_msg "$ETCD not present or not executable"
	exit 1
fi

fail_unless_root() {
	if [ "$(id -u)" != '0' ]; then
		log_failure_msg "$ETCD_DESC must be run as root"
		exit 1
	fi
}

ETCD_START="start-stop-daemon \
--start \
--background \
--quiet \
--exec $ETCD \
--make-pidfile \
--pidfile $ETCD_PIDFILE \
-- $ETCD_OPTS \
>> $ETCD_LOGFILE 2>&1"

ETCD_STOP="start-stop-daemon \
--stop \
--pidfile $ETCD_PIDFILE"

case "$1" in
	start)
		fail_unless_root
		log_begin_msg "Starting $ETCD_DESC: $BASE"
        $ETCD_START
		log_end_msg $?
		;;

	stop)
		fail_unless_root
		log_begin_msg "Stopping $ETCD_DESC: $BASE"
        $ETCD_STOP
		log_end_msg $?
		;;

	restart | force-reload)
		fail_unless_root
		log_begin_msg "Restarting $ETCD_DESC: $BASE"
        $ETCD_STOP
        $ETCD_START
		log_end_msg $?
		;;

	status)
		status_of_proc -p "$ETCD_PIDFILE" "$ETCD" "$ETCD_DESC"
		;;

	*)
		echo "Usage: $0 {start|stop|restart|status}"
		exit 1
		;;
esac
