#!/bin/sh
set -e

### BEGIN INIT INFO
# Provides:           flannel
# Required-Start:     $etcd
# Required-Stop:      
# Should-Start:       
# Should-Stop:        
# Default-Start:      
# Default-Stop:       
# Short-Description:  Start flannel networking service
# Description:
#  https://github.com/coreos/flannel
### END INIT INFO

export PATH=/sbin:/bin:/usr/sbin:/usr/bin:/usr/local/sbin:/usr/local/bin:/opt/bin:

BASE=$(basename $0)

# modify these in /etc/default/$BASE (/etc/default/flannel)
FLANNEL=/opt/bin/$BASE
# This is the pid file managed by kube-apiserver itself
FLANNEL_PIDFILE=/var/run/$BASE.pid
FLANNEL_LOGFILE=/var/log/$BASE.log
FLANNEL_OPTS=""
FLANNEL_DESC="Flannel"

# Get lsb functions
. /lib/lsb/init-functions

if [ -f /etc/default/$BASE ]; then
	. /etc/default/$BASE
fi

# see also init_is_upstart in /lib/lsb/init-functions (which isn't available in Ubuntu 12.04, or we'd use it)
if [ -x /sbin/initctl ] && /sbin/initctl version 2>/dev/null | grep -q upstart; then
	log_failure_msg "$FLANNEL_DESC is managed via upstart, try using service $BASE $1"
	exit 1
fi

# Check flanneld is present
if [ ! -x $FLANNEL ]; then
	log_failure_msg "$FLANNEL not present or not executable"
	exit 1
fi

fail_unless_root() {
	if [ "$(id -u)" != '0' ]; then
		log_failure_msg "$FLANNEL_DESC must be run as root"
		exit 1
	fi
}

FLANNEL_START="start-stop-daemon \
--start \
--background \
--quiet \
--exec $FLANNEL \
--make-pidfile --pidfile $FLANNEL_PIDFILE \
-- $FLANNEL_OPTS \
>> $FLANNEL_LOGFILE 2>&1"

FLANNEL_STOP="start-stop-daemon \
--stop \
--pidfile $FLANNEL_PIDFILE"

case "$1" in
	start)
		fail_unless_root
		log_begin_msg "Starting $FLANNEL_DESC: $BASE"
        $FLANNEL_START
		log_end_msg $?
		;;

	stop)
		fail_unless_root
		log_begin_msg "Stopping $FLANNEL_DESC: $BASE"
		$FLANNEL_STOP
		log_end_msg $?
		;;

	restart | force-reload)
		fail_unless_root
		log_begin_msg "Stopping $FLANNEL_DESC: $BASE"
		$FLANNEL_STOP
		$FLANNEL_START
		log_end_msg $?
		;;

	status)
		status_of_proc -p "$FLANNEL_DESC" "$FLANNEL" "$FLANNEL_DESC"
		;;

	*)
		echo "Usage: $0 {start|stop|restart|status}"
		exit 1
		;;
esac
