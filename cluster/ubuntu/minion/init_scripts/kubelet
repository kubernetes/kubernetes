#!/bin/sh
set -e

### BEGIN INIT INFO
# Provides:           kubelet
# Required-Start:     $flannel
# Required-Stop:      
# Should-Start:       
# Should-Stop:        
# Default-Start:      
# Default-Stop:       
# Short-Description:  Start kubelet service
# Description:
#  http://www.github.com/GoogleCloudPlatform/Kubernetes
### END INIT INFO

export PATH=/sbin:/bin:/usr/sbin:/usr/bin:/usr/local/sbin:/usr/local/bin:/opt/bin:

BASE=$(basename $0)

# modify these in /etc/default/$BASE (/etc/default/kube-apiserver)
KUBELET=/opt/bin/$BASE
# This is the pid file managed by kube-apiserver itself
KUBELET_PIDFILE=/var/run/$BASE.pid
KUBELET_LOGFILE=/var/log/$BASE.log
KUBELET_OPTS=""
KUBELET_DESC="Kubelet"

# Get lsb functions
. /lib/lsb/init-functions

if [ -f /etc/default/$BASE ]; then
	. /etc/default/$BASE
fi

# see also init_is_upstart in /lib/lsb/init-functions (which isn't available in Ubuntu 12.04, or we'd use it)
if [ -x /sbin/initctl ] && /sbin/initctl version 2>/dev/null | grep -q upstart; then
	log_failure_msg "$KUBELET_DESC is managed via upstart, try using service $BASE $1"
	exit 1
fi

# Check kube-apiserver is present
if [ ! -x $KUBELET ]; then
	log_failure_msg "$KUBELET not present or not executable"
	exit 1
fi

fail_unless_root() {
	if [ "$(id -u)" != '0' ]; then
		log_failure_msg "$KUBELET_DESC must be run as root"
		exit 1
	fi
}

KUBELET_START="start-stop-daemon \
--start \
--background \
--quiet \
--exec $KUBELET \
--make-pidfile --pidfile $KUBELET_PIDFILE \
-- $KUBELET_OPTS \
>> $KUBELET_LOGFILE 2>&1"

KUBELET_STOP="start-stop-daemon \
--stop \
--pidfile $KUBELET_PIDFILE"

case "$1" in
	start)
		fail_unless_root
		log_begin_msg "Starting $KUBELET_DESC: $BASE"
        $KUBELET_START
		log_end_msg $?
		;;

	stop)
		fail_unless_root
		log_begin_msg "Stopping $KUBELET_DESC: $BASE"
		$KUBELET_STOP
		log_end_msg $?
		;;

	restart | force-reload)
		fail_unless_root
		log_begin_msg "Stopping $KUBELET_DESC: $BASE"
		$KUBELET_STOP
		$KUBELET_START
		log_end_msg $?
		;;

	status)
		status_of_proc -p "$KUBELET_PIDFILE" "$KUBELET" "$KUBELET_DESC"
		;;

	*)
		echo "Usage: $0 {start|stop|restart|status}"
		exit 1
		;;
esac
