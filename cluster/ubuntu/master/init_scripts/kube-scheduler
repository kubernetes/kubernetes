#!/bin/sh
set -e

### BEGIN INIT INFO
# Provides:           kube-scheduler
# Required-Start:     $etcd
# Required-Stop:      
# Should-Start:       
# Should-Stop:        
# Default-Start:      
# Default-Stop:       
# Short-Description:  Start kube-scheduler service
# Description:
#  http://www.github.com/GoogleCloudPlatform/Kubernetes
### END INIT INFO

export PATH=/sbin:/bin:/usr/sbin:/usr/bin:/usr/local/sbin:/usr/local/bin:/opt/bin:

BASE=$(basename $0)

# modify these in /etc/default/$BASE (/etc/default/kube-scheduler)
KUBE_SCHEDULER=/opt/bin/$BASE
# This is the pid file managed by kube-scheduler itself
KUBE_SCHEDULER_PIDFILE=/var/run/$BASE.pid
KUBE_SCHEDULER_LOGFILE=/var/log/$BASE.log
KUBE_SCHEDULER_OPTS=""
KUBE_SCHEDULER_DESC="Kube-Scheduler"

# Get lsb functions
. /lib/lsb/init-functions

if [ -f /etc/default/$BASE ]; then
	. /etc/default/$BASE
fi

# see also init_is_upstart in /lib/lsb/init-functions (which isn't available in Ubuntu 12.04, or we'd use it)
if [ -x /sbin/initctl ] && /sbin/initctl version 2>/dev/null | grep -q upstart; then
	log_failure_msg "$KUBE_SCHEDULER_DESC is managed via upstart, try using service $BASE $1"
	exit 1
fi

# Check kube-scheduler is present
if [ ! -x $KUBE_SCHEDULER ]; then
	log_failure_msg "$KUBE_SCHEDULER not present or not executable"
	exit 1
fi

fail_unless_root() {
	if [ "$(id -u)" != '0' ]; then
		log_failure_msg "$KUBE_SCHEDULER_DESC must be run as root"
		exit 1
	fi
}

KUBE_SCHEDULER_START="start-stop-daemon \
--start \
--background \
--quiet \
--exec $KUBE_SCHEDULER \
--make-pidfile --pidfile $KUBE_SCHEDULER_PIDFILE \
-- $KUBE_SCHEDULER_OPTS \
>> $KUBE_SCHEDULER_LOGFILE 2>&1"

KUBE_SCHEDULER_STOP="start-stop-daemon \
--stop \
--pidfile $KUBE_SCHEDULER_PIDFILE"

case "$1" in
	start)
		fail_unless_root
		log_begin_msg "Starting $KUBE_SCHEDULER_DESC: $BASE"
        $KUBE_SCHEDULER_START
		log_end_msg $?
		;;

	stop)
		fail_unless_root
		log_begin_msg "Stopping $KUBE_SCHEDULER_DESC: $BASE"
		$KUBE_SCHEDULER_STOP
		log_end_msg $?
		;;

	restart | force-reload)
		fail_unless_root
		log_begin_msg "Restarting $KUBE_SCHEDULER_DESC: $BASE"
		$KUBE_SCHEDULER_STOP
		$KUBE_SCHEDULER_START
		log_end_msg $?
		;;

	status)
		status_of_proc -p "$KUBE_SCHEDULER_PIDFILE" "$KUBE_SCHEDULER" "$KUBE_SCHEDULER_DESC"
		;;

	*)
		echo "Usage: $0 {start|stop|restart|status}"
		exit 1
		;;
esac
