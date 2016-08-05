#!/bin/sh
set -e

### BEGIN INIT INFO
# Provides:           kube-controller-manager
# Required-Start:     $etcd
# Required-Stop:      
# Should-Start:       
# Should-Stop:        
# Default-Start:      
# Default-Stop:       
# Short-Description:  Start kube-controller-manager service
# Description:
#  http://www.github.com/GoogleCloudPlatform/Kubernetes
### END INIT INFO

export PATH=/sbin:/bin:/usr/sbin:/usr/bin:/usr/local/sbin:/usr/local/bin:/opt/bin:

BASE=$(basename $0)

# modify these in /etc/default/$BASE (/etc/default/kube-controller-manager)
KUBE_CONTROLLER_MANAGER=/opt/bin/$BASE
# This is the pid file managed by kube-controller-manager itself
KUBE_CONTROLLER_MANAGER_PIDFILE=/var/run/$BASE.pid
KUBE_CONTROLLER_MANAGER_LOGFILE=/var/log/$BASE.log
KUBE_CONTROLLER_MANAGER_OPTS=""
KUBE_CONTROLLER_MANAGER_DESC="Kube-Controller-Manager"

# Get lsb functions
. /lib/lsb/init-functions

if [ -f /etc/default/$BASE ]; then
	. /etc/default/$BASE
fi

# see also init_is_upstart in /lib/lsb/init-functions (which isn't available in Ubuntu 12.04, or we'd use it)
if [ -x /sbin/initctl ] && /sbin/initctl version 2>/dev/null | grep -q upstart; then
	log_failure_msg "$KUBE_CONTROLLER_MANAGER_DESC is managed via upstart, try using service $BASE $1"
	exit 1
fi

# Check kube-controller-manager is present
if [ ! -x $KUBE_CONTROLLER_MANAGER ]; then
	log_failure_msg "$KUBE_CONTROLLER_MANAGER not present or not executable"
	exit 1
fi

fail_unless_root() {
	if [ "$(id -u)" != '0' ]; then
		log_failure_msg "$KUBE_CONTROLLER_MANAGER_DESC must be run as root"
		exit 1
	fi
}

KUBE_CONTROLLER_MANAGER_START="start-stop-daemon 
--start --background \
--quiet \
--exec $KUBE_CONTROLLER_MANAGER \
--make-pidfile \
--pidfile $KUBE_CONTROLLER_MANAGER_PIDFILE \
-- $KUBE_CONTROLLER_MANAGER_OPTS \
>> $KUBE_CONTROLLER_MANAGER_LOGFILE 2>&1"

KUBE_CONTROLLER_MANAGER_STOP="start-stop-daemon \
--stop \
--pidfile $KUBE_CONTROLLER_MANAGER_PIDFILE"

case "$1" in
	start)
		fail_unless_root
		log_begin_msg "Starting $KUBE_CONTROLLER_MANAGER_DESC: $BASE"
        $KUBE_CONTROLLER_MANAGER_START
		log_end_msg $?
		;;

	stop)
		fail_unless_root
		log_begin_msg "Stopping $KUBE_CONTROLLER_MANAGER_DESC: $BASE"
        $KUBE_CONTROLLER_MANAGER_STOP
		log_end_msg $?
		;;

	restart | force-reload)
		fail_unless_root
        log_daemon_message "Restarting $KUBE_CONTROLLER_MANAGER" || true
        $KUBE_CONTROLLER_MANAGER_STOP
        $KUBE_CONTROLLER_MANAGER_START
		log_end_msg $?
		;;

	status)
		status_of_proc -p "$KUBE_CONTROLLER_MANAGER_PIDFILE" "$KUBE_CONTROLLER_MANAGER" "$KUBE_CONTROLLER_MANAGER_DESC"
		;;

	*)
		echo "Usage: $0 {start|stop|restart|status}"
		exit 1
		;;
esac
