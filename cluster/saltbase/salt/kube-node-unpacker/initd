#!/bin/bash
#
### BEGIN INIT INFO
# Provides:   kube-node-unpacker
# Required-Start:    $local_fs $network $syslog docker
# Required-Stop:
# Default-Start:     2 3 4 5
# Default-Stop:      0 1 6
# Short-Description: Kubernetes Node Unpacker
# Description:
#   Unpacks docker images on Kubernetes nodes
### END INIT INFO


# PATH should only include /usr/* if it runs after the mountnfs.sh script
PATH=/sbin:/usr/sbin:/bin:/usr/bin
DESC="Kubernetes Node Unpacker"
NAME=kube-node-unpacker
DAEMON_LOG_FILE=/var/log/$NAME.log
PIDFILE=/var/run/$NAME.pid
SCRIPTNAME=/etc/init.d/$NAME
KUBE_MASTER_ADDONS_SH=/etc/kubernetes/kube-node-unpacker.sh

# Define LSB log_* functions.
# Depend on lsb-base (>= 3.2-14) to ensure that this file is present
# and status_of_proc is working.
. /lib/lsb/init-functions




#
# Function that starts the daemon/service
#
do_start()
{
    ${KUBE_MASTER_ADDONS_SH} </dev/null >>${DAEMON_LOG_FILE} 2>&1 &
    echo $! > ${PIDFILE}
    disown
}

#
# Function that stops the daemon/service
#
do_stop()
{
    kill $(cat ${PIDFILE})
    rm ${PIDFILE}
    return
}

case "$1" in
  start)
        log_daemon_msg "Starting $DESC" "$NAME"
        do_start
        case "$?" in
                0|1) log_end_msg 0 || exit 0 ;;
                2) log_end_msg 1 || exit 1 ;;
        esac
        ;;
  stop)
        log_daemon_msg "Stopping $DESC" "$NAME"
        do_stop
        case "$?" in
                0|1) log_end_msg 0 ;;
                2) exit 1 ;;
        esac
        ;;
  status)
        status_of_proc -p $PIDFILE $KUBE_MASTER_ADDONS_SH $NAME
        ;;

  restart|force-reload)
        log_daemon_msg "Restarting $DESC" "$NAME"
        do_stop
        case "$?" in
          0|1)
                do_start
                case "$?" in
                        0) log_end_msg 0 ;;
                        1) log_end_msg 1 ;; # Old process is still running
                        *) log_end_msg 1 ;; # Failed to start
                esac
                ;;
          *)
                # Failed to stop
                log_end_msg 1
                ;;
        esac
        ;;
  *)
        echo "Usage: $SCRIPTNAME {start|stop|status|restart|force-reload}" >&2
        exit 3
        ;;
esac
