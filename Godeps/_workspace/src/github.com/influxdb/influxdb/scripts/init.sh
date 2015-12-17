#! /usr/bin/env bash

### BEGIN INIT INFO
# Provides:          influxd
# Required-Start:    $all
# Required-Stop:     $remote_fs $syslog
# Default-Start:     2 3 4 5
# Default-Stop:      0 1 6
# Short-Description: Start influxd at boot time
### END INIT INFO

# If you modify this, please make sure to also edit influxdb.service
# this init script supports three different variations:
#  1. New lsb that define start-stop-daemon
#  2. Old lsb that don't have start-stop-daemon but define, log, pidofproc and killproc
#  3. Centos installations without lsb-core installed
#
# In the third case we have to define our own functions which are very dumb
# and expect the args to be positioned correctly.

# Command-line options that can be set in /etc/default/influxdb.  These will override
# any config file values. Example: "-join http://1.2.3.4:8086"
DEFAULT=/etc/default/influxdb

# Daemon options
INFLUXD_OPTS=

# Process name ( For display )
NAME=influxdb

# User and group
USER=influxdb
GROUP=influxdb

# Daemon name, where is the actual executable
# If the daemon is not there, then exit.
DAEMON=/usr/bin/influxd
[ -x $DAEMON ] || exit 5

# Configuration file
CONFIG=/etc/influxdb/influxdb.conf

# PID file for the daemon
PIDFILE=/var/run/influxdb/influxd.pid
PIDDIR=`dirname $PIDFILE`
if [ ! -d "$PIDDIR" ]; then
    mkdir -p $PIDDIR
    chown $USER:$GROUP $PIDDIR
fi

# Max open files
OPEN_FILE_LIMIT=65536

if [ -r /lib/lsb/init-functions ]; then
    source /lib/lsb/init-functions
fi

# Logging
if [ -z "$STDOUT" ]; then
    STDOUT=/dev/null
fi

if [ ! -f "$STDOUT" ]; then
    mkdir -p $(dirname $STDOUT)
fi

if [ -z "$STDERR" ]; then
    STDERR=/var/log/influxdb/influxd.log
fi

if [ ! -f "$STDERR" ]; then
    mkdir -p $(dirname $STDERR)
fi

# Overwrite init script variables with /etc/default/influxdb values
if [ -r $DEFAULT ]; then
    source $DEFAULT
fi

function pidofproc() {
    if [ $# -ne 3 ]; then
        echo "Expected three arguments, e.g. $0 -p pidfile daemon-name"
    fi

    PID=`pgrep -f $3`
    local PIDFILE=`cat $2`

    if [ "x$PIDFILE" == "x" ]; then
        return 1
    fi

    if [ "x$PID" != "x" -a "$PIDFILE" == "$PID" ]; then
        return 0
    fi

    return 1
}

function killproc() {
    if [ $# -ne 3 ]; then
        echo "Expected three arguments, e.g. $0 -p pidfile signal"
    fi

    PID=`cat $2`

    /bin/kill -s $3 $PID
    while true; do
        pidof `basename $DAEMON` >/dev/null
        if [ $? -ne 0 ]; then
            return 0
        fi

        sleep 1
        n=$(expr $n + 1)
        if [ $n -eq 30 ]; then
            /bin/kill -s SIGKILL $PID
            return 0
        fi
    done
}

function log_failure_msg() {
    echo "$@" "[ FAILED ]"
}

function log_success_msg() {
    echo "$@" "[ OK ]"
}

case $1 in
    start)
        # Check if config file exist
        if [ ! -r $CONFIG ]; then
            log_failure_msg "config file doesn't exist (or you don't have permission to view)"
            exit 4
        fi

        # Checked the PID file exists and check the actual status of process
        if [ -e $PIDFILE ]; then
	    PID="$(pgrep -f $PIDFILE)"
	    if test ! -z $PID && kill -0 "$PID" &>/dev/null; then
		# If the status is SUCCESS then don't need to start again.
                log_failure_msg "$NAME process is running"
                exit 0 # Exit
            fi
        # if PID file does not exist, check if writable
        else
            su -s /bin/sh -c "touch $PIDFILE" $USER > /dev/null 2>&1
            if [ $? -ne 0 ]; then
                log_failure_msg "$PIDFILE not writable, check permissions"
                exit 5
            fi
        fi

        # Bump the file limits, before launching the daemon. These will carry over to
        # launched processes.
        ulimit -n $OPEN_FILE_LIMIT
        if [ $? -ne 0 ]; then
            log_failure_msg "set open file limit to $OPEN_FILE_LIMIT"
            exit 1
        fi

        log_success_msg "Starting the process" "$NAME"
        if which start-stop-daemon > /dev/null 2>&1; then
            start-stop-daemon --chuid $GROUP:$USER --start --quiet --pidfile $PIDFILE --exec $DAEMON -- -pidfile $PIDFILE -config $CONFIG $INFLUXD_OPTS >>$STDOUT 2>>$STDERR &
        else
            su -s /bin/sh -c "nohup $DAEMON -pidfile $PIDFILE -config $CONFIG $INFLUXD_OPTS >>$STDOUT 2>>$STDERR &" $USER
        fi
        log_success_msg "$NAME process was started"
        ;;

    stop)
        # Stop the daemon.
        if [ -e $PIDFILE ]; then
	    PID="$(pgrep -f $PIDFILE)"
	    if test ! -z $PID && kill -0 "$PID" &>/dev/null; then
                if killproc -p $PIDFILE SIGTERM && /bin/rm -rf $PIDFILE; then
                    log_success_msg "$NAME process was stopped"
                else
                    log_failure_msg "$NAME failed to stop service"
                fi
            fi
        else
            log_failure_msg "$NAME process is not running"
        fi
        ;;

    restart)
        # Restart the daemon.
        $0 stop && sleep 2 && $0 start
        ;;

    status)
        # Check the status of the process.
        if [ -e $PIDFILE ]; then
	    PID="$(pgrep -f $PIDFILE)"
	    if test ! -z $PID && test -d "/proc/$PID" &>/dev/null; then
                log_success_msg "$NAME Process is running"
                exit 0
            else
                log_failure_msg "$NAME Process is not running"
                exit 1
            fi
        else
            log_failure_msg "$NAME Process is not running"
            exit 3
        fi
        ;;

    version)
        $DAEMON version
        ;;

    *)
        # For invalid arguments, print the usage message.
        echo "Usage: $0 {start|stop|restart|status|version}"
        exit 2
        ;;
esac
