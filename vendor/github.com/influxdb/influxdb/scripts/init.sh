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
INFLUXD_OPTS=

USER=influxdb
GROUP=influxdb

if [ -r /lib/lsb/init-functions ]; then
    source /lib/lsb/init-functions
fi

DEFAULT=/etc/default/influxdb

if [ -r $DEFAULT ]; then
    source $DEFAULT
fi

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


OPEN_FILE_LIMIT=65536

function pidofproc() {
    if [ $# -ne 3 ]; then
        echo "Expected three arguments, e.g. $0 -p pidfile daemon-name"
    fi

    pid=$(pgrep -f $3)
    local pidfile=$(cat $2)

    if [ "x$pidfile" == "x" ]; then
        return 1
    fi

    if [ "x$pid" != "x" -a "$pidfile" == "$pid" ]; then
        return 0
    fi

    return 1
}

function killproc() {
    if [ $# -ne 3 ]; then
        echo "Expected three arguments, e.g. $0 -p pidfile signal"
    fi

    pid=$(cat $2)

    kill -s $3 $pid
}

function log_failure_msg() {
    echo "$@" "[ FAILED ]"
}

function log_success_msg() {
    echo "$@" "[ OK ]"
}

# Process name ( For display )
name=influxd

# Daemon name, where is the actual executable
daemon=/opt/influxdb/influxd

# pid file for the daemon
pidfile=/var/run/influxdb/influxd.pid
piddir=$(dirname $pidfile)

if [ ! -d "$piddir" ]; then
    mkdir -p $piddir
    chown $GROUP:$USER $piddir
fi

# Configuration file
config=/etc/opt/influxdb/influxdb.conf

# If the daemon is not there, then exit.
[ -x $daemon ] || exit 5

function wait_for_startup() {
    control=1
    while [ $control -lt 5 ]
    do
        if [ ! -e $pidfile ]; then
            sleep 1
            control=$((control+1))
        else
            break
        fi
    done
}

function is_process_running() {
    # Checked the PID file exists and check the actual status of process
    if [ -e $pidfile ]; then
        pidofproc -p $pidfile $daemon > /dev/null 2>&1 && status="0" || status="$?"
        # If the status is SUCCESS then don't need to start again.
        if [ "x$status" = "x0" ]; then
            return 0
        else
            return 1
        fi
    else
        return 1
    fi
}

case $1 in
    start)
        is_process_running
        if [ $? -eq 0 ]; then
            log_success_msg "$name process is running"
            exit 0 # Exit
        fi

        # Bump the file limits, before launching the daemon. These will carry over to
        # launched processes.
        ulimit -n $OPEN_FILE_LIMIT
        if [ $? -ne 0 ]; then
            log_failure_msg "set open file limit to $OPEN_FILE_LIMIT"
            exit 1
        fi

        log_success_msg "Starting the process" "$name"
        if which start-stop-daemon > /dev/null 2>&1; then
            start-stop-daemon --chuid $GROUP:$USER --start --quiet --pidfile $pidfile --exec $daemon -- -pidfile $pidfile -config $config $INFLUXD_OPTS >>$STDOUT 2>>$STDERR &
        else
            su $USER -c "nohup $daemon -pidfile $pidfile -config $config $INFLUXD_OPTS >>$STDOUT 2>>$STDERR &"
        fi

        wait_for_startup && is_process_running
        if [ $? -ne 0 ]; then
            log_failure_msg "$name process failed to start"
            exit 1
        else
            log_success_msg "$name process was started"
            exit 0
        fi
        ;;

    stop)
        # Stop the daemon.
        is_process_running
        if [ $? -ne 0 ]; then
            log_success_msg "$name process is not running"
            exit 0 # Exit
        else
            if killproc -p $pidfile SIGTERM && /bin/rm -rf $pidfile; then
                log_success_msg "$name process was stopped"
                exit 0
            else
                log_failure_msg "$name failed to stop service"
                exit 1
            fi
        fi
        ;;

    restart)
        # Restart the daemon.
        $0 stop && sleep 2 && $0 start
        ;;

    status)
        # Check the status of the process.
        is_process_running
        if [ $? -eq 0 ]; then
            log_success_msg "$name Process is running"
            exit 0
        else
            log_failure_msg "$name Process is not running"
            exit 3
        fi
        ;;

    version)
        $daemon version
        ;;

    *)
        # For invalid arguments, print the usage message.
        echo "Usage: $0 {start|stop|restart|status|version}"
        exit 2
        ;;
esac
