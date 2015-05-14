#!/bin/bash

XVFB=/usr/bin/Xvfb
DISPLAY_NUMBER=99
XVFBARGS=":$DISPLAY_NUMBER -ac -screen 0 1024x768x16"
PIDFILE=/tmp/xvfb_${DISPLAY_NUMBER}.pid

case "$1" in
  start)
    echo -n "Starting virtual X frame buffer: Xvfb"
    /sbin/start-stop-daemon --start --quiet --pidfile $PIDFILE --make-pidfile --background --exec $XVFB -- $XVFBARGS
    echo "."
    ;;

  stop)
    echo -n "Stopping virtual X frame buffer: Xvfb"
    /sbin/start-stop-daemon --stop --quiet --pidfile $PIDFILE
    echo "."
    ;;

  restart)
    $0 stop
    $0 start
    ;;
  *)

  echo "Usage: script/xvfb {start|stop|restart}"
  exit 1
esac

exit 0
