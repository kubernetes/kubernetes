#!/bin/sh
set -e

### BEGIN INIT INFO
# Provides:           docker
# Required-Start:     $syslog $remote_fs
# Required-Stop:      $syslog $remote_fs
# Should-Start:       cgroupfs-mount cgroup-lite
# Should-Stop:        cgroupfs-mount cgroup-lite
# Default-Start:      2 3 4 5
# Default-Stop:       0 1 6
# Short-Description:  Create lightweight, portable, self-sufficient containers.
# Description:
#  Docker is an open-source project to easily create lightweight, portable,
#  self-sufficient containers from any application. The same container that a
#  developer builds and tests on a laptop can run at scale, in production, on
#  VMs, bare metal, OpenStack clusters, public clouds and more.
### END INIT INFO

export PATH=/sbin:/bin:/usr/sbin:/usr/bin:/usr/local/sbin:/usr/local/bin

BASE=docker

# modify these in /etc/default/$BASE (/etc/default/docker)
DOCKERD=/usr/bin/dockerd
# This is the pid file managed by docker itself
DOCKER_PIDFILE=/var/run/$BASE.pid
# This is the pid file created/managed by start-stop-daemon
DOCKER_SSD_PIDFILE=/var/run/$BASE-ssd.pid
DOCKER_LOGFILE=/var/log/$BASE.log
DOCKER_OPTS=
DOCKER_DESC="Docker"

# Get lsb functions
. /lib/lsb/init-functions

if [ -f /etc/default/$BASE ]; then
	. /etc/default/$BASE
fi

# Check docker is present
if [ ! -x $DOCKERD ]; then
	log_failure_msg "$DOCKERD not present or not executable"
	exit 1
fi

check_init() {
	# see also init_is_upstart in /lib/lsb/init-functions (which isn't available in Ubuntu 12.04, or we'd use it directly)
	if [ -x /sbin/initctl ] && /sbin/initctl version 2>/dev/null | grep -q upstart; then
		log_failure_msg "$DOCKER_DESC is managed via upstart, try using service $BASE $1"
		exit 1
	fi
}

fail_unless_root() {
	if [ "$(id -u)" != '0' ]; then
		log_failure_msg "$DOCKER_DESC must be run as root"
		exit 1
	fi
}

cgroupfs_mount() {
	# see also https://github.com/tianon/cgroupfs-mount/blob/master/cgroupfs-mount
	if grep -v '^#' /etc/fstab | grep -q cgroup \
		|| [ ! -e /proc/cgroups ] \
		|| [ ! -d /sys/fs/cgroup ]; then
		return
	fi
	if ! mountpoint -q /sys/fs/cgroup; then
		mount -t tmpfs -o uid=0,gid=0,mode=0755 cgroup /sys/fs/cgroup
	fi
	(
		cd /sys/fs/cgroup
		for sys in $(awk '!/^#/ { if ($4 == 1) print $1 }' /proc/cgroups); do
			mkdir -p $sys
			if ! mountpoint -q $sys; then
				if ! mount -n -t cgroup -o $sys cgroup $sys; then
					rmdir $sys || true
				fi
			fi
		done
	)
}

case "$1" in
	start)
		check_init
		
		fail_unless_root

		cgroupfs_mount

		touch "$DOCKER_LOGFILE"
		chgrp docker "$DOCKER_LOGFILE"

		ulimit -n 1048576

		# Having non-zero limits causes performance problems due to accounting overhead
		# in the kernel. We recommend using cgroups to do container-local accounting.
		if [ "$BASH" ]; then
			ulimit -u unlimited
		else
			ulimit -p unlimited
		fi

		log_begin_msg "Starting $DOCKER_DESC: $BASE"
		start-stop-daemon --start --background \
			--no-close \
			--exec "$DOCKERD" \
			--pidfile "$DOCKER_SSD_PIDFILE" \
			--make-pidfile \
			-- \
				-p "$DOCKER_PIDFILE" \
				$DOCKER_OPTS \
					>> "$DOCKER_LOGFILE" 2>&1
		log_end_msg $?
		;;

	stop)
		check_init
		fail_unless_root
		if [ -f "$DOCKER_SSD_PIDFILE" ]; then
			log_begin_msg "Stopping $DOCKER_DESC: $BASE"
			start-stop-daemon --stop --pidfile "$DOCKER_SSD_PIDFILE" --retry 10
			log_end_msg $?
		else
			log_warning_msg "Docker already stopped - file $DOCKER_SSD_PIDFILE not found."
		fi
		;;

	restart)
		check_init
		fail_unless_root
		docker_pid=`cat "$DOCKER_SSD_PIDFILE" 2>/dev/null`
		[ -n "$docker_pid" ] \
			&& ps -p $docker_pid > /dev/null 2>&1 \
			&& $0 stop
		$0 start
		;;

	force-reload)
		check_init
		fail_unless_root
		$0 restart
		;;

	status)
		check_init
		status_of_proc -p "$DOCKER_SSD_PIDFILE" "$DOCKERD" "$DOCKER_DESC"
		;;

	*)
		echo "Usage: service docker {start|stop|restart|status}"
		exit 1
		;;
esac
