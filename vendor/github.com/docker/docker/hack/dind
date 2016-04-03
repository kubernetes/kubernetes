#!/bin/bash
set -e

# DinD: a wrapper script which allows docker to be run inside a docker container.
# Original version by Jerome Petazzoni <jerome@docker.com>
# See the blog post: https://blog.docker.com/2013/09/docker-can-now-run-within-docker/
#
# This script should be executed inside a docker container in privilieged mode
# ('docker run --privileged', introduced in docker 0.6).

# Usage: dind CMD [ARG...]

# apparmor sucks and Docker needs to know that it's in a container (c) @tianon
export container=docker

# First, make sure that cgroups are mounted correctly.
CGROUP=/cgroup

mkdir -p "$CGROUP"

if ! mountpoint -q "$CGROUP"; then
	mount -n -t tmpfs -o uid=0,gid=0,mode=0755 cgroup $CGROUP || {
		echo >&2 'Could not make a tmpfs mount. Did you use --privileged?'
		exit 1
	}
fi

if [ -d /sys/kernel/security ] && ! mountpoint -q /sys/kernel/security; then
	mount -t securityfs none /sys/kernel/security || {
		echo >&2 'Could not mount /sys/kernel/security.'
		echo >&2 'AppArmor detection and -privileged mode might break.'
	}
fi

# Mount the cgroup hierarchies exactly as they are in the parent system.
for HIER in $(cut -d: -f2 /proc/1/cgroup); do

	# The following sections address a bug which manifests itself
	# by a cryptic "lxc-start: no ns_cgroup option specified" when
	# trying to start containers within a container.
	# The bug seems to appear when the cgroup hierarchies are not
	# mounted on the exact same directories in the host, and in the
	# container.

	SUBSYSTEMS="${HIER%name=*}"

	# If cgroup hierarchy is named(mounted with "-o name=foo") we
	# need to mount it in $CGROUP/foo to create exect same
	# directoryes as on host. Else we need to mount it as is e.g.
	# "subsys1,subsys2" if it has two subsystems

	# Named, control-less cgroups are mounted with "-o name=foo"
	# (and appear as such under /proc/<pid>/cgroup) but are usually
	# mounted on a directory named "foo" (without the "name=" prefix).
	# Systemd and OpenRC (and possibly others) both create such a
	# cgroup. So just mount them on directory $CGROUP/foo.

	OHIER=$HIER
	HIER="${HIER#*name=}"

	mkdir -p "$CGROUP/$HIER"

	if ! mountpoint -q "$CGROUP/$HIER"; then
		mount -n -t cgroup -o "$OHIER" cgroup "$CGROUP/$HIER"
	fi

	# Likewise, on at least one system, it has been reported that
	# systemd would mount the CPU and CPU accounting controllers
	# (respectively "cpu" and "cpuacct") with "-o cpuacct,cpu"
	# but on a directory called "cpu,cpuacct" (note the inversion
	# in the order of the groups). This tries to work around it.

	if [ "$HIER" = 'cpuacct,cpu' ]; then
		ln -s "$HIER" "$CGROUP/cpu,cpuacct"
	fi

	# If hierarchy has multiple subsystems, in /proc/<pid>/cgroup
	# we will see ":subsys1,subsys2,subsys3,name=foo:" substring,
	# we need to mount it to "$CGROUP/foo" and if there were no
	# name to "$CGROUP/subsys1,subsys2,subsys3", so we must create
	# symlinks for docker daemon to find these subsystems:
	# ln -s $CGROUP/foo $CGROUP/subsys1
	# ln -s $CGROUP/subsys1,subsys2,subsys3 $CGROUP/subsys1

	if [ "$SUBSYSTEMS" != "${SUBSYSTEMS//,/ }" ]; then
		SUBSYSTEMS="${SUBSYSTEMS//,/ }"
		for SUBSYS in $SUBSYSTEMS
		do
			ln -s "$CGROUP/$HIER" "$CGROUP/$SUBSYS"
		done
	fi
done

# Note: as I write those lines, the LXC userland tools cannot setup
# a "sub-container" properly if the "devices" cgroup is not in its
# own hierarchy. Let's detect this and issue a warning.
if ! grep -q :devices: /proc/1/cgroup; then
	echo >&2 'WARNING: the "devices" cgroup should be in its own hierarchy.'
fi
if ! grep -qw devices /proc/1/cgroup; then
	echo >&2 'WARNING: it looks like the "devices" cgroup is not mounted.'
fi

# Mount /tmp
mount -t tmpfs none /tmp

if [ $# -gt 0 ]; then
	exec "$@"
fi

echo >&2 'ERROR: No command specified.'
echo >&2 'You probably want to run hack/make.sh, or maybe a shell?'
