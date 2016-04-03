#!/bin/bash

trap - EXIT # reset EXIT trap applied in .integration-daemon-start

for pidFile in $(find "$DEST" -name docker.pid); do
	pid=$(set -x; cat "$pidFile")
	( set -x; kill "$pid" )
	if ! wait "$pid"; then
		echo >&2 "warning: PID $pid from $pidFile had a nonzero exit code"
	fi
done

if [ -z "$DOCKER_TEST_HOST" ]; then
	# Stop apparmor if it is enabled
	if [ -e "/sys/module/apparmor/parameters/enabled" ] && [ "$(cat /sys/module/apparmor/parameters/enabled)" == "Y" ]; then
		(
			set -x
			/etc/init.d/apparmor stop
		)
	fi
fi
