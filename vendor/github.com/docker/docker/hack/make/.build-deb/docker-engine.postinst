#!/bin/sh
set -e

case "$1" in
	configure)
		if [ -z "$2" ]; then
			if ! getent group docker > /dev/null; then
				groupadd --system docker
			fi
		fi
		;;
	abort-*)
		# How'd we get here??
		exit 1
		;;
	*)
		;;
esac

#DEBHELPER#
