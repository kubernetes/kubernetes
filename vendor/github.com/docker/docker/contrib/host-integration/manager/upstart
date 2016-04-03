#!/bin/sh
set -e

cid="$1"
auth="$2"
desc="$3"

cat <<-EOF
	description "$(echo "$desc" | sed 's/"/\\"/g')"
	author "$(echo "$auth" | sed 's/"/\\"/g')"
	start on filesystem and started lxc-net and started docker
	stop on runlevel [!2345]
	respawn
	exec /usr/bin/docker start -a "$cid"
EOF
