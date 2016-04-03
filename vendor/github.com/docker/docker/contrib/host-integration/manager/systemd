#!/bin/sh
set -e

cid="$1"
auth="$2"
desc="$3"

cat <<-EOF
	[Unit]
	Description=$desc
	Author=$auth
	After=docker.service

	[Service]
	ExecStart=/usr/bin/docker start -a $cid
	ExecStop=/usr/bin/docker stop -t 2 $cid

	[Install]
	WantedBy=local.target
EOF
