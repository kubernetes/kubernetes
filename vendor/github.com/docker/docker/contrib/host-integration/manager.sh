#!/bin/sh
set -e

usage() {
	echo >&2 "usage: $0 [-a author] [-d description] container [manager]"
	echo >&2 "   ie: $0 -a 'John Smith' 4ec9612a37cd systemd"
	echo >&2 "   ie: $0 -d 'Super Cool System' 4ec9612a37cd # defaults to upstart"
	exit 1
}

auth='<none>'
desc='<none>'
have_auth=
have_desc=
while getopts a:d: opt; do
	case "$opt" in
		a)
			auth="$OPTARG"
			have_auth=1
			;;
		d)
			desc="$OPTARG"
			have_desc=1
			;;
	esac
done
shift $(($OPTIND - 1))

[ $# -ge 1 -a $# -le 2 ] || usage

cid="$1"
script="${2:-upstart}"
if [ ! -e "manager/$script" ]; then
	echo >&2 "Error: manager type '$script' is unknown (PRs always welcome!)."
	echo >&2 'The currently supported types are:'
	echo >&2 "  $(cd manager && echo *)"
	exit 1
fi

# TODO https://github.com/docker/docker/issues/734 (docker inspect formatting)
#if command -v docker > /dev/null 2>&1; then
#	image="$(docker inspect -f '{{.Image}}' "$cid")"
#	if [ "$image" ]; then
#		if [ -z "$have_auth" ]; then
#			auth="$(docker inspect -f '{{.Author}}' "$image")"
#		fi
#		if [ -z "$have_desc" ]; then
#			desc="$(docker inspect -f '{{.Comment}}' "$image")"
#		fi
#	fi
#fi

exec "manager/$script" "$cid" "$auth" "$desc"
