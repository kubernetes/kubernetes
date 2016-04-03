#!/usr/bin/env bash
set -e

mkimg="$(basename "$0")"

usage() {
	echo >&2 "usage: $mkimg [-d dir] [-t tag] script [script-args]"
	echo >&2 "   ie: $mkimg -t someuser/debian debootstrap --variant=minbase jessie"
	echo >&2 "       $mkimg -t someuser/ubuntu debootstrap --include=ubuntu-minimal --components=main,universe trusty"
	echo >&2 "       $mkimg -t someuser/busybox busybox-static"
	echo >&2 "       $mkimg -t someuser/centos:5 rinse --distribution centos-5"
	echo >&2 "       $mkimg -t someuser/mageia:4 mageia-urpmi --version=4"
	echo >&2 "       $mkimg -t someuser/mageia:4 mageia-urpmi --version=4 --mirror=http://somemirror/"
	exit 1
}

scriptDir="$(dirname "$(readlink -f "$BASH_SOURCE")")/mkimage"

optTemp=$(getopt --options '+d:t:h' --longoptions 'dir:,tag:,help' --name "$mkimg" -- "$@")
eval set -- "$optTemp"
unset optTemp

dir=
tag=
while true; do
	case "$1" in
		-d|--dir) dir="$2" ; shift 2 ;;
		-t|--tag) tag="$2" ; shift 2 ;;
		-h|--help) usage ;;
		--) shift ; break ;;
	esac
done

script="$1"
[ "$script" ] || usage
shift

if [ ! -x "$scriptDir/$script" ]; then
	echo >&2 "error: $script does not exist or is not executable"
	echo >&2 "  see $scriptDir for possible scripts"
	exit 1
fi

# don't mistake common scripts like .febootstrap-minimize as image-creators
if [[ "$script" == .* ]]; then
	echo >&2 "error: $script is a script helper, not a script"
	echo >&2 "  see $scriptDir for possible scripts"
	exit 1
fi

delDir=
if [ -z "$dir" ]; then
	dir="$(mktemp -d ${TMPDIR:-/var/tmp}/docker-mkimage.XXXXXXXXXX)"
	delDir=1
fi

rootfsDir="$dir/rootfs"
( set -x; mkdir -p "$rootfsDir" )

# pass all remaining arguments to $script
"$scriptDir/$script" "$rootfsDir" "$@"

# Docker mounts tmpfs at /dev and procfs at /proc so we can remove them
rm -rf "$rootfsDir/dev" "$rootfsDir/proc"
mkdir -p "$rootfsDir/dev" "$rootfsDir/proc"

# make sure /etc/resolv.conf has something useful in it
mkdir -p "$rootfsDir/etc"
cat > "$rootfsDir/etc/resolv.conf" <<'EOF'
nameserver 8.8.8.8
nameserver 8.8.4.4
EOF

tarFile="$dir/rootfs.tar.xz"
touch "$tarFile"

(
	set -x
	tar --numeric-owner -caf "$tarFile" -C "$rootfsDir" --transform='s,^./,,' .
)

echo >&2 "+ cat > '$dir/Dockerfile'"
cat > "$dir/Dockerfile" <<'EOF'
FROM scratch
ADD rootfs.tar.xz /
EOF

# if our generated image has a decent shell, let's set a default command
for shell in /bin/bash /usr/bin/fish /usr/bin/zsh /bin/sh; do
	if [ -x "$rootfsDir/$shell" ]; then
		( set -x; echo 'CMD ["'"$shell"'"]' >> "$dir/Dockerfile" )
		break
	fi
done

( set -x; rm -rf "$rootfsDir" )

if [ "$tag" ]; then
	( set -x; docker build -t "$tag" "$dir" )
elif [ "$delDir" ]; then
	# if we didn't specify a tag and we're going to delete our dir, let's just build an untagged image so that we did _something_
	( set -x; docker build "$dir" )
fi

if [ "$delDir" ]; then
	( set -x; rm -rf "$dir" )
fi
