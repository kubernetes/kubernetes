#!/usr/bin/env bash
set -e

dir="$1"

if [ -z "$dir" ]; then
	{
		echo 'This script is for destroying old /var/lib/docker directories more safely than'
		echo '  "rm -rf", which can cause data loss or other serious issues.'
		echo
		echo "usage: $0 directory"
		echo "   ie: $0 /var/lib/docker"
	} >&2
	exit 1
fi

if [ "$(id -u)" != 0 ]; then
	echo >&2 "error: $0 must be run as root"
	exit 1
fi

if [ ! -d "$dir" ]; then
	echo >&2 "error: $dir is not a directory"
	exit 1
fi

dir="$(readlink -f "$dir")"

echo
echo "Nuking $dir ..."
echo '  (if this is wrong, press Ctrl+C NOW!)'
echo

( set -x; sleep 10 )
echo

dir_in_dir() {
	inner="$1"
	outer="$2"
	[ "${inner#$outer}" != "$inner" ]
}

# let's start by unmounting any submounts in $dir
#   (like -v /home:... for example - DON'T DELETE MY HOME DIRECTORY BRU!)
for mount in $(awk '{ print $5 }' /proc/self/mountinfo); do
	mount="$(readlink -f "$mount" || true)"
	if [ "$dir" != "$mount" ] && dir_in_dir "$mount" "$dir"; then
		( set -x; umount -f "$mount" )
	fi
done

# now, let's go destroy individual btrfs subvolumes, if any exist
if command -v btrfs > /dev/null 2>&1; then
	# Find btrfs subvolumes under $dir checking for inode 256
	# Source: http://stackoverflow.com/a/32865333
	for subvol in $(find "$dir" -type d -inum 256 | sort -r); do
		if [ "$dir" != "$subvol" ]; then
			( set -x; btrfs subvolume delete "$subvol" )
		fi
	done
fi

# finally, DESTROY ALL THINGS
( shopt -s dotglob; set -x; rm -rf "$dir"/* )
