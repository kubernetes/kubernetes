#!/usr/bin/env bash
#
# Create a base CentOS Docker image.

# This script is useful on systems with rinse available (e.g.,
# building a CentOS image on Debian).  See contrib/mkimage-yum.sh for
# a way to build CentOS images on systems with yum installed.

set -e

echo >&2
echo >&2 'warning: this script is deprecated - see mkimage.sh and mkimage/rinse'
echo >&2

repo="$1"
distro="$2"
mirror="$3"

if [ ! "$repo" ] || [ ! "$distro" ]; then
	self="$(basename $0)"
	echo >&2 "usage: $self repo distro [mirror]"
	echo >&2
	echo >&2 "   ie: $self username/centos centos-5"
	echo >&2 "       $self username/centos centos-6"
	echo >&2
	echo >&2 "   ie: $self username/slc slc-5"
	echo >&2 "       $self username/slc slc-6"
	echo >&2
	echo >&2 "   ie: $self username/centos centos-5 http://vault.centos.org/5.8/os/x86_64/CentOS/"
	echo >&2 "       $self username/centos centos-6 http://vault.centos.org/6.3/os/x86_64/Packages/"
	echo >&2
	echo >&2 'See /etc/rinse for supported values of "distro" and for examples of'
	echo >&2 '  expected values of "mirror".'
	echo >&2
	echo >&2 'This script is tested to work with the original upstream version of rinse,'
	echo >&2 '  found at http://www.steve.org.uk/Software/rinse/ and also in Debian at'
	echo >&2 '  http://packages.debian.org/wheezy/rinse -- as always, YMMV.'
	echo >&2
	exit 1
fi

target="${TMPDIR:-/var/tmp}/docker-rootfs-rinse-$distro-$$-$RANDOM"

cd "$(dirname "$(readlink -f "$BASH_SOURCE")")"
returnTo="$(pwd -P)"

rinseArgs=( --arch amd64 --distribution "$distro" --directory "$target" )
if [ "$mirror" ]; then
	rinseArgs+=( --mirror "$mirror" )
fi

set -x

mkdir -p "$target"

sudo rinse "${rinseArgs[@]}"

cd "$target"

# rinse fails a little at setting up /dev, so we'll just wipe it out and create our own
sudo rm -rf dev
sudo mkdir -m 755 dev
(
	cd dev
	sudo ln -sf /proc/self/fd ./
	sudo mkdir -m 755 pts
	sudo mkdir -m 1777 shm
	sudo mknod -m 600 console c 5 1
	sudo mknod -m 600 initctl p
	sudo mknod -m 666 full c 1 7
	sudo mknod -m 666 null c 1 3
	sudo mknod -m 666 ptmx c 5 2
	sudo mknod -m 666 random c 1 8
	sudo mknod -m 666 tty c 5 0
	sudo mknod -m 666 tty0 c 4 0
	sudo mknod -m 666 urandom c 1 9
	sudo mknod -m 666 zero c 1 5
)

# effectively: febootstrap-minimize --keep-zoneinfo --keep-rpmdb --keep-services "$target"
#  locales
sudo rm -rf usr/{{lib,share}/locale,{lib,lib64}/gconv,bin/localedef,sbin/build-locale-archive}
#  docs and man pages
sudo rm -rf usr/share/{man,doc,info,gnome/help}
#  cracklib
sudo rm -rf usr/share/cracklib
#  i18n
sudo rm -rf usr/share/i18n
#  yum cache
sudo rm -rf var/cache/yum
sudo mkdir -p --mode=0755 var/cache/yum
#  sln
sudo rm -rf sbin/sln
#  ldconfig
#sudo rm -rf sbin/ldconfig
sudo rm -rf etc/ld.so.cache var/cache/ldconfig
sudo mkdir -p --mode=0755 var/cache/ldconfig

# allow networking init scripts inside the container to work without extra steps
echo 'NETWORKING=yes' | sudo tee etc/sysconfig/network > /dev/null

# to restore locales later:
#  yum reinstall glibc-common

version=
if [ -r etc/redhat-release ]; then
	version="$(sed -E 's/^[^0-9.]*([0-9.]+).*$/\1/' etc/redhat-release)"
elif [ -r etc/SuSE-release ]; then
	version="$(awk '/^VERSION/ { print $3 }' etc/SuSE-release)"
fi

if [ -z "$version" ]; then
	echo >&2 "warning: cannot autodetect OS version, using $distro as tag"
	sleep 20
	version="$distro"
fi

sudo tar --numeric-owner -c . | docker import - $repo:$version

docker run -i -t $repo:$version echo success

cd "$returnTo"
sudo rm -rf "$target"
