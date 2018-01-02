#!/bin/sh
#
# Generate a minimal filesystem for PLD Linux and load it into the local docker as "pld".
# https://www.pld-linux.org/packages/docker
#
set -e

if [ "$(id -u)" != "0" ]; then
	echo >&2 "$0: requires root"
	exit 1
fi

image_name=pld

tmpdir=$(mktemp -d ${TMPDIR:-/var/tmp}/pld-docker-XXXXXX)
root=$tmpdir/rootfs
install -d -m 755 $root

# to clean up:
docker rmi $image_name || :

# build
rpm -r $root --initdb

set +e
install -d $root/dev/pts
mknod $root/dev/random c 1 8 -m 644
mknod $root/dev/urandom c 1 9 -m 644
mknod $root/dev/full c 1 7 -m 666
mknod $root/dev/null c 1 3 -m 666
mknod $root/dev/zero c 1 5 -m 666
mknod $root/dev/console c 5 1 -m 660
set -e

poldek -r $root --up --noask -u \
	--noignore \
	-O 'rpmdef=_install_langs C' \
	-O 'rpmdef=_excludedocs 1' \
	vserver-packages \
	bash iproute2 coreutils grep poldek

# fix netsharedpath, so containers would be able to install when some paths are mounted
sed -i -e 's;^#%_netsharedpath.*;%_netsharedpath /dev/shm:/sys:/proc:/dev:/etc/hostname;' $root/etc/rpm/macros

# no need for alternatives
poldek-config -c $root/etc/poldek/poldek.conf ignore systemd-init

# this makes initscripts to believe network is up
touch $root/var/lock/subsys/network

# cleanup large optional packages
remove_packages="ca-certificates"
for pkg in $remove_packages; do
	rpm -r $root -q $pkg && rpm -r $root -e $pkg --nodeps
done

# cleanup more
rm -v $root/etc/ld.so.cache
rm -rfv $root/var/cache/hrmib/*
rm -rfv $root/usr/share/man/man?/*
rm -rfv $root/usr/share/locale/*/
rm -rfv $root/usr/share/help/*/
rm -rfv $root/usr/share/doc/*
rm -rfv $root/usr/src/examples/*
rm -rfv $root/usr/share/pixmaps/*

# and import
tar --numeric-owner --xattrs --acls -C $root -c . | docker import - $image_name

# and test
docker run -i -u root $image_name /bin/echo Success.

rm -r $tmpdir
