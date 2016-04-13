#!/usr/bin/env bash
# Generate a minimal filesystem for CRUX/Linux and load it into the local
# docker as "cruxlinux"
# requires root and the crux iso (http://crux.nu)

set -e

die () {
    echo >&2 "$@"
    exit 1
}

[ "$#" -eq 1 ] || die "1 argument(s) required, $# provided. Usage: ./mkimage-crux.sh /path/to/iso"

ISO=${1}

ROOTFS=$(mktemp -d ${TMPDIR:-/var/tmp}/rootfs-crux-XXXXXXXXXX)
CRUX=$(mktemp -d ${TMPDIR:-/var/tmp}/crux-XXXXXXXXXX)
TMP=$(mktemp -d ${TMPDIR:-/var/tmp}/XXXXXXXXXX)

VERSION=$(basename --suffix=.iso $ISO | sed 's/[^0-9.]*\([0-9.]*\).*/\1/')

# Mount the ISO
mount -o ro,loop $ISO $CRUX

# Extract pkgutils
tar -C $TMP -xf $CRUX/tools/pkgutils#*.pkg.tar.gz

# Put pkgadd in the $PATH
export PATH="$TMP/usr/bin:$PATH"

# Install core packages
mkdir -p $ROOTFS/var/lib/pkg
touch $ROOTFS/var/lib/pkg/db
for pkg in $CRUX/crux/core/*; do
    pkgadd -r $ROOTFS $pkg
done

# Remove agetty and inittab config
if (grep agetty ${ROOTFS}/etc/inittab 2>&1 > /dev/null); then
    echo "Removing agetty from /etc/inittab ..."
    chroot ${ROOTFS} sed -i -e "/agetty/d" /etc/inittab
    chroot ${ROOTFS} sed -i -e "/shutdown/d" /etc/inittab
    chroot ${ROOTFS} sed -i -e "/^$/N;/^\n$/d" /etc/inittab
fi

# Remove kernel source
rm -rf $ROOTFS/usr/src/*

# udev doesn't work in containers, rebuild /dev
DEV=$ROOTFS/dev
rm -rf $DEV
mkdir -p $DEV
mknod -m 666 $DEV/null c 1 3
mknod -m 666 $DEV/zero c 1 5
mknod -m 666 $DEV/random c 1 8
mknod -m 666 $DEV/urandom c 1 9
mkdir -m 755 $DEV/pts
mkdir -m 1777 $DEV/shm
mknod -m 666 $DEV/tty c 5 0
mknod -m 600 $DEV/console c 5 1
mknod -m 666 $DEV/tty0 c 4 0
mknod -m 666 $DEV/full c 1 7
mknod -m 600 $DEV/initctl p
mknod -m 666 $DEV/ptmx c 5 2

IMAGE_ID=$(tar --numeric-owner -C $ROOTFS -c . | docker import - crux:$VERSION)
docker tag $IMAGE_ID crux:latest
docker run -i -t crux echo Success.

# Cleanup
umount $CRUX
rm -rf $ROOTFS
rm -rf $CRUX
rm -rf $TMP
