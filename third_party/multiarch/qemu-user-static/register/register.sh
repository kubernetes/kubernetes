#!/usr/bin/env bash

if [ "$( uname -s )" = 'Darwin' ]; then
    {
        echo 'You are running on Darwin, which has no binfmt_misc support.'
        echo 'However Docker Desktop for Mac has (some of) those binfmt_misc rules already set up.'
        echo 'Make sure you have Docker Desktop for Mac 2.1.0.0+ installed.'
    } >&2
    exit 0
fi

if [ "$EUID" -ne 0 ]; then
    exec sudo "$0" "$@"
    exit $?  # just in case exec fails
fi

QEMU_BIN_DIR=${QEMU_BIN_DIR:-/usr/bin}

if [ ! -d /proc/sys/fs/binfmt_misc ]; then
    echo "No binfmt support in the kernel."
    echo "  Try: '/sbin/modprobe binfmt_misc' from the host"
    exit 1
fi


if [ ! -f /proc/sys/fs/binfmt_misc/register ]; then
    mount binfmt_misc -t binfmt_misc /proc/sys/fs/binfmt_misc
fi

entries="aarch64 aarch64_be alpha arm armeb hppa m68k microblaze microblazeel mips mips64 mips64el mipsel mipsn32 mipsn32el ppc ppc64 ppc64le riscv32 riscv64 s390x sh4 sh4eb sparc sparc32plus sparc64 xtensa xtensaeb"

if [ "${1}" = "--reset" ]; then
    shift
    (
    cd /proc/sys/fs/binfmt_misc
    for file in $entries; do
        if [ -f qemu-${file} ]; then
            echo -1 > qemu-${file}
        fi
    done
    )
fi

exec $(dirname "${BASH_SOURCE}")/qemu-binfmt-conf.sh --qemu-path="${QEMU_BIN_DIR}" $@
