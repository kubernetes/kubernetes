#!/usr/bin/env bash

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
    cd /proc/sys/fs/binfmt_misc || exit
    for file in $entries; do
        if [ -f "qemu-${file}" ]; then
            echo -1 > "qemu-${file}"
        fi
    done
    )
fi

exec $(dirname "${BASH_SOURCE[0]}")/qemu-binfmt-conf.sh --qemu-path="${QEMU_BIN_DIR}" "$@"
