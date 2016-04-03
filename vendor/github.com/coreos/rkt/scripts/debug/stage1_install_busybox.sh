#!/bin/bash

function usage_and_exit {
    echo "Usage: stage1_install_busybox.sh UUID"
    exit 1
}

ARGC=$#

if [ ${ARGC} -lt 1 ]; then
    usage_and_exit
fi

while test $# -gt 0
do
    case "${1}" in
        --*)
            usage_and_exit
            ;;
        *) UUID="${1}"
            ;;
    esac
    shift
done

BUSYBOX=${BUSYBOX_BINARY:-$(which busybox 2> /dev/null)}

if [ ! -x "${BUSYBOX}" ]; then
    echo "error: busybox binary is not executable: Install it or set BUSYBOX_BINARY env variable"
    exit 1
fi

IS_STATIC=$(file ${BUSYBOX} | grep static)

if [ -z "${IS_STATIC}" ]; then
    echo "error: busybox binary is not statically linked"
    exit 1
fi

RKT_RUN_DIR="/var/lib/rkt/pods/run"
POD_DIR="${RKT_RUN_DIR}/${UUID}"

BUSYBOX_LINKS="ls cp cat mount vi awk chmod chown mv df ps rm tar top tr wc which ping"

NSPAWN_PID=$(ps aux | grep "[u]uid=$UUID" | awk '{print $2}')

sudo nsenter -m -t "${NSPAWN_PID}" cp ${BUSYBOX} ${POD_DIR}/stage1/rootfs/bin
sudo nsenter -m -t "${NSPAWN_PID}" chmod +x "${POD_DIR}/stage1/rootfs/bin/busybox"

for link in ${BUSYBOX_LINKS}; do
    sudo nsenter -m -t "${NSPAWN_PID}" ln -sf busybox "${POD_DIR}/stage1/rootfs/bin/${link}"
done

echo "Busybox installed. Use the following command to enter pod's stage1:"
SYSTEMD_PPID=$(sudo cat "${RKT_RUN_DIR}/${UUID}/ppid")
SYSTEMD_PID=$(sudo cat /proc/$SYSTEMD_PPID/task/$SYSTEMD_PPID/children)
echo sudo nsenter -m -u -i -p -t ${SYSTEMD_PID}
