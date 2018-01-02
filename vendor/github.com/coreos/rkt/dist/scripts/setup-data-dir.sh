#!/usr/bin/env bash

# Usage:
# ./setup-data-dir.sh [path to rkt data dir]
# The script can take one parameter being a path to the rkt data
# directory. If none is passed, /var/lib/rkt is assumed.

# Please keep it in sync with dist/init/systemd/tmpfiles.d/rkt.conf!

set -e

if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root"
   exit 1
fi

datadir="${1}"

if [[ -z "${datadir}" ]]; then
    datadir="/var/lib/rkt"
fi

# Creates the directory with the given mode and given group
# 1 - directory to create if it does not exist
# 2 - mode to set the directory to
# 3 - group to set the directory ownership to
make_directory() {
    local dir="${1}"
    local mode="${2}"
    local group="${3}"

    if [[ -e "${dir}" ]]; then
        chmod "${mode}" "${dir}"
    else
        mkdir --mode="${mode}" "${dir}"
    fi
    chgrp "${group}" "${dir}"
}

# Creates the file with the given mode and rkt group
# 1 - file to create if it does not exist
# 2 - mode to set the file to
create_rkt_file() {
    local file="${1}"
    local mode="${2}"

    if [[ ! -e "${file}" ]]; then
        touch "${file}"
    fi
    chmod "${mode}" "${file}"
    chgrp rkt "${file}"
}

getent group rkt-admin || groupadd --force --system rkt-admin
getent group rkt || groupadd --force --system rkt

if which systemd-tmpfiles; then
    systemd-tmpfiles --create "$(realpath "$(dirname "$0")")/../init/systemd/tmpfiles.d/rkt.conf" && exit
fi

make_directory "${datadir}" 2750 "rkt"
make_directory "${datadir}/tmp" 2750 "rkt"

make_directory "${datadir}/cas" 2770 "rkt"
make_directory "${datadir}/cas/db" 2770 "rkt"
create_rkt_file "${datadir}/cas/db/ql.db" 0660
# the ql database uses a WAL file whose name is generated from the sha1 hash of
# the database name
create_rkt_file "${datadir}/cas/db/.34a8b4c1ad933745146fdbfef3073706ee571625" 0660
make_directory "${datadir}/cas/imagelocks" 2770 "rkt"
make_directory "${datadir}/cas/imageManifest" 2770 "rkt"
make_directory "${datadir}/cas/blob" 2770 "rkt"
make_directory "${datadir}/cas/tmp" 2770 "rkt"
make_directory "${datadir}/cas/tree" 2700 "rkt"
make_directory "${datadir}/cas/treestorelocks" 2700 "rkt"
make_directory "${datadir}/locks" 2750 "rkt"

make_directory "${datadir}/pods" 2750 "rkt"
make_directory "${datadir}/pods/embryo" 2750 "rkt"
make_directory "${datadir}/pods/prepare" 2750 "rkt"
make_directory "${datadir}/pods/prepared" 2750 "rkt"
make_directory "${datadir}/pods/run" 2750 "rkt"
make_directory "${datadir}/pods/exited-garbage" 2750 "rkt"
make_directory "${datadir}/pods/garbage" 2750 "rkt"

make_directory "/etc/rkt" 2775 "rkt-admin"
