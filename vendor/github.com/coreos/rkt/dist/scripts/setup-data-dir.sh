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

# Creates the directory with the given mode and rkt group
# 1 - directory to create if it does not exist
# 2 - mode to set the directory to
make_rkt_directory() {
    local dir="${1}"
    local mode="${2}"

    if [[ -e "${dir}" ]]; then
        chmod "${mode}" "${dir}"
    else
        mkdir --mode="${mode}" "${dir}"
    fi
    chgrp rkt "${dir}"
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

make_rkt_directory "${datadir}" 2750
make_rkt_directory "${datadir}/tmp" 2750

make_rkt_directory "${datadir}/cas" 2770
make_rkt_directory "${datadir}/cas/db" 2770
create_rkt_file "${datadir}/cas/db/ql.db" 0660
# the ql database uses a WAL file whose name is generated from the sha1 hash of
# the database name
create_rkt_file "${datadir}/cas/db/.34a8b4c1ad933745146fdbfef3073706ee571625" 0660
make_rkt_directory "${datadir}/cas/imagelocks" 2770
make_rkt_directory "${datadir}/cas/imageManifest" 2770
make_rkt_directory "${datadir}/cas/blob" 2770
make_rkt_directory "${datadir}/cas/tmp" 2770
make_rkt_directory "${datadir}/cas/tree" 2700
make_rkt_directory "${datadir}/cas/treestorelocks" 2700
make_rkt_directory "${datadir}/locks" 2750

make_rkt_directory "${datadir}/pods" 2750
make_rkt_directory "${datadir}/pods/embryo" 2750
make_rkt_directory "${datadir}/pods/prepare" 2750
make_rkt_directory "${datadir}/pods/prepared" 2750
make_rkt_directory "${datadir}/pods/run" 2750
make_rkt_directory "${datadir}/pods/exited-garbage" 2750
make_rkt_directory "${datadir}/pods/garbage" 2750
