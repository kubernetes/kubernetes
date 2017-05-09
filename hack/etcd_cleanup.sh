#!/bin/bash

# Copyright 2016 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script is designed to run on master and assume etcd is running on localhost.
# It would recursively look into etcd keys and remove any empty directory
# It will download it's own version of etcdctl.
# Usage: (on master the mount point may not have exec flag, use bash to run the script)
# > bash etcd_cleanup.sh

set -o errexit
set -o nounset
set -o pipefail

TMPDIR=$(mktemp -d)
echo "using temp directory ${TMPDIR}"

# MLINE is output of mount (findmnt) for the mount-point TMPDIR is part of, it will be further processed to
# extract options (to see if noexec is in it) and mount point (to remount it if necessary)
MLINE=$( (MYDIR=$TMPDIR; until findmnt ${MYDIR} ; do MYDIR=$(dirname ${MYDIR}) ; done) | tail -n 1 | tr -s ' ')

FSTRING=$(echo ${MLINE} | cut -f 4 -d ' ')
MPOINT=$(echo ${MLINE} | cut -f 1 -d ' ')

if [[ ${FSTRING} == *"noexec"* ]]; then
  echo "non-executable mount point ${MPOINT} detected. remounting with exec flag..."
  sudo mount -o remount,exec ${MPOINT}
  TMP_REMOUNTED=1
else
  TMP_REMOUNTED=0
fi

function get_etcdctl {
	curl -s -L  https://github.com/coreos/etcd/releases/download/v2.3.4/etcd-v2.3.4-linux-amd64.tar.gz -o $1/etcd.tar.gz > /dev/null
	tar xzvf $1/etcd.tar.gz -C $1  > /dev/null
	ETCDCTL_PATH="$1/etcd-v2.3.4-linux-amd64/etcdctl"
}

function cleanup_empty_dirs {
  if [[ $(${ETCDCTL_PATH} ls $1) ]]; then
    for SUBDIR in $(${ETCDCTL_PATH} ls -p $1 | grep "/$")
    do
      cleanup_empty_dirs ${SUBDIR}
    done
  else
    echo "Removing empty key $1 ..."
    ${ETCDCTL_PATH} rmdir $1
  fi
}

get_etcdctl ${TMPDIR}
trap "rm -rf ${TMPDIR}" EXIT
cleanup_empty_dirs "/registry"

if [[ ${TMP_REMOUNTED} == 1 ]]
then
  echo "remounting ${MPOINT} with nonexec flag..."
  sudo mount -o remount,noexec ${MPOINT}
fi
