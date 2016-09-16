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

# Helper script that installs a package, wrapping it with a policy that
# means we won't try to start services.
set -o errexit
set -o nounset
set -o pipefail

ACTION=${1}
NAME=${2}
VERSION=${3}
SRC=${4}

if [[ -z "${ACTION}" || -z "${NAME}" || -z "${VERSION}" || -z "${SRC}" ]]; then
  echo "Syntax: ${0} <action> <name> <version> <src>"
  exit 1
fi

old_policy=""

function install_no_start {
  # Query the existing installed version, assuming that an error means package not found
  existing=`dpkg-query -W -f='${Version}' ${NAME} 2>/dev/null || echo ""`
  if [[ -n "${existing}" ]]; then
    if [[ "${existing}" == "${VERSION}" ]]; then
      return
    fi
    echo "Different version of package ${NAME} installed: ${VERSION} vs ${existing}"
  fi

  if [[ -e "/usr/sbin/policy-rc.d" ]]; then
    tmpfile=`mktemp`
    mv /usr/sbin/policy-rc.d ${tmpfile}
    old_policy=${tmpfile}
  fi
  trap cleanup EXIT
  echo -e '#!/bin/sh\nexit 101' > /usr/sbin/policy-rc.d
  chmod 755 /usr/sbin/policy-rc.d

  echo "Installing package ${NAME} from ${SRC}"
  dpkg --install ${SRC}
}

function cleanup {
  rm -f /usr/sbin/policy-rc.d
  if [[ -n "${old_policy}" ]]; then
    mv ${old_policy} /usr/sbin/policy-rc.d
  fi
}

if [[ "${ACTION}" == "install-no-start" ]]; then
  install_no_start
else
  echo "Unknown action: ${ACTION}"
  exit 1
fi
