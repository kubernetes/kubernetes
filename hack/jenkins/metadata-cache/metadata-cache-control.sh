#!/bin/bash
# Copyright 2016 The Kubernetes Authors All rights reserved.
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

# See README.md for system usage instructions.

set -o errexit
set -o nounset

BASENAME="$(basename "${0}")"
DIRNAME="$(dirname "${0}")"

fail() {
  echo "${@}"
  exit 1
}

usage() {
  fail "Usage: ${BASENAME} <update|on|off|start|stop|connect|remote_*>"
}

if [[ -z "${1:-}" ]]; then
  usage
fi

MGI='metadata.google.internal'
METADATA_SERVER="169.254.169.254"
TARGET="/etc/hosts"
IP_URL="http://${MGI}/computeMetadata/v1/instance/network-interfaces/0/ip"

remote_ip() {
  curl -s -f -H Metadata-Flavor:Google "${IP_URL}"
}

drop_metadata_server() {
  sed -i -e "/${MGI}/d" "${TARGET}" || fail "Could not drop metadata entries"
}

print_metadata_cache() {
  internal_ip="$(remote_ip)"

  if [[ ! "${internal_ip}" =~ 10(\.[0-9]{1,3}){3} ]]; then
    fail "Could not find local 10. address at ${IP_URL}: ${internal_ip}"
  fi
  echo
  echo "# Metadata server configuration"
  echo "# Route most requests to the cache at 10."
  echo "# However testOnGCE requires the real entry to exist."
  for i in {1..10}; do
    echo "${internal_ip} metadata.google.internal  # Metadata cache"
  done
  echo "${METADATA_SERVER} metadata.google.internal  # Real metadata server"
}

configure_metadata_server() {
  new_info="$(print_metadata_cache)"
  echo "${new_info}" >> "${TARGET}" || fail "Could not add metadata entries"
}

do_local() {
  case $1 in
    on)
      echo -n "Adding metadata cache to configuration at ${TARGET}: "
      drop_metadata_server
      configure_metadata_server
      echo "updated."
      ;;
    off)
      echo -n "Removing metadata cache from configuration at ${TARGET}: "
      drop_metadata_server
      echo "removed."
      ;;
    stop)
      echo -n "Stopping metadata-cache: "
      pids="$(ps ax | grep 'python metadata-cache.py' | grep -v grep | grep -o -E '^[ 0-9]+' || echo)"
      if [[ -z "${pids}" ]]; then
        echo 'Not running'
      elif [[ -n "${pids}" ]]; then
        echo "Killing ${pids}"
        kill ${pids} || fail "Could not kill ${pids}"
      fi
      echo "stopped"
      ;;
    start)
      echo -n "Starting metadata-cache session: "
      screen -d -m -S metadata-cache python metadata-cache.py
      echo "started"
      ps ax | grep metadata
      ps ax | grep screen
      ;;
    connect)
      echo "Connecting to metadata-cache session (press C-a a to detach, C-c to kill)..."
      sleep 1
      screen -r -S metadata-cache
      "${0}" test
      ;;
    bootstrap)
      echo "Installing package prerequisites:"
      apt-get install -y python-pip screen curl
      pip install flask requests
      ;;
    cat)
      cat /etc/hosts
      ;;
    test)
      echo "Ping metadata server:"
      ping -c 1 "${MGI}"
      echo "Download local ip from metadata server:"
      remote_ip || fail "Could not find internal ip address from metadata server."
      echo
      ;;
    update)
      "${0}" bootstrap
      "${0}" off
      "${0}" stop
      "${0}" start
      "${0}" on
      "${0}" test
      ;;
    *)
      usage
      ;;
  esac
}

do_remote() {
  cmd="${1}"
  instance="${2:-}"
  if [[ -z "${instance}" ]]; then
    cmd=""
  fi
  shift
  shift
  case "${cmd}" in
    remote_create)
      echo "Creating ${instance}"
      gcloud compute instances create "${instance}"
      ;;
    remote_delete)
      echo "Deleting ${instance}"
      gcloud compute instances delete "${instance}"
      ;;
    remote_logs)
      echo "Grabbing logs from ${instance}"
      gcloud compute instances get-serial-port-output "${instance}"
      ;;
    remote_copy)
      echo "Copy files to ${instance}"
      gcloud compute copy-files "${DIRNAME}"/* "${instance}:/home/${USER}/"
      ;;
    remote_ssh)
      echo "Running ${BASENAME} on ${instance}"
      gcloud compute ssh "${instance}" -t -- sudo "/home/${USER}/${BASENAME}" "${@}"
      ;;
    remote_update)
      "${0}" remote_copy "${instance}"
      "${0}" remote_ssh "${instance}" update
      ;;
    *)
      fail "Remote usage: ${BASENAME} remote_<create|update|copy|ssh|logs|delete> <instance> [args ...]"
      ;;
  esac
}

case "${1}" in
  remote_*)
    do_remote "$@"
    ;;
  *)
    do_local "$@"
    ;;
esac
