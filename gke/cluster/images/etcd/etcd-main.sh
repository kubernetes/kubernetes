#!/bin/bash

# Copyright 2020 The Kubernetes Authors.
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

set -o errexit
set -o nounset
set -x

# _sleep_sec - sleeps  without need to embed sleep into docker image.
function _sleep_sec {
  coproc read -t "${1}" && wait || true
}

# is_running [pid] - returns 0 (SUCCESS) if the process is still running
function _is_running {
  [[ $(jobs -p) =~ ${1} ]] > /dev/null
}

# run [cmd...] - runs given command, still interpreting SIGTERM signals during
# the execution.
function run {
  local command=("${@}")
  local repro=("${command[@]@Q}")

  echo "etcd-main.sh: % ${repro[*]}"
  "${@}" &
  WAITPID=$!

  while true; do
    # https://stackoverflow.com/questions/9629710/proper-way-to-detect-shell-exit-code-when-errexit-option-set
    wait ${WAITPID} && error_code=${?} || error_code=${?}
    echo "etcd-main.sh:  wait(pid:${WAITPID}) exited with ${error_code}"
    if _is_running ${WAITPID}; then
      continue
    else
      break
    fi
  done

  # No pid to terminate in case of SIGTERM signal.
  WAITPID=0

  if [ ${error_code} -ne 0 ]; then
    echo -e "etcd-main.sh: FAIL: (code:${error_code}):\n  % ${repro}"
  fi
  return ${error_code}
}

# When sigterm comes and we have active process, we forward it and
# after 20 seconds we follow up with SIGABRT
function _sigterm_handler {
  echo "etcd-main.sh: Got SIGTERM (active process: ${WAITPID})"
  if [ ${WAITPID} -gt 1 ]; then
    echo "etcd-main.sh: Propagating SIGTERM to the process to let it gracefully shutdown"
    kill -15 "${WAITPID}" || true

    echo "etcd-main.sh: Sleeping 20s to let the process terminate."
    # No 'seq' command:
    for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
      if ! _is_running ${WAITPID}; then
        echo "etcd-main.sh: Process '${WAITPID}' not running: _sigterm_handler is DONE"
        return 0
      fi
      _sleep_sec 1
    done
    echo "etcd-main.sh: SIGABRTing the child process."
    kill -6 "${WAITPID}" || true
  fi
  echo "etcd-main.sh: _sigterm_handler over"
}

function main {
  trap _sigterm_handler 15
  trap "echo 'etcd-main.sh: Got SIGQUIT'" 3
  trap "echo 'etcd-main.sh: Got SIGABRT'" 6
  trap "echo 'etcd-main.sh: Got exit'" exit

  echo
  echo '========================================================================='
  echo 'etcd-main.sh: Starting etcd script ...'

  DATA_DIRECTORY=${DATA_DIRECTORY:-/var/etcd/data/}

  echo "etcd-main.sh: environment:"
  declare -x

  echo 'etcd-main.sh: Backup before start ...'
  # Failure to perform a backup should not cause lack of ability to run etcd.
  run /usr/local/bin/backup || true

  echo 'etcd-main.sh: Migrate if needed ...'
  run /usr/local/bin/migrate

  echo 'etcd-main.sh: Running etcd ...'
  run /usr/local/bin/etcd "${@}"

  echo 'etcd-main.sh: Finished script'
}
