#!/bin/bash

# Copyright 2014 Google Inc. All rights reserved.
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

# This command builds and runs a local kubernetes cluster.

if [ "$(which etcd)" == "" ]; then
	echo "etcd must be in your PATH"
	exit 1
fi

# Stop right away if the build fails
set -e

# Only build what we need
(
  source $(dirname $0)/config-go.sh
  cd "${KUBE_TARGET}"
  BINARIES="kubecfg localkube"
  for b in $BINARIES; do
    echo "+++ Building ${b}"
    go build "${KUBE_GO_PACKAGE}"/cmd/${b}
  done
)

echo "Starting etcd"

ETCD_DIR=$(mktemp -d -t kube-integration.XXXXXX)
trap "rm -rf ${ETCD_DIR}" EXIT

(etcd -name test -data-dir ${ETCD_DIR} > /tmp/etcd.log) &
ETCD_PID=$!

sleep 5

echo "Running localkube as root (so it can talk to docker's unix socket)"
sudo $(dirname $0)/../output/go/localkube $*

kill $ETCD_PID
