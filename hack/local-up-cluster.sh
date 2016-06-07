#!/bin/bash

# Copyright 2014 The Kubernetes Authors All rights reserved.
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

KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..

if [ "$(id -u)" != "0" ]; then
    echo "WARNING : This script MAY be run as root for docker socket / iptables functionality; if failures occur, retry as root." 2>&1
fi

cat <<EOF
WARNING : This script is deprecated. Now the "local" KUBERNETES_PROVIDER is implemented, so to set up local cluster, use the following commands:

  export KUBERNETES_PROVIDER=local
  cluster/kube-up.sh

And to shut down the cluster:

  cluster/kube-down.sh

EOF

function usage {
            echo "This script starts a local kube cluster. "
            echo "Example 1: hack/local-up-cluster.sh -o _output/dockerized/bin/linux/amd64/ (run from docker output)"
            echo "Example 2: hack/local-up-cluster.sh (build a local copy of the source)"
}

### Allow user to supply the source directory.
GO_OUT=""
while getopts "o:" OPTION
do
    case $OPTION in
        o)
            echo "skipping build"
            echo "using source $OPTARG"
            GO_OUT="$OPTARG"
            if [ $GO_OUT == "" ]; then
                echo "You provided an invalid value for the build output directory."
                exit
            fi
            ;;
        ?)
            usage
            exit
            ;;
    esac
done

export GO_OUT
export ENABLE_DAEMON=false
export KUBERNETES_PROVIDER=local

${KUBE_ROOT}/cluster/kube-up.sh
