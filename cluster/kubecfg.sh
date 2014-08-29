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

source $(dirname $0)/kube-env.sh
source $(dirname $0)/$KUBERNETES_PROVIDER/util.sh

CLOUDCFG=$(dirname $0)/../_output/go/bin/kubecfg
if [ ! -x $CLOUDCFG ]; then
  echo "Could not find kubecfg binary. Run hack/build-go.sh to build it."
  exit 1
fi

# When we are using vagrant it has hard coded auth.  We repeat that here so that
# we don't clobber auth that might be used for a publicly facing cluster.
if [ "$KUBERNETES_PROVIDER" == "vagrant" ]; then
  cat >~/.kubernetes_vagrant_auth <<EOF
{
  "User": "vagrant",
  "Password": "vagrant"
}
EOF
  AUTH_CONFIG="-auth $HOME/.kubernetes_vagrant_auth"
fi

detect-master > /dev/null
if [ "$KUBE_MASTER_IP" != "" ] && [ "$KUBERNETES_MASTER" == "" ]; then
  export KUBERNETES_MASTER=https://${KUBE_MASTER_IP}
fi

$CLOUDCFG $AUTH_CONFIG "$@"
