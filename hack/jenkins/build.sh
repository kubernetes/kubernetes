#!/bin/bash

# Copyright 2015 Google Inc. All rights reserved.
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

# kubernetes-build job: Triggered by github checkins on a 5 minute
# poll. We abort this job if it takes longer than 10m. (Typically this
# job takes about ~5m as of 0.8.0, but it's actually not completely
# hermetic right now due to things like the golang image. It can take
# ~8m if you force it to be totally hermetic.)

set -o errexit
set -o nounset
set -o pipefail
set -o xtrace

if [[ "${CIRCLECI:-}" == "true" ]]; then
    JOB_NAME="circleci-${CIRCLE_PROJECT_USERNAME}-${CIRCLE_PROJECT_REPONAME}"
    BUILD_NUMBER=${CIRCLE_BUILD_NUM}
    WORKSPACE=`pwd`
else
    # Jenkins?

    # !!! ALERT !!! Jenkins default $HOME is /var/lib/jenkins, which is
    # global across jobs. We change $HOME instead to ${WORKSPACE}, which
    # is an incoming variable Jenkins provides us for this job's scratch
    # space.
    export HOME=${WORKSPACE} # Nothing should want Jenkins $HOME
fi

export PATH=$PATH:/usr/local/go/bin
export KUBE_RELEASE_RUN_TESTS=n
export KUBE_SKIP_CONFIRMATIONS=y

# Clean stuff out. Assume the worst - the last build may have left the
# tree in an odd state. There's a Jenkins git clean plugin, but we
# have the docker images to worry about as well, so be really pedantic
# about cleaning.
rm -rf ~/.kube*
make clean
git clean -fdx
docker ps -aq | xargs -r docker rm
docker images -q | xargs -r docker rmi

# Build
go run ./hack/e2e.go -v --build

# Push to GCS
./build/push-ci-build.sh

sha256sum _output/release-tars/kubernetes*.tar.gz
