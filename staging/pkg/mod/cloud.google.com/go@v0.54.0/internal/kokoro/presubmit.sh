#!/bin/bash
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License..

# TODO(deklerk): Add integration tests when it's secure to do so. b/64723143

# Fail on any error
set -eo pipefail

# Display commands being run
set -x

# cd to project dir on Kokoro instance
cd git/gocloud

go version

export GOCLOUD_HOME=$KOKORO_ARTIFACTS_DIR/google-cloud-go/
export PATH="$GOPATH/bin:$PATH"
export GO111MODULE=on
export GOPROXY=https://proxy.golang.org

# Move code into artifacts dir
mkdir -p $GOCLOUD_HOME
git clone . $GOCLOUD_HOME
cd $GOCLOUD_HOME

try3() { eval "$*" || eval "$*" || eval "$*"; }

# All packages, including +build tools, are fetched.
try3 go mod download
go install github.com/jstemmer/go-junit-report
./internal/kokoro/vet.sh
./internal/kokoro/check_incompat_changes.sh

exit_code=0
# Run tests and tee output to log file, to be pushed to GCS as artifact.
for i in `find . -name go.mod`; do
  pushd `dirname $i`;
    go test -race -v -timeout 15m -short ./... 2>&1 \
      | tee sponge_log.log
    # Takes the kokoro output log (raw stdout) and creates a machine-parseable xml
    # file (xUnit). Then it exits with whatever exit code the last command had.
    cat sponge_log.log \
      | go-junit-report -set-exit-code > sponge_log.xml
    exit_code=$(($exit_code + $?))
  popd;
done

exit $exit_code
