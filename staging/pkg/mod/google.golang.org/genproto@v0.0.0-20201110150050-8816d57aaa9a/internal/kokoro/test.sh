#!/bin/bash

# Fail on any error
set -eo pipefail

# Display commands being run
set -x

# cd to project dir on Kokoro instance
cd github/go-genproto

go version

# Set $GOPATH
export GOPATH="$HOME/go"
export GENPROTO_HOME=$GOPATH/src/google.golang.org/genproto
export PATH="$GOPATH/bin:$PATH"
export GO111MODULE=on
mkdir -p $GENPROTO_HOME


# Move code into $GOPATH and get dependencies
git clone . $GENPROTO_HOME
cd $GENPROTO_HOME

try3() { eval "$*" || eval "$*" || eval "$*"; }

# All packages, including +build tools, are fetched.
try3 go mod download
./internal/kokoro/vet.sh

go get github.com/jstemmer/go-junit-report

set +e

# Run tests and tee output to log file, to be pushed to GCS as artifact.
go test -race -v ./... 2>&1 | tee $KOKORO_ARTIFACTS_DIR/sponge_log.log

cat $KOKORO_ARTIFACTS_DIR/sponge_log.log | go-junit-report -set-exit-code > $KOKORO_ARTIFACTS_DIR/sponge_log.xml
exit_code=$?

# Send logs to the Build Cop Bot for continuous builds.
if [[ $KOKORO_BUILD_ARTIFACTS_SUBDIR = *"continuous"* ]]; then
  chmod +x $KOKORO_GFILE_DIR/linux_amd64/buildcop
  $KOKORO_GFILE_DIR/linux_amd64/buildcop \
    -logs_dir=$KOKORO_ARTIFACTS_DIR \
    -repo=googleapis/go-genproto
fi

exit $exit_code