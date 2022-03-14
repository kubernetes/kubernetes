#!/usr/bin/env bash

# Copyright 2022 The Kubernetes Authors.
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

# This script generates go.work so that it includes all Go packages
# in this repo, with a few exceptions.

set -o errexit
set -o nounset
set -o pipefail

# Go tools really don't like it if you have a symlink in `pwd`.
cd "$(pwd -P)"

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

# This sets up the environment, like GOCACHE, which keeps the worktree cleaner.
kube::golang::new::setup_env
export GO111MODULE=on # TODO(thockin): remove this when it's the default

cd "${KUBE_ROOT}"

# First nuke the top-level workspace files.  We have to do this for a few
# steps.  They will be recreated later.
rm -f go.work go.work.sum

# Treat hack/tools as a distinct workspace, to prevent tooling from trying to
# merge deps between it and the top-level workspace.  This block has to go
# first because `go work` doesn't know how to deal with nested workspaces
# ("go: .../go.work already exists").
# https://github.com/golang/go/issues/53938
(
    cd hack/tools
    rm -f go.work go.work.sum
    go work init .
    X=$(cat go.work)
    cat > go.work << __EOF__
// This is a hack, but it prevents go from climbing further and trying to
// reconcile the various deps across the "real" modules and this one.

$X
__EOF__
    go mod download # generate go.work.sum
)

# Generate the top-level workspace.
go work init
go work edit -use .
git ls-files -z ':(glob)./staging/src/k8s.io/**/go.mod' \
    | xargs -0 -n1 dirname -z \
    | xargs -0 -n1 go work edit -use
go mod download # generate go.work.sum
