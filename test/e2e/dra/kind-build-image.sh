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

# This scripts invokes `kind build image` so that the resulting
# image has a containerd with CDI support.
#
# Usage: kind-build-image.sh <tag of generated image>

set -ex
set -o pipefail

tag="$1"
containerd="containerd-1.6.0-830-g34d078e99" # from https://github.com/kind-ci/containerd-nightlies/releases

tmpdir="$(mktemp -d)"
cleanup() {
    rm -rf "$tmpdir"
}
trap cleanup EXIT

kind build node-image --image "$tag" "$(pwd)"
curl -L --silent https://github.com/kind-ci/containerd-nightlies/releases/download/$containerd/$containerd-linux-amd64.tar.gz | tar -C "$tmpdir" -vzxf -
curl -L --silent https://github.com/kind-ci/containerd-nightlies/releases/download/$containerd/runc.amd64 >"$tmpdir/runc"

cat >"$tmpdir/Dockerfile" <<EOF
FROM $tag

COPY bin/* /usr/local/bin/
RUN chmod a+rx /usr/local/bin/*
COPY runc /usr/local/sbin
RUN chmod a+rx /usr/local/sbin/runc

# Enable CDI as described in https://github.com/container-orchestrated-devices/container-device-interface#containerd-configuration
RUN sed -i -e '/\[plugins."io.containerd.grpc.v1.cri"\]/a \ \ enable_cdi = true' /etc/containerd/config.toml
EOF

docker build --tag "$tag" "$tmpdir"
