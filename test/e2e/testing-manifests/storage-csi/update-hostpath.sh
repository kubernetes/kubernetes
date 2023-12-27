#!/bin/sh

# Copyright 2021 The Kubernetes Authors.
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

# This script will update all sidecar RBAC files and the CSI hostpath
# deployment files such that they match what is in a hostpath driver
# release.
#
# Beware that this will wipe out all local modifications!

# Can be a tag or a branch.
script="$0"
hostpath_version="$1"

if ! [ "$hostpath_version" ]; then
    cat >&2 <<EOF
Usage: $0 <hostpath tag or branch name>

Required parameter is missing.
EOF
    exit 1
fi

set -xe
cd "$(dirname "$0")"

# Remove stale files.
rm -rf external-attacher external-provisioner external-resizer external-snapshotter external-health-monitor hostpath csi-driver-host-path

# Check out desired release.
git clone https://github.com/kubernetes-csi/csi-driver-host-path.git
(cd csi-driver-host-path && git checkout "$hostpath_version")
trap "rm -rf csi-driver-host-path" EXIT

# Main YAML files.
mkdir hostpath
cat >hostpath/README.md <<EOF
The files in this directory are exact copies of "kubernetes-latest" in
https://github.com/kubernetes-csi/csi-driver-host-path/tree/$hostpath_version/deploy/

Do not edit manually. Run $script to refresh the content.
EOF
cp -r csi-driver-host-path/deploy/kubernetes-latest/hostpath hostpath/
cat >hostpath/hostpath/e2e-test-rbac.yaml <<EOF
# privileged Pod Security Policy, previously defined just for gcePD via PrivilegedTestPSPClusterRoleBinding()
kind: ClusterRoleBinding
apiVersion: rbac.authorization.k8s.io/v1
metadata:
  name: psp-csi-hostpath-role
subjects:
  # This list of ServiceAccount intentionally covers everything that might
  # be needed. In practice, only some of these accounts are actually
  # used.
  - kind: ServiceAccount
    name: csi-attacher
    namespace: default
  - kind: ServiceAccount
    name: csi-provisioner
    namespace: default
  - kind: ServiceAccount
    name: csi-snapshotter
    namespace: default
  - kind: ServiceAccount
    name: csi-resizer
    namespace: default
  - kind: ServiceAccount
    name: csi-external-health-monitor-controller
    namespace: default
  - kind: ServiceAccount
    name: csi-hostpathplugin-sa
    namespace: default
roleRef:
  kind: ClusterRole
  name: e2e-test-privileged-psp
  apiGroup: rbac.authorization.k8s.io
EOF

download () {
    project="$1"
    path="$2"
    tag="$3"
    rbac="$4"

    mkdir -p "$project/$path"
    url="https://github.com/kubernetes-csi/$project/raw/$tag/deploy/kubernetes/$path/$rbac"
    cat >"$project/$path/$rbac" <<EOF
# Do not edit, downloaded from $url
# for csi-driver-host-path $hostpath_version
# by $script
#
EOF
    curl --fail --location "$url" >>"$project/$path/$rbac"
}

# RBAC files for each sidecar.
# This relies on the convention that "external-something" has "csi-something" as image name.
# external-health-monitor is special, it has two images.
# The repository for each image is ignored.
images=$(grep -r '^ *image:.*csi' hostpath/hostpath | sed -e 's;.*image:.*/;;' | grep -v 'node-driver-registrar' | sort -u)
for image in $images; do
    tag=$(echo "$image" | sed -e 's/.*://')
    path=
    rbac="rbac.yaml"
    case $image in
        csi-external-*)
            # csi-external-health-monitor-agent:v0.2.0
            project=$(echo "$image" | sed -e 's/csi-\(.*\)-[^:]*:.*/\1/')
            path=$(echo "$image" | sed -e 's/csi-\([^:]*\):.*/\1/')
            ;;
        *)
            project=$(echo "$image" | sed -e 's/:.*//' -e 's/^csi/external/')
            case $project in
                external-snapshotter)
                    # Another special case...
                    path="csi-snapshotter"
                    rbac="rbac-csi-snapshotter.yaml"
                    ;;
            esac
            ;;
    esac
    download "$project" "$path" "$tag" "$rbac"
done

# Update the mock driver manifests, too.
grep -r image: hostpath/hostpath/csi-hostpath-plugin.yaml | while read -r image; do
    version=$(echo "$image" | sed -e 's/.*:\(.*\)/\1/')
    image=$(echo "$image" | sed -e 's/.*image: \([^:]*\).*/\1/')
    sed -i '' -e "s;$image:.*;$image:$version;" mock/*.yaml
done
