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
The files in this directory are exact copys of "kubernetes-latest" in
https://github.com/kubernetes-csi/csi-driver-host-path/tree/$hostpath_version/deploy/

Do not edit manually. Run $script to refresh the content.
EOF
cp -r csi-driver-host-path/deploy/kubernetes-latest/hostpath hostpath/
cat >hostpath/hostpath/e2e-test-rbac.yaml <<EOF
# priviledged Pod Security Policy, previously defined just for gcePD via PrivilegedTestPSPClusterRoleBinding()
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

#########################################################################
# The following section is responsible for generating csi-manifest.go   #
# and updating all container image references to make use of the images #
# listed in csi-manifest.go. This is so that CSI test images can be     #
# mirrored like the rest of the test images                             #
#########################################################################
manifest="../../../utils/image/manifest.go"
csiManifest="../../../utils/image/csi-manifest.go"

set +x # disable shell tracing, so that printed messages can be seden

# A temp on-disk file for holding information about container images
# Format: {key}\t{registry variable}\t{image name}\t{image version}
lookupFile=lookup.tmp
> $lookupFile # clear the file if it exists
trap 'rm -f $lookupFile' EXIT


# Lookup the registry variable name (in manifest.go) for the given registry
lookupRegistry() {
    registry=$(echo "$1" | sed -e 's/\//\\\//') # escape backslash characters
    awk "\$0 ~ /\"$registry\"/{print substr(\$1,1,length(\$1)-1)}" $manifest
}

# Lookup the key used for the given container image from the lookup file
lookupImageKey() {
    name=$1
    awk "\$0 ~ /$name/{print \$1}" $lookupFile
}

# If the given file line contains a templated image reference extract the key used for the container image
extractKey() {
    echo "$1" | grep "{{.*Image}}" | sed -e 's/.*{{\.\([[:upper:]][[:lower:]]*\)Image}}.*/\1/'
}

# Given a key to a container image, copy the configuration information from csi-manifest into the lookup file
# This will mean that the given container image information will be carried over when csi-manifest is regenerated
addExistingToLookup() {
    key=$1
    grep "configs\[${key}\]" $csiManifest | sed -e 's;.*configs\[\(.*\)\] = Config{\(.*\), "\(.*\)", "\(.*\)"};\1\t\2\t\3\t\4;' >> $lookupFile
}

###########################################################
# Find the new image versions to use
grep -r image: hostpath/hostpath/csi-hostpath-plugin.yaml | while read -r image; do
    version=$(echo "$image" | sed -e 's/.*:\(.*\)/\1/')
    image=$(echo "$image" | sed -e 's/.*image: \([^:]*\).*/\1/')

    key="$(extractKey "$image")"
    if [ -z "${key}" ] ; then
        #echo "DEBUG: Could not find key in image $image, adding new entry to lookup file"
        registry=$(echo "$image" | sed -e 's/\(.*\)\/.*/\1/')
        name=$(echo "$image" | sed -e 's/.*\/\(.*\)/\1/')
        key="$(echo "${name:0:1}" | tr '[:lower:]' '[:upper:]')$(echo "${name:1}" | sed -e 's/-//g')"

        mapping="$(lookupRegistry "$registry")"
        if [ -z "$mapping" ] ; then
            #echo "Could not find mapping for registry $registry, not updating $image reference"
            continue
        fi

        printf "${key}\tlist.${mapping}\t${name}\t${version}\n" >> $lookupFile
    else
        #echo "DEBUG: found key $key in image, copying from csi-manifest.go"
        addExistingToLookup "$key"
    fi
done

###########################################################
# Check to make sure that all referenced images are in the lookup file
grep -r --include \*.yaml.in "{{.*Image}}" | while read -r key; do
    key="$(extractKey "$key")"

    mapping=$(awk "\$1 ~ /$key/{print \$3}" "$lookupFile")
    if [ -z "$mapping" ] ; then
        #echo "DEBUG: Didn't find existing mapping for key $key, copying from csi-manifest.go"
        addExistingToLookup "$key"
    fi
done

###########################################################
# Find any manifests with image references and change the file extension to .yaml.in
# Note: This will move any file with an image reference, even if it has an image
#       we are not going to template
grep -r --files-with-matches --include \*.yaml --exclude-dir=csi-driver-host-path --exclude-dir=gce-pd image: | while read -r filename; do
    # If the file is version controlled use git to move it, else do a plain move
    git mv "$filename" "${filename}.in" 2>/dev/null || mv "$filename" "${filename}.in"

    # Find all references to this file and update them to use the templated file
    (cd ../../ && grep -r --include \*.go "${filename}\"") | while read -r reference; do
        reference=$(echo "$reference" | sed -e 's/\([^:]\):.*/\1/')
	sed -i -e "s;${filename}\";${filename}.in\";" ../../"$reference"
    done
done

###########################################################
# Replace all image references with template references
grep -r --include \*.yaml.in image: | while read -r image; do
    file=$(echo "$image" | sed -e 's/\([^:]\):.*/\1/')
    image=$(echo "$image" | sed -e 's/.*image: \([^:]*\).*/\1/')
    name=$(echo "$image" | sed -e 's/.*\/\(.*\)/\1/')

    key="$(extractKey "$image")"
    if [ -n "$key" ] ; then
        #echo "DEBUG: already templated $name in $file"
        continue
    fi

    key="$(lookupImageKey "$name")"
    if [ -z "$key" ] ; then
        echo "Could not find a mapping for image $image, not templating it"
        continue
    fi

    sed -i -e "s;$image:.*;{{.${key}Image}};" "$file"
done

###########################################################
# Generate the new csi-manifest.go file
(
cat << EOF
/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/*
This file is generated and managed by test/e2e/testing-manifests/storage-csi/update-hostpath.sh
Do not edit
*/

package image

const (
    // Offset the CSI images so there is no collision
    CSINone = iota + 500
EOF
awk '{print "    "$1}' $lookupFile
cat << EOF
)

type TestCSIImagesStruct struct {
EOF
awk '{print "    "$1"Image string"}' $lookupFile
cat << EOF
}

var TestCSIImages TestCSIImagesStruct
func init() {
    TestCSIImages = TestCSIImagesStruct{
EOF
awk '{print "        GetE2EImage("$1"),"}' $lookupFile
cat << EOF
    }
}

func initCSIImageConfigs(list RegistryList, configs map[int]Config) {
EOF
# 1:key 2:registry 3:name 4:version
awk '{print "    configs["$1"] = Config{"$2", \""$3"\", \""$4"\"}"}' $lookupFile
cat << EOF
}
EOF
) > $csiManifest
gofmt -w $csiManifest
