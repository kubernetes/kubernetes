#!/bin/bash

# Copyright 2025 The Kubernetes Authors.
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

set -o errexit
set -o nounset
set -o pipefail

readonly REGISTRY_URL="registry.k8s.io"
readonly REGISTRY_DIR="/registry"

# This script prepares a directory with container images to be used as a fake registry,
# then creates a tarball of that directory inside the container.

# function to download an image manifest and its blobs to create a fake registry layout.
prepare_image() {
    local image_name="$1"
    local tag="$2"
    local internal_tag="$3"
    local image_dir="$REGISTRY_DIR/$image_name"

    echo "--- Preparing image: ${image_name}:${tag} as ${image_name}:${internal_tag} ---"

    mkdir -p "$image_dir/manifests"
    mkdir -p "$image_dir/blobs"

    echo "Downloading and filtering manifest list for $image_name:$tag..."
    local tmp_manifest_path="$image_dir/manifests/tmp_${internal_tag}"
    # download the manifest and pipe it to jq to filter out windows images
    crane manifest "$REGISTRY_URL/$image_name:$tag" | jq '.manifests |= map(select(.platform.os != "windows"))' > "$tmp_manifest_path"
    echo "Saved manifest list to $tmp_manifest_path"

    local manifest_digest
    manifest_digest="sha256:$(sha256sum < "$tmp_manifest_path" | awk '{print $1}')"
    mv "$tmp_manifest_path" "$image_dir/manifests/$manifest_digest"
    echo "Saved manifest list to $image_dir/manifests/$manifest_digest"

    # the file named after the tag now contains only the digest, acting as a redirect pointer
    echo "$manifest_digest" > "$image_dir/manifests/${internal_tag}"
    echo "Created tag file ${internal_tag} pointing to digest $manifest_digest"

    echo "Parsing manifest list and downloading individual manifests and blobs..."
    
    jq -r '.manifests[].digest' < "$image_dir/manifests/$manifest_digest" | while read -r individual_manifest_digest; do
      echo "  Downloading manifest $individual_manifest_digest..."
      local individual_manifest_path="$image_dir/manifests/$individual_manifest_digest"
      crane manifest "$REGISTRY_URL/$image_name@$individual_manifest_digest" > "$individual_manifest_path"
      echo "  Saved manifest to $individual_manifest_path"

      local config_digest
      config_digest=$(jq -r '.config.digest' < "$individual_manifest_path")
      echo "    Downloading config blob $config_digest..."
      crane blob "$REGISTRY_URL/$image_name@$config_digest" > "$image_dir/blobs/$config_digest"
      echo "    Saved config blob to $image_dir/blobs/$config_digest"

      jq -r '.layers[].digest' < "$individual_manifest_path" | while read -r layer_digest; do
        echo "    Downloading layer blob $layer_digest..."
        crane blob "$REGISTRY_URL/$image_name@$layer_digest" > "$image_dir/blobs/$layer_digest"
        echo "    Saved layer blob to $image_dir/blobs/$layer_digest"
      done
    done

    echo "--- Successfully prepared ${image_name}:${internal_tag} ---"
}

# create the registry directory
mkdir -p "$REGISTRY_DIR"

echo "--> Processing images.txt..."
while read -r image tag internal_tag; do
    # skip empty lines or comments
    [[ -z "$image" || "$image" == \#* ]] && continue
    prepare_image "$image" "$tag" "$internal_tag"
done < /images.txt

echo "--> Done"
