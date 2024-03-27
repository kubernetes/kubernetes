#!/usr/bin/env bash

# Copyright 2016 The Kubernetes Authors.
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

# This file creates release artifacts (tar files, container images) that are
# ready to distribute to install or distribute to end users.

###############################################################################
# Most of the ::release:: namespace functions have been moved to
# github.com/kubernetes/release.  Have a look in that repo and specifically in
# lib/releaselib.sh for ::release::-related functionality.
###############################################################################

# This is where the final release artifacts are created locally
readonly RELEASE_STAGE="${LOCAL_OUTPUT_ROOT}/release-stage"
readonly RELEASE_TARS="${LOCAL_OUTPUT_ROOT}/release-tars"
readonly RELEASE_IMAGES="${LOCAL_OUTPUT_ROOT}/release-images"

KUBE_BUILD_CONFORMANCE=${KUBE_BUILD_CONFORMANCE:-n}
KUBE_BUILD_WINDOWS=${KUBE_BUILD_WINDOWS:-n}
KUBE_BUILD_PULL_LATEST_IMAGES=${KUBE_BUILD_PULL_LATEST_IMAGES:-y}

# ---------------------------------------------------------------------------
# Build final release artifacts
function kube::release::clean_cruft() {
  # Clean out cruft
  find "${RELEASE_STAGE}" -name '*~' -exec rm {} \;
  find "${RELEASE_STAGE}" -name '#*#' -exec rm {} \;
  find "${RELEASE_STAGE}" -name '.DS*' -exec rm {} \;
}

function kube::release::package_tarballs() {
  # Clean out any old releases
  rm -rf "${RELEASE_STAGE}" "${RELEASE_TARS}" "${RELEASE_IMAGES}"
  mkdir -p "${RELEASE_TARS}"
  kube::release::package_src_tarball &
  kube::release::package_client_tarballs &
  kube::release::package_kube_manifests_tarball &
  kube::util::wait-for-jobs || { kube::log::error "previous tarball phase failed"; return 1; }

  # _node and _server tarballs depend on _src tarball
  kube::release::package_node_tarballs &
  kube::release::package_server_tarballs &
  kube::util::wait-for-jobs || { kube::log::error "previous tarball phase failed"; return 1; }

  kube::release::package_final_tarball & # _final depends on some of the previous phases
  kube::release::package_test_tarballs & # _test doesn't depend on anything
  kube::util::wait-for-jobs || { kube::log::error "previous tarball phase failed"; return 1; }
}

# Package the source code we built, for compliance/licensing/audit/yadda.
function kube::release::package_src_tarball() {
  local -r src_tarball="${RELEASE_TARS}/kubernetes-src.tar.gz"
  kube::log::status "Building tarball: src"
  if [[ "${KUBE_GIT_TREE_STATE-}" = 'clean' ]]; then
    git archive -o "${src_tarball}" HEAD
  else
    find "${KUBE_ROOT}" -mindepth 1 -maxdepth 1 \
      ! \( \
        \( -path "${KUBE_ROOT}"/_\*       -o \
           -path "${KUBE_ROOT}"/.git\*    -o \
           -path "${KUBE_ROOT}"/.config\* -o \
           -path "${KUBE_ROOT}"/.gsutil\*    \
        \) -prune \
      \) -print0 \
    | "${TAR}" czf "${src_tarball}" --transform "s|${KUBE_ROOT#/*}|kubernetes|" --null -T -
  fi
}

# Package up all of the cross compiled clients. Over time this should grow into
# a full SDK
function kube::release::package_client_tarballs() {
  # Find all of the built client binaries
  local long_platforms=("${LOCAL_OUTPUT_BINPATH}"/*/*)
  if [[ -n ${KUBE_BUILD_PLATFORMS-} ]]; then
    read -ra long_platforms <<< "${KUBE_BUILD_PLATFORMS}"
  fi

  for platform_long in "${long_platforms[@]}"; do
    local platform
    local platform_tag
    platform=${platform_long##"${LOCAL_OUTPUT_BINPATH}"/} # Strip LOCAL_OUTPUT_BINPATH
    platform_tag=${platform/\//-} # Replace a "/" for a "-"
    kube::log::status "Starting tarball: client $platform_tag"

    (
      local release_stage="${RELEASE_STAGE}/client/${platform_tag}/kubernetes"
      rm -rf "${release_stage}"
      mkdir -p "${release_stage}/client/bin"

      local client_bins=("${KUBE_CLIENT_BINARIES[@]}")
      if [[ "${platform%/*}" = 'windows' ]]; then
        client_bins=("${KUBE_CLIENT_BINARIES_WIN[@]}")
      fi

      # This fancy expression will expand to prepend a path
      # (${LOCAL_OUTPUT_BINPATH}/${platform}/) to every item in the
      # client_bins array.
      cp "${client_bins[@]/#/${LOCAL_OUTPUT_BINPATH}/${platform}/}" \
        "${release_stage}/client/bin/"

      kube::release::clean_cruft

      local package_name="${RELEASE_TARS}/kubernetes-client-${platform_tag}.tar.gz"
      kube::release::create_tarball "${package_name}" "${release_stage}/.."
    ) &
  done

  kube::log::status "Waiting on tarballs"
  kube::util::wait-for-jobs || { kube::log::error "client tarball creation failed"; exit 1; }
}

# Package up all of the node binaries
function kube::release::package_node_tarballs() {
  local platform
  for platform in "${KUBE_NODE_PLATFORMS[@]}"; do
    local platform_tag
    local arch
    platform_tag=${platform/\//-} # Replace a "/" for a "-"
    arch=$(basename "${platform}")
    kube::log::status "Building tarball: node $platform_tag"

    local release_stage="${RELEASE_STAGE}/node/${platform_tag}/kubernetes"
    rm -rf "${release_stage}"
    mkdir -p "${release_stage}/node/bin"

    local node_bins=("${KUBE_NODE_BINARIES[@]}")
    if [[ "${platform%/*}" = 'windows' ]]; then
      node_bins=("${KUBE_NODE_BINARIES_WIN[@]}")
    fi
    # This fancy expression will expand to prepend a path
    # (${LOCAL_OUTPUT_BINPATH}/${platform}/) to every item in the
    # node_bins array.
    cp "${node_bins[@]/#/${LOCAL_OUTPUT_BINPATH}/${platform}/}" \
      "${release_stage}/node/bin/"

    # TODO: Docker images here
    # kube::release::create_docker_images_for_server "${release_stage}/server/bin" "${arch}" "${os}"

    # Include the client binaries here too as they are useful debugging tools.
    local client_bins=("${KUBE_CLIENT_BINARIES[@]}")
    if [[ "${platform%/*}" = 'windows' ]]; then
      client_bins=("${KUBE_CLIENT_BINARIES_WIN[@]}")
    fi
    # This fancy expression will expand to prepend a path
    # (${LOCAL_OUTPUT_BINPATH}/${platform}/) to every item in the
    # client_bins array.
    cp "${client_bins[@]/#/${LOCAL_OUTPUT_BINPATH}/${platform}/}" \
      "${release_stage}/node/bin/"

    cp -R "${KUBE_ROOT}/LICENSES" "${release_stage}/"

    cp "${RELEASE_TARS}/kubernetes-src.tar.gz" "${release_stage}/"

    kube::release::clean_cruft

    local package_name="${RELEASE_TARS}/kubernetes-node-${platform_tag}.tar.gz"
    kube::release::create_tarball "${package_name}" "${release_stage}/.."
  done
}

# Package up all of the server binaries in docker images
function kube::release::build_server_images() {
  kube::util::ensure-docker-buildx
  docker buildx create --name img-builder --use || true
  docker buildx inspect --bootstrap
  kube::util::trap_add 'docker buildx rm' EXIT

  platforms=("${KUBE_SERVER_PLATFORMS[@]}")
  if [[ "${KUBE_BUILD_WINDOWS}" =~ [yY] ]]; then
    platforms=("${KUBE_SERVER_PLATFORMS[@]}" "windows/amd64")
  fi

  # Clean out any old images
  rm -rf "${RELEASE_IMAGES}"
  local platform
  for platform in "${platforms[@]}"; do
    local platform_tag
    local os
    local arch
    platform_tag=${platform/\//-} # Replace a "/" for a "-"
    os=$(dirname "${platform}")
    arch=$(basename "${platform}")
    kube::log::status "Building images: $platform_tag"

    local release_stage
    release_stage="${RELEASE_STAGE}/server/${platform_tag}/kubernetes"
    rm -rf "${release_stage}"
    mkdir -p "${release_stage}/server/bin"

    # This fancy expression will expand to prepend a path
    # (${LOCAL_OUTPUT_BINPATH}/${platform}/) to every item in the
    # KUBE_SERVER_LINUX_IMAGE_BINARIES / KUBE_SERVER_WINDOWS_IMAGE_BINARIES array.
    if [[ "${os}" = "windows" ]]; then
      cp "${KUBE_SERVER_WINDOWS_IMAGE_BINARIES[@]/#/${LOCAL_OUTPUT_BINPATH}/${platform}/}" \
        "${release_stage}/server/bin/"
    else
      cp "${KUBE_SERVER_LINUX_IMAGE_BINARIES[@]/#/${LOCAL_OUTPUT_BINPATH}/${platform}/}" \
        "${release_stage}/server/bin/"
    fi

    kube::release::create_docker_images_for_server "${release_stage}/server/bin" "${arch}" "${os}"
  done
}

# Package up all of the server binaries
function kube::release::package_server_tarballs() {
  kube::release::build_server_images
  local platform
  for platform in "${KUBE_SERVER_PLATFORMS[@]}"; do
    local platform_tag
    local arch
    platform_tag=${platform/\//-} # Replace a "/" for a "-"
    arch=$(basename "${platform}")
    kube::log::status "Building tarball: server $platform_tag"

    # NOTE: this directory was setup in kube::release::build_server_images
    local release_stage
    release_stage="${RELEASE_STAGE}/server/${platform_tag}/kubernetes"
    mkdir -p "${release_stage}/addons"

    # This fancy expression will expand to prepend a path
    # (${LOCAL_OUTPUT_BINPATH}/${platform}/) to every item in the
    # KUBE_SERVER_BINARIES array.
    cp "${KUBE_SERVER_BINARIES[@]/#/${LOCAL_OUTPUT_BINPATH}/${platform}/}" \
      "${release_stage}/server/bin/"

    # Include the client binaries here too as they are useful debugging tools.
    local client_bins
    client_bins=("${KUBE_CLIENT_BINARIES[@]}")
    if [[ "${platform%/*}" = 'windows' ]]; then
      client_bins=("${KUBE_CLIENT_BINARIES_WIN[@]}")
    fi
    # This fancy expression will expand to prepend a path
    # (${LOCAL_OUTPUT_BINPATH}/${platform}/) to every item in the
    # client_bins array.
    cp "${client_bins[@]/#/${LOCAL_OUTPUT_BINPATH}/${platform}/}" \
      "${release_stage}/server/bin/"

    cp -R "${KUBE_ROOT}/LICENSES" "${release_stage}/"

    cp "${RELEASE_TARS}/kubernetes-src.tar.gz" "${release_stage}/"

    kube::release::clean_cruft

    local package_name
    package_name="${RELEASE_TARS}/kubernetes-server-${platform_tag}.tar.gz"
    kube::release::create_tarball "${package_name}" "${release_stage}/.."
  done
}

function kube::release::md5() {
  if which md5 >/dev/null 2>&1; then
    md5 -q "$1"
  else
    md5sum "$1" | awk '{ print $1 }'
  fi
}

function kube::release::sha1() {
  if which sha1sum >/dev/null 2>&1; then
    sha1sum "$1" | awk '{ print $1 }'
  else
    shasum -a1 "$1" | awk '{ print $1 }'
  fi
}

function kube::release::build_conformance_image() {
  local -r arch="$1"
  local -r registry="$2"
  local -r version="$3"
  local -r save_dir="${4-}"
  kube::log::status "Building conformance image for arch: ${arch}"
  ARCH="${arch}" REGISTRY="${registry}" VERSION="${version}" \
    make -C test/conformance/image/ build >/dev/null

  local conformance_tag
  conformance_tag="${registry}/conformance-${arch}:${version}"
  if [[ -n "${save_dir}" ]]; then
    "${DOCKER[@]}" save "${conformance_tag}" > "${save_dir}/conformance-${arch}.tar"
  fi
  kube::log::status "Deleting conformance image ${conformance_tag}"
  "${DOCKER[@]}" rmi "${conformance_tag}" &>/dev/null || true
}

# This builds all the release docker images (One docker image per binary)
# Args:
#  $1 - binary_dir, the directory to save the tared images to.
#  $2 - arch, architecture for which we are building docker images.
#  $3 - os, the OS for which we are building docker images.
function kube::release::create_docker_images_for_server() {
  # Create a sub-shell so that we don't pollute the outer environment
  (
    local binary_dir
    local os
    local arch
    local binaries
    local images_dir
    binary_dir="$1"
    arch="$2"
    os="${3:-linux}"
    binaries=$(kube::build::get_docker_wrapped_binaries "${os}")
    images_dir="${RELEASE_IMAGES}/${arch}"
    mkdir -p "${images_dir}"

    # registry.k8s.io is the constant tag in the docker archives, this is also the default for config scripts in GKE.
    # We can use KUBE_DOCKER_REGISTRY to include and extra registry in the docker archive.
    # If we use KUBE_DOCKER_REGISTRY="registry.k8s.io", then the extra tag (same) is ignored, see release_docker_image_tag below.
    local -r docker_registry="registry.k8s.io"
    # Docker tags cannot contain '+'
    local docker_tag="${KUBE_GIT_VERSION/+/_}"
    if [[ -z "${docker_tag}" ]]; then
      kube::log::error "git version information missing; cannot create Docker tag"
      return 1
    fi

    # provide `--pull` argument to `docker build` if `KUBE_BUILD_PULL_LATEST_IMAGES`
    # is set to y or Y; otherwise try to build the image without forcefully
    # pulling the latest base image.
    local docker_build_opts
    docker_build_opts=
    if [[ "${KUBE_BUILD_PULL_LATEST_IMAGES}" =~ [yY] ]]; then
        docker_build_opts='--pull'
    fi

    for wrappable in $binaries; do

      # The full binary name might have a file extension (.exe).
      local full_binary_name=${wrappable%%,*}
      local binary_name=${full_binary_name%%.*}
      local base_image=${wrappable##*,}
      local binary_file_path="${binary_dir}/${full_binary_name}"
      local tar_path="${binary_dir}/${full_binary_name}.tar"
      local docker_build_path="${binary_file_path}.dockerbuild"
      local docker_image_tag="${docker_registry}/${binary_name}-${arch}:${docker_tag}"

      local docker_file_path="${KUBE_ROOT}/build/server-image/Dockerfile"
      # If this binary has its own Dockerfile use that else use the generic Dockerfile.
      if [[ "${os}" = "windows" && -f "${KUBE_ROOT}/build/server-image/${binary_name}/Dockerfile_windows" ]]; then
          docker_file_path="${KUBE_ROOT}/build/server-image/${binary_name}/Dockerfile_windows"
      elif [[ -f "${KUBE_ROOT}/build/server-image/${binary_name}/Dockerfile" ]]; then
          docker_file_path="${KUBE_ROOT}/build/server-image/${binary_name}/Dockerfile"
      fi

      kube::log::status "Starting docker build for image: ${binary_name}-${os}-${arch}"
      (
        rm -rf "${docker_build_path}"
        mkdir -p "${docker_build_path}"
        ln "${binary_file_path}" "${docker_build_path}/${full_binary_name}"

        # If we are building an official/alpha/beta release we want to keep
        # docker images and tag them appropriately.
        local additional_tag=()
        local release_docker_image_tag="${KUBE_DOCKER_REGISTRY-$docker_registry}/${binary_name}-${arch}:${KUBE_DOCKER_IMAGE_TAG-$docker_tag}"

	# append the OS name only for Windows images into the image name.
        if [[ "${os}" = "windows" ]]; then
          docker_image_tag="${docker_registry}/${binary_name}-${os}-${arch}:${docker_tag}"
          release_docker_image_tag="${KUBE_DOCKER_REGISTRY-$docker_registry}/${binary_name}-${os}-${arch}:${KUBE_DOCKER_IMAGE_TAG-$docker_tag}"
	fi

        if [[ "${release_docker_image_tag}" != "${docker_image_tag}" ]]; then
          kube::log::status "Adding additional tag ${release_docker_image_tag} for docker image ${docker_image_tag}"
          "${DOCKER[@]}" rmi "${release_docker_image_tag}" 2>/dev/null || true
          additional_tag=("-t" "${release_docker_image_tag}")
        fi

        local build_log="${docker_build_path}/build.log"
        if ! DOCKER_CLI_EXPERIMENTAL=enabled "${DOCKER[@]}" buildx build \
          -f "${docker_file_path}" \
          --platform "${os}/${arch}" \
          --output type=docker,dest="${tar_path}" \
          ${docker_build_opts:+"${docker_build_opts}"} \
          -t "${docker_image_tag}" "${additional_tag[@]}" \
          --build-arg BASEIMAGE="${base_image}" \
          --build-arg SETCAP_IMAGE="${KUBE_BUILD_SETCAP_IMAGE}" \
          --build-arg BINARY="${full_binary_name}" \
          "${docker_build_path}" >"${build_log}" 2>&1; then
            cat "${build_log}"
            exit 1
        fi
        rm "${build_log}"

        echo "${docker_tag}" > "${binary_file_path}.docker_tag"
        rm -rf "${docker_build_path}"
        ln "${tar_path}" "${images_dir}/"

        kube::log::status "Deleting docker image ${docker_image_tag}"
        "${DOCKER[@]}" rmi "${docker_image_tag}" &>/dev/null || true
      ) &
    done

    if [[ "${KUBE_BUILD_CONFORMANCE}" =~ [yY] ]]; then
      kube::release::build_conformance_image "${arch}" "${docker_registry}" \
        "${docker_tag}" "${images_dir}" &
    fi

    kube::util::wait-for-jobs || { kube::log::error "previous Docker build failed"; return 1; }
    kube::log::status "Docker builds done"
  )

}

# This will pack kube-system manifests files for distros such as COS.
function kube::release::package_kube_manifests_tarball() {
  kube::log::status "Building tarball: manifests"

  local src_dir="${KUBE_ROOT}/cluster/gce/manifests"

  local release_stage="${RELEASE_STAGE}/manifests/kubernetes"
  rm -rf "${release_stage}"

  local dst_dir="${release_stage}/gci-trusty"
  mkdir -p "${dst_dir}"
  cp "${src_dir}/kube-proxy.manifest" "${dst_dir}/"
  cp "${src_dir}/cluster-autoscaler.manifest" "${dst_dir}/"
  cp "${src_dir}/etcd.manifest" "${dst_dir}"
  cp "${src_dir}/kube-scheduler.manifest" "${dst_dir}"
  cp "${src_dir}/kube-apiserver.manifest" "${dst_dir}"
  cp "${src_dir}/konnectivity-server.yaml" "${dst_dir}"
  cp "${src_dir}/abac-authz-policy.jsonl" "${dst_dir}"
  cp "${src_dir}/cloud-controller-manager.manifest" "${dst_dir}"
  cp "${src_dir}/kube-controller-manager.manifest" "${dst_dir}"
  cp "${src_dir}/kube-addon-manager.yaml" "${dst_dir}"
  cp "${src_dir}/glbc.manifest" "${dst_dir}"
  find "${src_dir}" -name 'internal-*' -exec cp {} "${dst_dir}" \;
  cp "${KUBE_ROOT}/cluster/gce/gci/configure-helper.sh" "${dst_dir}/gci-configure-helper.sh"
  cp "${KUBE_ROOT}/cluster/gce/gci/configure-kubeapiserver.sh" "${dst_dir}/configure-kubeapiserver.sh"
  if [[ -e "${KUBE_ROOT}/cluster/gce/gci/gke-internal-configure-helper.sh" ]]; then
    cp "${KUBE_ROOT}/cluster/gce/gci/gke-internal-configure-helper.sh" "${dst_dir}/"
  fi
  cp "${KUBE_ROOT}/cluster/gce/gci/health-monitor.sh" "${dst_dir}/health-monitor.sh"
  # Merge GCE-specific addons with general purpose addons.
  for d in cluster/addons cluster/gce/addons; do
    find "${KUBE_ROOT}/${d}" \( \( -name \*.yaml -o -name \*.yaml.in -o -name \*.json \) -a ! \( -name \*demo\* \) \) -print0 | "${TAR}" c --transform "s|${KUBE_ROOT#/*}/${d}||" --null -T - | "${TAR}" x -C "${dst_dir}"
  done

  kube::release::clean_cruft

  local package_name="${RELEASE_TARS}/kubernetes-manifests.tar.gz"
  kube::release::create_tarball "${package_name}" "${release_stage}/.."
}

# Builds tarballs for each test platform containing the appropriate binaries.
function kube::release::package_test_platform_tarballs() {
  local platform
  rm -rf "${RELEASE_STAGE}/test"
  # KUBE_TEST_SERVER_PLATFORMS is a subset of KUBE_TEST_PLATFORMS,
  # so process it first.
  for platform in "${KUBE_TEST_SERVER_PLATFORMS[@]}"; do
    local platform_tag=${platform/\//-} # Replace a "/" for a "-"
    local release_stage="${RELEASE_STAGE}/test/${platform_tag}/kubernetes"
    mkdir -p "${release_stage}/test/bin"
    # This fancy expression will expand to prepend a path
    # (${LOCAL_OUTPUT_BINPATH}/${platform}/) to every item in the
    # KUBE_TEST_SERVER_BINARIES array.
    cp "${KUBE_TEST_SERVER_BINARIES[@]/#/${LOCAL_OUTPUT_BINPATH}/${platform}/}" \
      "${release_stage}/test/bin/"
  done
  for platform in "${KUBE_TEST_PLATFORMS[@]}"; do
    (
      local platform_tag=${platform/\//-} # Replace a "/" for a "-"
      kube::log::status "Starting tarball: test $platform_tag"
      local release_stage="${RELEASE_STAGE}/test/${platform_tag}/kubernetes"
      mkdir -p "${release_stage}/test/bin"

      local test_bins=("${KUBE_TEST_BINARIES[@]}")
      if [[ "${platform%/*}" = 'windows' ]]; then
        test_bins=("${KUBE_TEST_BINARIES_WIN[@]}")
      fi
      # This fancy expression will expand to prepend a path
      # (${LOCAL_OUTPUT_BINPATH}/${platform}/) to every item in the
      # test_bins array.
      cp "${test_bins[@]/#/${LOCAL_OUTPUT_BINPATH}/${platform}/}" \
        "${release_stage}/test/bin/"

      local package_name="${RELEASE_TARS}/kubernetes-test-${platform_tag}.tar.gz"
      kube::release::create_tarball "${package_name}" "${release_stage}/.."
    ) &
  done

  kube::log::status "Waiting on test tarballs"
  kube::util::wait-for-jobs || { kube::log::error "test tarball creation failed"; exit 1; }
}


# This is the stuff you need to run tests from the binary distribution.
function kube::release::package_test_tarballs() {
  kube::release::package_test_platform_tarballs

  kube::log::status "Building tarball: test portable"

  local release_stage="${RELEASE_STAGE}/test/kubernetes"
  rm -rf "${release_stage}"
  mkdir -p "${release_stage}"

  # First add test image files and other portable sources so we can create
  # the portable test tarball.
  mkdir -p "${release_stage}/test/images"
  cp -fR "${KUBE_ROOT}/test/images" "${release_stage}/test/"
  "${TAR}" c "${KUBE_TEST_PORTABLE[@]}" | "${TAR}" x -C "${release_stage}"

  kube::release::clean_cruft

  local portable_tarball_name="${RELEASE_TARS}/kubernetes-test-portable.tar.gz"
  kube::release::create_tarball "${portable_tarball_name}" "${release_stage}/.."
}

# This is all the platform-independent stuff you need to run/install kubernetes.
# Arch-specific binaries will need to be downloaded separately (possibly by
# using the bundled cluster/get-kube-binaries.sh script).
# Included in this tarball:
#   - Cluster spin up/down scripts and configs for various cloud providers
#   - Tarballs for manifest configs that are ready to be uploaded
#   - Examples (which may or may not still work)
#   - The remnants of the docs/ directory
function kube::release::package_final_tarball() {
  kube::log::status "Building tarball: final"

  # This isn't a "full" tarball anymore, but the release lib still expects
  # artifacts under "full/kubernetes/"
  local release_stage="${RELEASE_STAGE}/full/kubernetes"
  rm -rf "${release_stage}"
  mkdir -p "${release_stage}"

  mkdir -p "${release_stage}/client"
  cat <<EOF > "${release_stage}/client/README"
Client binaries are no longer included in the Kubernetes final tarball.

Run cluster/get-kube-binaries.sh to download client and server binaries.
EOF

  # We want everything in /cluster.
  cp -R "${KUBE_ROOT}/cluster" "${release_stage}/"

  mkdir -p "${release_stage}/server"
  cp "${RELEASE_TARS}/kubernetes-manifests.tar.gz" "${release_stage}/server/"
  cat <<EOF > "${release_stage}/server/README"
Server binary tarballs are no longer included in the Kubernetes final tarball.

Run cluster/get-kube-binaries.sh to download client and server binaries.
EOF

  # Include hack/lib as a dependency for the cluster/ scripts
  mkdir -p "${release_stage}/hack"
  cp -R "${KUBE_ROOT}/hack/lib" "${release_stage}/hack/"

  cp -R "${KUBE_ROOT}/docs" "${release_stage}/"
  cp "${KUBE_ROOT}/README.md" "${release_stage}/"
  cp -R "${KUBE_ROOT}/LICENSES" "${release_stage}/"

  echo "${KUBE_GIT_VERSION}" > "${release_stage}/version"

  kube::release::clean_cruft

  local package_name="${RELEASE_TARS}/kubernetes.tar.gz"
  kube::release::create_tarball "${package_name}" "${release_stage}/.."
}

# Build a release tarball.  $1 is the output tar name.  $2 is the base directory
# of the files to be packaged.  This assumes that ${2}/kubernetes is what is
# being packaged.
function kube::release::create_tarball() {
  kube::build::ensure_tar

  local tarfile=$1
  local stagingdir=$2

  "${TAR}" czf "${tarfile}" -C "${stagingdir}" kubernetes --owner=0 --group=0
}
