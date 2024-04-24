#!/usr/bin/env bash

# Copyright 2017 The Kubernetes Authors.
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

TASK=${1}
WHAT=${2}

# docker buildx command is still experimental as of Docker 19.03.0
export DOCKER_CLI_EXPERIMENTAL="enabled"

# Connecting to a Remote Docker requires certificates for authentication, which can be found
# at this path. By default, they can be found in the ${HOME} folder. We're expecting to find
# here ".docker-${os_version}" folders which contains the necessary certificates.
DOCKER_CERT_BASE_PATH="${DOCKER_CERT_BASE_PATH:-${HOME}}"

KUBE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
source "${KUBE_ROOT}/hack/lib/logging.sh"
source "${KUBE_ROOT}/hack/lib/util.sh"

# Mapping of go ARCH to actual architectures shipped part of multiarch/qemu-user-static project
declare -A QEMUARCHS=( ["amd64"]="x86_64" ["arm"]="arm" ["arm64"]="aarch64" ["ppc64le"]="ppc64le" ["s390x"]="s390x" )

# NOTE(claudiub): In the test image build jobs, this script is not being run in a git repository,
# which would cause git log to fail. Instead, we can use the GIT_COMMIT_ID set in cloudbuild.yaml.
GIT_COMMIT_ID=$(git log -1 --format=%h || echo "${GIT_COMMIT_ID}")
windows_os_versions=(1809 ltsc2022)
declare -A WINDOWS_OS_VERSIONS_MAP

initWindowsOsVersions() {
  for os_version in "${windows_os_versions[@]}"; do
    img_base="mcr.microsoft.com/windows/nanoserver:${os_version}"
    # we use awk to also trim the quotes around the OS version string.
    full_version=$(docker manifest inspect "${img_base}" | grep "os.version" | head -n 1 | awk -F\" '{print $4}') || true
    WINDOWS_OS_VERSIONS_MAP["${os_version}"]="${full_version}"
  done
}

initWindowsOsVersions

# Returns list of all supported architectures from BASEIMAGE file
listOsArchs() {
  local image=${1}
  cut -d "=" -f 1 "${image}"/BASEIMAGE
}

splitOsArch() {
  local image=${1}
  local os_arch=${2}

  if [[ $os_arch =~ .*/.*/.* ]]; then
    # for Windows, we have to support both LTS and SAC channels, so we're building multiple Windows images.
    # the format for this case is: OS/ARCH/OS_VERSION.
    os_name=$(echo "$os_arch" | cut -d "/" -f 1)
    arch=$(echo "$os_arch" | cut -d "/" -f 2)
    os_version=$(echo "$os_arch" | cut -d "/" -f 3)
    suffix="$os_name-$arch-$os_version"
  elif [[ $os_arch =~ .*/.* ]]; then
    os_name=$(echo "$os_arch" | cut -d "/" -f 1)
    arch=$(echo "$os_arch" | cut -d "/" -f 2)
    os_version=""
    suffix="$os_name-$arch"
  else
    echo "The BASEIMAGE file for the ${image} image is not properly formatted. Expected entries to start with 'os/arch', found '${os_arch}' instead."
    exit 1
  fi
}

# Returns baseimage need to used in Dockerfile for any given architecture
getBaseImage() {
  os_arch=$1
  grep "${os_arch}=" BASEIMAGE | cut -d= -f2
}

# This function will build test image for all the architectures
# mentioned in BASEIMAGE file. In the absence of BASEIMAGE file,
# it will build for all the supported arch list - amd64, arm,
# arm64, ppc64le, s390x
build() {
  local image=${1}
  local img_folder=${1}
  local output_type=${2}
  docker_version_check

  if [[ -f "${img_folder}/BASEIMAGE" ]]; then
    os_archs=$(listOsArchs "$image")
  else
    # prepend linux/ to the QEMUARCHS items.
    os_archs=$(printf 'linux/%s\n' "${!QEMUARCHS[@]}")
  fi

  # image tag
  TAG=$(<"${img_folder}/VERSION")

  alias_name="$(cat "${img_folder}/ALIAS" 2>/dev/null || true)"
  if [[ -n "${alias_name}" ]]; then
    echo "Found an alias for '${image}'. Building / tagging image as '${alias_name}.'"
    image="${alias_name}"
  fi

  kube::util::ensure-gnu-sed
  kube::util::ensure-docker-buildx

  for os_arch in ${os_archs}; do
    splitOsArch "${image}" "${os_arch}"
    if [[ "${os_name}" == "windows" && "${output_type}" == "docker" ]]; then
      echo "Cannot build the image '${image}' for ${os_arch}. Built Windows container images need to be pushed to a registry."
      continue
    fi

    echo "Building image for ${image} OS/ARCH: ${os_arch}..."

    # Create a temporary directory for every architecture and copy the image content
    # and build the image from temporary directory
    temp_dir="$(kube::realpath "$(mktemp -d -t "$(basename "$0").XXXXXX")")"
    kube::util::trap_add "rm -rf ${temp_dir}" EXIT

    cp -r "${img_folder}"/* "${temp_dir}"
    if [[ -f ${img_folder}/Makefile ]]; then
      # make bin will take care of all the prerequisites needed
      # for building the docker image
      make -C "${img_folder}" bin OS="${os_name}" ARCH="${arch}" TARGET="${temp_dir}"
    fi
    pushd "${temp_dir}"

    # NOTE(claudiub): Some Windows images might require their own Dockerfile
    # while simpler ones will not. If we're building for Windows, check if
    # "Dockerfile_windows" exists or not.
    dockerfile_name="Dockerfile"
    if [[ "$os_name" = "windows" && -f "Dockerfile_windows" ]]; then
      dockerfile_name="Dockerfile_windows"
    fi

    base_image=""
    if [[ -f BASEIMAGE ]]; then
      base_image=$(getBaseImage "${os_arch}" | "${SED}" "s|REGISTRY|${REGISTRY}|g")
      "${SED}" -i "s|BASEARCH|${arch}|g" $dockerfile_name
    fi

    # Only the cross-build on x86 is guaranteed by far, other arches like aarch64 doesn't support cross-build
    # thus, there is no need to tackle a disability feature on those platforms, and also help to prevent from
    # ending up a wrong image tag on non-amd64 platforms.
    build_arch=$(uname -m)
    if [[ ${build_arch} = 'x86_64' ]]; then
        # copy the qemu-*-static binary to docker image to build the multi architecture image on x86 platform
        if grep -q 'CROSS_BUILD_' Dockerfile; then
          if [[ "${arch}" = 'amd64' ]]; then
            "${SED}" -i '/CROSS_BUILD_/d' Dockerfile
          else
            "${SED}" -i "s|QEMUARCH|${QEMUARCHS[$arch]}|g" Dockerfile
            # Register qemu-*-static for all supported processors except the current one
            echo 'Registering qemu-*-static binaries in the kernel'
            local sudo=""
            if [[ $(id -u) -ne 0 ]]; then
	            sudo="sudo"
            fi
            ${sudo} docker run --rm --privileged tonistiigi/binfmt:latest --install all
            curl -sSL https://github.com/multiarch/qemu-user-static/releases/download/"${QEMUVERSION}"/x86_64_qemu-"${QEMUARCHS[$arch]}"-static.tar.gz | tar -xz -C "${temp_dir}"
            # Ensure we don't get surprised by umask settings
            chmod 0755 "${temp_dir}/qemu-${QEMUARCHS[$arch]}-static"
            "${SED}" -i 's/CROSS_BUILD_//g' Dockerfile
          fi
        fi
    elif [[ "${QEMUARCHS[$arch]}" != "${build_arch}" ]]; then
		echo "skip cross-build $arch on non-supported platform ${build_arch}."
		popd
        continue
    else
        "${SED}" -i '/CROSS_BUILD_/d' Dockerfile
    fi

    # `--provenance=false --sbom=false` is set to avoid creating a manifest list: https://github.com/kubernetes/kubernetes/issues/123266
    docker buildx build --progress=plain --no-cache --pull --output=type="${output_type}" --platform "${os_name}/${arch}" --provenance=false --sbom=false \
        --build-arg BASEIMAGE="${base_image}" --build-arg REGISTRY="${REGISTRY}" --build-arg OS_VERSION="${os_version}" \
        -t "${REGISTRY}/${image}:${TAG}-${suffix}" -f "${dockerfile_name}" \
	--label "image_version=${TAG}" --label "commit_id=${GIT_COMMIT_ID}" \
	--label "git_url=https://github.com/kubernetes/kubernetes/tree/${GIT_COMMIT_ID}/test/images/${img_folder}" .

    popd
  done
}

docker_version_check() {
  # docker manifest annotate --os-version has been introduced in 20.10.0,
  # so we need to make sure we have it.
  docker_version=$(docker version --format '{{.Client.Version}}' | cut -d"-" -f1)
  if [[ ${docker_version} != 20.10.0 && ${docker_version} < 20.10.0 ]]; then
    echo "Minimum docker version 20.10.0 is required for annotating the OS Version in the manifest list images: ${docker_version}]"
    exit 1
  fi
}

# This function will push the docker images
push() {
  local image=${1}
  docker_version_check

  TAG=$(<"${image}"/VERSION)
  if [[ -f ${image}/BASEIMAGE ]]; then
    os_archs=$(listOsArchs "$image")
  else
    # prepend linux/ to the QEMUARCHS items.
    os_archs=$(printf 'linux/%s\n' "${!QEMUARCHS[@]}")
  fi

  pushd "${image}"
  alias_name="$(cat ALIAS 2>/dev/null || true)"
  if [[ -n "${alias_name}" ]]; then
    echo "Found an alias for '${image}'. Pushing image as '${alias_name}.'"
    image="${alias_name}"
  fi

  kube::util::ensure-gnu-sed

  # reset manifest list; needed in case multiple images are being built / pushed.
  manifest=()
  # Make os_archs list into image manifest. Eg: 'linux/amd64 linux/ppc64le' to '${REGISTRY}/${image}:${TAG}-linux-amd64 ${REGISTRY}/${image}:${TAG}-linux-ppc64le'
  while IFS='' read -r line; do manifest+=("$line"); done < <(echo "$os_archs" | "${SED}" "s~\/~-~g" | "${SED}" -e "s~[^ ]*~$REGISTRY\/$image:$TAG\-&~g")
  docker manifest create --amend "${REGISTRY}/${image}:${TAG}" "${manifest[@]}"

  for os_arch in ${os_archs}; do
    splitOsArch "${image}" "${os_arch}"

    # For Windows images, we also need to include the "os.version" in the manifest list, so the Windows node
    # can pull the proper image it needs.
    if [[ "$os_name" = "windows" ]]; then
      full_version="${WINDOWS_OS_VERSIONS_MAP[$os_version]}"
      docker manifest annotate --os "${os_name}" --arch "${arch}" --os-version "${full_version}" "${REGISTRY}/${image}:${TAG}" "${REGISTRY}/${image}:${TAG}-${suffix}"
    else
      docker manifest annotate --os "${os_name}" --arch "${arch}" "${REGISTRY}/${image}:${TAG}" "${REGISTRY}/${image}:${TAG}-${suffix}"
    fi
  done
  popd
  docker manifest push --purge "${REGISTRY}/${image}:${TAG}"
}

# This function is for building AND pushing images. Useful if ${WHAT} is "all-conformance".
# This will allow images to be pushed immediately after they've been built.
build_and_push() {
  local image=${1}
  build "${image}" "registry"
  push "${image}"
}

# This function is for building the go code
bin() {
  local arch_prefix=""
  if [[ "${ARCH:-}" == "arm" ]]; then
    arch_prefix="GOARM=${GOARM:-7}"
  fi
  for SRC in "$@";
  do
  docker run --rm -v "${TARGET}:${TARGET}:Z" -v "${KUBE_ROOT}":/go/src/k8s.io/kubernetes:Z \
        golang:"${GOLANG_VERSION}" \
        /bin/bash -c "\
                cd /go/src/k8s.io/kubernetes/test/images/${SRC_DIR} && \
                CGO_ENABLED=0 ${arch_prefix} GOOS=${OS} GOARCH=${ARCH} go build -a -installsuffix cgo --ldflags \"-w ${LD_FLAGS:-}\" -o ${TARGET}/${SRC} ./$(dirname "${SRC}")"
  done
}

shift

if [[ "${WHAT}" == "all-conformance" ]]; then
  # NOTE(claudiub): Building *ALL* the images under the kubernetes/test/images folder takes an extremely
  # long time (especially some images), and some images are rarely used and rarely updated, so there's
  # no point in rebuilding all of them every time. This will only build the Conformance-related images.
  # Discussed during Conformance Office Hours Meeting (2019.12.17):
  # https://docs.google.com/document/d/1W31nXh9RYAb_VaYkwuPLd1hFxuRX3iU0DmaQ4lkCsX8/edit#heading=h.l87lu17xm9bh
  shift
  conformance_images=("busybox" "agnhost" "jessie-dnsutils" "kitten" "nautilus" "nonewprivs" "resource-consumer" "sample-apiserver")
  for image in "${conformance_images[@]}"; do
    "${TASK}" "${image}" "$@"
  done
else
  "${TASK}" "$@"
fi
