#!/usr/bin/env bash

# This script provides constants for the Golang binary build process

readonly OS_GO_PACKAGE=github.com/openshift/origin

readonly OS_BUILD_ENV_GOLANG="${OS_BUILD_ENV_GOLANG:-1.13}"
readonly OS_BUILD_ENV_IMAGE="${OS_BUILD_ENV_IMAGE:-openshift/origin-release:golang-${OS_BUILD_ENV_GOLANG}}"
readonly OS_REQUIRED_GO_VERSION="go${OS_BUILD_ENV_GOLANG}"
readonly OS_GLIDE_MINOR_VERSION="13"
readonly OS_REQUIRED_GLIDE_VERSION="0.$OS_GLIDE_MINOR_VERSION"

readonly OS_GOFLAGS_TAGS="include_gcs include_oss containers_image_openpgp"
readonly OS_GOFLAGS_TAGS_LINUX_AMD64="gssapi selinux"
readonly OS_GOFLAGS_TAGS_LINUX_S390X="gssapi selinux"
readonly OS_GOFLAGS_TAGS_LINUX_ARM64="gssapi selinux"
readonly OS_GOFLAGS_TAGS_LINUX_PPC64LE="gssapi selinux"

readonly OS_OUTPUT_BASEPATH="${OS_OUTPUT_BASEPATH:-_output}"
readonly OS_BASE_OUTPUT="${OS_ROOT}/${OS_OUTPUT_BASEPATH}"
readonly OS_OUTPUT_SCRIPTPATH="${OS_OUTPUT_SCRIPTPATH:-"${OS_BASE_OUTPUT}/scripts"}"

readonly OS_OUTPUT_SUBPATH="${OS_OUTPUT_SUBPATH:-${OS_OUTPUT_BASEPATH}/local}"
readonly OS_OUTPUT="${OS_ROOT}/${OS_OUTPUT_SUBPATH}"
readonly OS_OUTPUT_RELEASEPATH="${OS_OUTPUT}/releases"
readonly OS_OUTPUT_RPMPATH="${OS_OUTPUT_RELEASEPATH}/rpms"
readonly OS_OUTPUT_BINPATH="${OS_OUTPUT}/bin"
readonly OS_OUTPUT_PKGDIR="${OS_OUTPUT}/pkgdir"

readonly OS_IMAGE_COMPILE_TARGETS_LINUX=(
  vendor/k8s.io/kubernetes/cmd/kube-apiserver
  vendor/k8s.io/kubernetes/cmd/kube-controller-manager
  vendor/k8s.io/kubernetes/cmd/kube-scheduler
  vendor/k8s.io/kubernetes/cmd/kubelet
)
readonly OS_SCRATCH_IMAGE_COMPILE_TARGETS_LINUX=(
  ""
)
readonly OS_IMAGE_COMPILE_BINARIES=("${OS_SCRATCH_IMAGE_COMPILE_TARGETS_LINUX[@]##*/}" "${OS_IMAGE_COMPILE_TARGETS_LINUX[@]##*/}")

readonly OS_GOVET_BLACKLIST=(
)

#If you update this list, be sure to get the images/origin/Dockerfile
readonly OS_BINARY_RELEASE_SERVER_LINUX=(
  './*'
)
readonly OS_BINARY_RELEASE_CLIENT_EXTRA=(
  ${OS_ROOT}/README.md
  ${OS_ROOT}/LICENSE
)

# os::build::get_product_vars exports variables that we expect to change
# depending on the distribution of Origin
function os::build::get_product_vars() {
  export OS_BUILD_LDFLAGS_IMAGE_PREFIX="${OS_IMAGE_PREFIX:-"openshift/origin"}"
  export OS_BUILD_LDFLAGS_DEFAULT_IMAGE_STREAMS="${OS_BUILD_LDFLAGS_DEFAULT_IMAGE_STREAMS:-"centos7"}"
}

# os::build::ldflags calculates the -ldflags argument for building OpenShift
function os::build::ldflags() {
  # Run this in a subshell to prevent settings/variables from leaking.
  set -o errexit
  set -o nounset
  set -o pipefail

  cd "${OS_ROOT}"

  os::build::version::get_vars
  os::build::get_product_vars

  local buildDate="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"

  declare -a ldflags=(
    "-s"
    "-w"
  )

  ldflags+=($(os::build::ldflag "${OS_GO_PACKAGE}/pkg/version.majorFromGit" "${OS_GIT_MAJOR}"))
  ldflags+=($(os::build::ldflag "${OS_GO_PACKAGE}/pkg/version.minorFromGit" "${OS_GIT_MINOR}"))
  ldflags+=($(os::build::ldflag "${OS_GO_PACKAGE}/pkg/version.versionFromGit" "${OS_GIT_VERSION}"))
  ldflags+=($(os::build::ldflag "${OS_GO_PACKAGE}/pkg/version.commitFromGit" "${OS_GIT_COMMIT}"))
  ldflags+=($(os::build::ldflag "${OS_GO_PACKAGE}/pkg/version.gitTreeState" "${OS_GIT_TREE_STATE}"))
  ldflags+=($(os::build::ldflag "${OS_GO_PACKAGE}/pkg/version.buildDate" "${buildDate}"))

  ldflags+=($(os::build::ldflag "${OS_GO_PACKAGE}/vendor/k8s.io/component-base/version.gitMajor" "${KUBE_GIT_MAJOR}"))
  ldflags+=($(os::build::ldflag "${OS_GO_PACKAGE}/vendor/k8s.io/component-base/version.gitMinor" "${KUBE_GIT_MINOR}"))
  ldflags+=($(os::build::ldflag "${OS_GO_PACKAGE}/vendor/k8s.io/component-base/version.gitCommit" "${OS_GIT_COMMIT}"))
  ldflags+=($(os::build::ldflag "${OS_GO_PACKAGE}/vendor/k8s.io/component-base/version.gitVersion" "${KUBE_GIT_VERSION}"))
  ldflags+=($(os::build::ldflag "${OS_GO_PACKAGE}/vendor/k8s.io/component-base/version.buildDate" "${buildDate}"))
  ldflags+=($(os::build::ldflag "${OS_GO_PACKAGE}/vendor/k8s.io/component-base/version.gitTreeState" "clean"))
  
  ldflags+=($(os::build::ldflag "${OS_GO_PACKAGE}/vendor/k8s.io/client-go/pkg/version.gitMajor" "${KUBE_GIT_MAJOR}"))
  ldflags+=($(os::build::ldflag "${OS_GO_PACKAGE}/vendor/k8s.io/client-go/pkg/version.gitMinor" "${KUBE_GIT_MINOR}"))
  ldflags+=($(os::build::ldflag "${OS_GO_PACKAGE}/vendor/k8s.io/client-go/pkg/version.gitCommit" "${OS_GIT_COMMIT}"))
  ldflags+=($(os::build::ldflag "${OS_GO_PACKAGE}/vendor/k8s.io/client-go/pkg/version.gitVersion" "${KUBE_GIT_VERSION}"))
  ldflags+=($(os::build::ldflag "${OS_GO_PACKAGE}/vendor/k8s.io/client-go/pkg/version.buildDate" "${buildDate}"))
  ldflags+=($(os::build::ldflag "${OS_GO_PACKAGE}/vendor/k8s.io/client-go/pkg/version.gitTreeState" "clean")
)

  # The -ldflags parameter takes a single string, so join the output.
  echo "${ldflags[*]-}"
}
readonly -f os::build::ldflags

# os::util::list_go_src_files lists files we consider part of our project
# source code, useful for tools that iterate over source to provide vet-
# ting or linting, etc.
#
# Globals:
#  None
# Arguments:
#  None
# Returns:
#  None
function os::util::list_go_src_files() {
	find . -not \( \
		\( \
		-wholename './_output' \
		-o -wholename './.*' \
		-o -wholename './pkg/assets/bindata.go' \
		-o -wholename './pkg/assets/*/bindata.go' \
		-o -wholename './pkg/oc/clusterup/manifests/bindata.go' \
		-o -wholename './openshift.local.*' \
		-o -wholename './test/extended/testdata/bindata.go' \
		-o -wholename '*/vendor/*' \
		-o -wholename './assets/bower_components/*' \
		\) -prune \
	\) -name '*.go' | sort -u
}
readonly -f os::util::list_go_src_files

# os::util::list_go_src_dirs lists dirs in origin/ and cmd/ dirs excluding
# doc.go, useful for tools that iterate over source to provide vetting or
# linting, or for godep-save etc.
#
# Globals:
#  None
# Arguments:
#  None
# Returns:
#  None
function os::util::list_go_src_dirs() {
    go list -e ./... | grep -Ev "/(third_party|vendor|staging|clientset_generated)/" | LC_ALL=C sort -u
}
readonly -f os::util::list_go_src_dirs

# os::util::list_go_deps outputs the list of dependencies for the project.
function os::util::list_go_deps() {
  go list -f '{{.ImportPath}}{{.Imports}}' ./test/... ./pkg/... ./cmd/... ./vendor/k8s.io/... | tr '[]' '  ' |
    sed -e 's|github.com/openshift/origin/vendor/||g' |
    sed -e 's|k8s.io/kubernetes/staging/src/||g'
}

# os::util::list_test_packages_under lists all packages containing Golang test files that we
# want to run as unit tests under the given base dir in the source tree
function os::util::list_test_packages_under() {
    local basedir=$*

    # we do not quote ${basedir} to allow for multiple arguments to be passed in as well as to allow for
    # arguments that use expansion, e.g. paths containing brace expansion or wildcards
    # we do not quote ${basedir} to allow for multiple arguments to be passed in as well as to allow for
    # arguments that use expansion, e.g. paths containing brace expansion or wildcards
    find ${basedir} -not \(                   \
        \(                                    \
              -path 'vendor'                  \
              -o -path '*_output'             \
              -o -path '*.git'                \
              -o -path '*openshift.local.*'   \
              -o -path '*vendor/*'            \
              -o -path '*assets/node_modules' \
              -o -path '*test/*'              \
              -o -path '*pkg/proxy'           \
              -o -path '*k8s.io/kubernetes/cluster/gce*' \
        \) -prune                             \
    \) -name '*_test.go' | xargs -n1 dirname | sort -u | xargs -n1 printf "${OS_GO_PACKAGE}/%s\n"

    local kubernetes_path="vendor/k8s.io/kubernetes"

    if [[ -n "${TEST_KUBE-}" ]]; then
      # we need to find all of the kubernetes test suites, excluding those we directly whitelisted before, the end-to-end suite, and
      # cmd wasn't done before using glide and constantly flakes
      # the forked etcd packages are used only by the gce etcd containers
      find -L vendor/k8s.io/{api,apimachinery,apiserver,client-go,kube-aggregator,kubernetes} -not \( \
        \(                                                                                          \
          -path "${kubernetes_path}/staging"                                                        \
          -o -path "${kubernetes_path}/cmd"                                                         \
          -o -path "${kubernetes_path}/test"                                                        \
          -o -path "${kubernetes_path}/third_party/forked/etcd*"                                    \
          -o -path "${kubernetes_path}/cluster/gce" \
       \) -prune                                                                                   \
      \) -name '*_test.go' | cut -f 2- -d / | xargs -n1 dirname | sort -u | xargs -n1 printf "${OS_GO_PACKAGE}/vendor/%s\n"
    else
      echo "${OS_GO_PACKAGE}/vendor/k8s.io/api/..."
      echo "${OS_GO_PACKAGE}/vendor/k8s.io/kubernetes/pkg/api/..."
      echo "${OS_GO_PACKAGE}/vendor/k8s.io/kubernetes/pkg/apis/..."
    fi
}
readonly -f os::util::list_test_packages_under

# Generates the .syso file used to add compile-time VERSIONINFO metadata to the
# Windows binary.
function os::build::generate_windows_versioninfo() {
  os::build::version::get_vars
  local major="${OS_GIT_MAJOR}"
  local minor="${OS_GIT_MINOR%+}"
  local patch="${OS_GIT_PATCH}"
  local windows_versioninfo_file=`mktemp --suffix=".versioninfo.json"`
  cat <<EOF >"${windows_versioninfo_file}"
{
       "FixedFileInfo":
       {
               "FileVersion": {
                       "Major": ${major},
                       "Minor": ${minor},
                       "Patch": ${patch}
               },
               "ProductVersion": {
                       "Major": ${major},
                       "Minor": ${minor},
                       "Patch": ${patch}
               },
               "FileFlagsMask": "3f",
               "FileFlags ": "00",
               "FileOS": "040004",
               "FileType": "01",
               "FileSubType": "00"
       },
       "StringFileInfo":
       {
               "Comments": "",
               "CompanyName": "Red Hat, Inc.",
               "InternalName": "openshift client",
               "FileVersion": "${OS_GIT_VERSION}",
               "InternalName": "oc",
               "LegalCopyright": "© Red Hat, Inc. Licensed under the Apache License, Version 2.0",
               "LegalTrademarks": "",
               "OriginalFilename": "oc.exe",
               "PrivateBuild": "",
               "ProductName": "OpenShift Client",
               "ProductVersion": "${OS_GIT_VERSION}",
               "SpecialBuild": ""
       },
       "VarFileInfo":
       {
               "Translation": {
                       "LangID": "0409",
                       "CharsetID": "04B0"
               }
       }
}
EOF
  goversioninfo -o ${OS_ROOT}/vendor/github.com/openshift/oc/cmd/oc/oc.syso ${windows_versioninfo_file}
}
readonly -f os::build::generate_windows_versioninfo

# Removes the .syso file used to add compile-time VERSIONINFO metadata to the
# Windows binary.
function os::build::clean_windows_versioninfo() {
  rm ${OS_ROOT}/vendor/github.com/openshift/oc/cmd/oc/oc.syso
}
readonly -f os::build::clean_windows_versioninfo

# OS_ALL_IMAGES is the list of images built by os::build::images.
readonly OS_ALL_IMAGES=(
  origin-hyperkube
  origin-tests
)

# os::build::check_binaries ensures that binary sizes do not grow without approval.
function os::build::check_binaries() {
  platform=$(os::build::host_platform)
  if [[ "${platform}" != "linux/amd64" && "${platform}" != "darwin/amd64" ]]; then
    return 0
  fi
  duexe="du"

  # In OSX, the 'du' binary does not provide the --apparent-size flag. However, the homebrew
  # provide GNU coreutils which provide 'gdu' binary which is equivalent to Linux du.
  # For now, if the 'gdu' binary is not installed, print annoying warning and don't check the
  # binary size (the CI will capture possible violation anyway).
  if [[ "${platform}" == "darwin/amd64" ]]; then
    duexe=$(which gdu || true)
    if [[ -z "${duexe}" ]]; then
        os::log::warning "Unable to locate 'gdu' binary to determine size of the binary. Please install it using: 'brew install coreutils'"
        return 0
    fi
  fi

  if [[ -f "${OS_OUTPUT_BINPATH}/${platform}/pod" ]]; then
    size=$($duexe --apparent-size -m "${OS_OUTPUT_BINPATH}/${platform}/pod" | cut -f 1)
    if [[ "${size}" -gt "2" ]]; then
      os::log::fatal "pod binary has grown substantially to ${size}. You must have approval before bumping this limit."
    fi
  fi
}

# os::build::images builds all images in this repo.
function os::build::images() {
  # Create link to file if the FS supports hardlinks, otherwise copy the file
  function ln_or_cp {
    local src_file=$1
    local dst_dir=$2
    if os::build::archive::internal::is_hardlink_supported "${dst_dir}" ; then
      ln -f "${src_file}" "${dst_dir}"
    else
      cp -pf "${src_file}" "${dst_dir}"
    fi
  }

  # determine the correct tag prefix
  tag_prefix="${OS_IMAGE_PREFIX:-"openshift/origin"}"

  # images that depend on "${tag_prefix}-source" or "${tag_prefix}-base"
  ( os::build::image "${tag_prefix}-hyperkube"               images/hyperkube ) &

  for i in $(jobs -p); do wait "$i"; done

  # images that depend on "${tag_prefix}-cli" or hyperkube
  ( os::build::image "${tag_prefix}-tests"          images/tests ) &

  for i in $(jobs -p); do wait "$i"; done
}
readonly -f os::build::images
