#!/usr/bin/env bash

# Copyright 2023 The Kubernetes Authors.
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

# This presents several functions for packages which want to use kubernetes
# code-generation tools.

set -o errexit
set -o nounset
set -o pipefail

KUBE_CODEGEN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

function kube::codegen::internal::git_find() {
    # Similar to find but faster and easier to understand.  We want to include
    # modified and untracked files because this might be running against code
    # which is not tracked by git yet.
    git ls-files -cmo --exclude-standard "$@"
}

function kube::codegen::internal::git_grep() {
    # We want to include modified and untracked files because this might be
    # running against code which is not tracked by git yet.
    git grep --untracked "$@"
}

# Generate tagged helper code: conversions, deepcopy, and defaults
#
# USAGE: kube::codegen::gen_helpers [FLAGS] <input-dir>
#
# <input-dir>
#   The root directory under which to search for Go files which request code to
#   be generated.  This must be a local path, not a Go package.
#
# FLAGS:
#
#   --boilerplate <string = path_to_kube_codegen_boilerplate>
#     An optional override for the header file to insert into generated files.
#
#   --extra-peer-dir <string>
#     An optional list (this flag may be specified multiple times) of "extra"
#     directories to consider during conversion generation.
#
function kube::codegen::gen_helpers() {
    local in_dir=""
    local boilerplate="${KUBE_CODEGEN_ROOT}/hack/boilerplate.go.txt"
    local v="${KUBE_VERBOSE:-0}"
    local extra_peers=()

    while [ "$#" -gt 0 ]; do
        case "$1" in
            "--boilerplate")
                boilerplate="$2"
                shift 2
                ;;
            "--extra-peer-dir")
                extra_peers+=("$2")
                shift 2
                ;;
            *)
                if [[ "$1" =~ ^-- ]]; then
                    echo "unknown argument: $1" >&2
                    return 1
                fi
                if [ -n "$in_dir" ]; then
                    echo "too many arguments: $1 (already have $in_dir)" >&2
                    return 1
                fi
                in_dir="$1"
                shift
                ;;
        esac
    done

    if [ -z "${in_dir}" ]; then
        echo "input-dir argument is required" >&2
        return 1
    fi

    (
        # To support running this from anywhere, first cd into this directory,
        # and then install with forced module mode on and fully qualified name.
        cd "${KUBE_CODEGEN_ROOT}"
        BINS=(
            conversion-gen
            deepcopy-gen
            defaulter-gen
        )
        # shellcheck disable=2046 # printf word-splitting is intentional
        GO111MODULE=on go install $(printf "k8s.io/code-generator/cmd/%s " "${BINS[@]}")
    )
    # Go installs in $GOBIN if defined, and $GOPATH/bin otherwise
    gobin="${GOBIN:-$(go env GOPATH)/bin}"

    # Deepcopy
    #
    local input_pkgs=()
    while read -r dir; do
        pkg="$(cd "${dir}" && GO111MODULE=on go list -find .)"
        input_pkgs+=("${pkg}")
    done < <(
        ( kube::codegen::internal::git_grep -l --null \
            -e '+k8s:deepcopy-gen=' \
            ":(glob)${in_dir}"/'**/*.go' \
            || true \
        ) | while read -r -d $'\0' F; do dirname $F; done \
          | LC_ALL=C sort -u
    )

    if [ "${#input_pkgs[@]}" != 0 ]; then
        echo "Generating deepcopy code for ${#input_pkgs[@]} targets"

        kube::codegen::internal::git_find -z \
            ":(glob)${in_dir}"/'**/zz_generated.deepcopy.go' \
            | xargs -0 rm -f

        local input_args=()
        for arg in "${input_pkgs[@]}"; do
            input_args+=("--input-dirs" "$arg")
        done
        "${gobin}/deepcopy-gen" \
            -v "${v}" \
            -O zz_generated.deepcopy \
            --go-header-file "${boilerplate}" \
            "${input_args[@]}"
    fi

    # Defaults
    #
    local input_pkgs=()
    while read -r dir; do
        pkg="$(cd "${dir}" && GO111MODULE=on go list -find .)"
        input_pkgs+=("${pkg}")
    done < <(
        ( kube::codegen::internal::git_grep -l --null \
            -e '+k8s:defaulter-gen=' \
            ":(glob)${in_dir}"/'**/*.go' \
            || true \
        ) | while read -r -d $'\0' F; do dirname $F; done \
          | LC_ALL=C sort -u
    )

    if [ "${#input_pkgs[@]}" != 0 ]; then
        echo "Generating defaulter code for ${#input_pkgs[@]} targets"

        kube::codegen::internal::git_find -z \
            ":(glob)${in_dir}"/'**/zz_generated.defaults.go' \
            | xargs -0 rm -f

        local input_args=()
        for arg in "${input_pkgs[@]}"; do
            input_args+=("--input-dirs" "$arg")
        done
        "${gobin}/defaulter-gen" \
            -v "${v}" \
            -O zz_generated.defaults \
            --go-header-file "${boilerplate}" \
            "${input_args[@]}"
    fi

    # Conversions
    #
    local input_pkgs=()
    while read -r dir; do
        pkg="$(cd "${dir}" && GO111MODULE=on go list -find .)"
        input_pkgs+=("${pkg}")
    done < <(
        ( kube::codegen::internal::git_grep -l --null \
            -e '+k8s:conversion-gen=' \
            ":(glob)${in_dir}"/'**/*.go' \
            || true \
        ) | while read -r -d $'\0' F; do dirname $F; done \
          | LC_ALL=C sort -u
    )

    if [ "${#input_pkgs[@]}" != 0 ]; then
        echo "Generating conversion code for ${#input_pkgs[@]} targets"

        kube::codegen::internal::git_find -z \
            ":(glob)${in_dir}"/'**/zz_generated.conversion.go' \
            | xargs -0 rm -f

        local input_args=()
        for arg in "${input_pkgs[@]}"; do
            input_args+=("--input-dirs" "$arg")
        done
        local extra_peer_args=()
        for arg in "${extra_peers[@]:+"${extra_peers[@]}"}"; do
            extra_peer_args+=("--extra-peer-dirs" "$arg")
        done
        "${gobin}/conversion-gen" \
            -v "${v}" \
            -O zz_generated.conversion \
            --go-header-file "${boilerplate}" \
            "${extra_peer_args[@]:+"${extra_peer_args[@]}"}" \
            "${input_args[@]}"
    fi
}

# Generate openapi code
#
# USAGE: kube::codegen::gen_openapi [FLAGS] <input-dir>
#
# <input-dir>
#   The root directory under which to search for Go files which request openapi
#   to be generated.  This must be a local path, not a Go package.
#
# FLAGS:
#
#   --output-dir <string>
#     The directory into which to emit code.
#
#   --extra-pkgs <string>
#     An optional list of additional packages to be imported during openapi
#     generation.  The argument must be Go package syntax, e.g.
#     "k8s.io/foo/bar".  It may be a single value or a comma-delimited list.
#     This flag may be repeated.
#
#   --report-filename <string = "/dev/null">
#     An optional path at which to write an API violations report.  "-" means
#     stdout.
#
#   --update-report
#     If specified, update the report file in place, rather than diffing it.
#
#   --boilerplate <string = path_to_kube_codegen_boilerplate>
#     An optional override for the header file to insert into generated files.
#
function kube::codegen::gen_openapi() {
    local in_dir=""
    local out_dir=""
    local extra_pkgs=()
    local report="/dev/null"
    local update_report=""
    local boilerplate="${KUBE_CODEGEN_ROOT}/hack/boilerplate.go.txt"
    local v="${KUBE_VERBOSE:-0}"

    while [ "$#" -gt 0 ]; do
        case "$1" in
            "--output-dir")
                out_dir="$2"
                shift 2
                ;;
            "--extra-pkgs")
                extra_pkgs+=("$2")
                shift 2
                ;;
            "--report-filename")
                report="$2"
                shift 2
                ;;
            "--update-report")
                update_report="true"
                shift
                ;;
            "--boilerplate")
                boilerplate="$2"
                shift 2
                ;;
            *)
                if [[ "$1" =~ ^-- ]]; then
                    echo "unknown argument: $1" >&2
                    return 1
                fi
                if [ -n "$in_dir" ]; then
                    echo "too many arguments: $1 (already have $in_dir)" >&2
                    return 1
                fi
                in_dir="$1"
                shift
                ;;
        esac
    done

    if [ -z "${in_dir}" ]; then
        echo "input-dir argument is required" >&2
        return 1
    fi
    if [ -z "${out_dir}" ]; then
        echo "--output-dir is required" >&2
        return 1
    fi

    local new_report
    new_report="$(mktemp -t "$(basename "$0").api_violations.XXXXXX")"
    if [ -n "${update_report}" ]; then
        new_report="${report}"
    fi

    (
        # To support running this from anywhere, first cd into this directory,
        # and then install with forced module mode on and fully qualified name.
        cd "${KUBE_CODEGEN_ROOT}"
        BINS=(
            openapi-gen
        )
        # shellcheck disable=2046 # printf word-splitting is intentional
        GO111MODULE=on go install $(printf "k8s.io/code-generator/cmd/%s " "${BINS[@]}")
    )
    # Go installs in $GOBIN if defined, and $GOPATH/bin otherwise
    gobin="${GOBIN:-$(go env GOPATH)/bin}"

    local input_pkgs=( "${extra_pkgs[@]:+"${extra_pkgs[@]}"}")
    while read -r dir; do
        pkg="$(cd "${dir}" && GO111MODULE=on go list -find .)"
        input_pkgs+=("${pkg}")
    done < <(
        ( kube::codegen::internal::git_grep -l --null \
            -e '+k8s:openapi-gen=' \
            ":(glob)${in_dir}"/'**/*.go' \
            || true \
        ) | while read -r -d $'\0' F; do dirname $F; done \
          | LC_ALL=C sort -u
    )

    if [ "${#input_pkgs[@]}" != 0 ]; then
        echo "Generating openapi code for ${#input_pkgs[@]} targets"

        kube::codegen::internal::git_find -z \
            ":(glob)${in_dir}"/'**/zz_generated.openapi.go' \
            | xargs -0 rm -f

        local inputs=()
        for arg in "${input_pkgs[@]}"; do
            inputs+=("--input-dirs" "$arg")
        done
        "${gobin}/openapi-gen" \
            -v "${v}" \
            -O zz_generated.openapi \
            --go-header-file "${boilerplate}" \
            --output-base "${out_dir}" \
            --report-filename "${new_report}" \
            --input-dirs "k8s.io/apimachinery/pkg/apis/meta/v1" \
            --input-dirs "k8s.io/apimachinery/pkg/runtime" \
            --input-dirs "k8s.io/apimachinery/pkg/version" \
            "${inputs[@]}"
    fi

    touch "${report}" # in case it doesn't exist yet
    if ! diff -u "${report}" "${new_report}"; then
        echo -e "ERROR:"
        echo -e "\tAPI rule check failed for ${report}: new reported violations"
        echo -e "\tPlease read api/api-rules/README.md"
        return 1
    fi
}

# Generate client code
#
# USAGE: kube::codegen::gen_client [FLAGS] <input-dir>
#
# <input-dir>
#   The root package under which to search for Go files which request clients
#   to be generated. This must be a local path, not a Go package.
#
# FLAGS:
#   --one-input-api <string>
#     A specific API (a directory) under the input-dir for which to generate a
#     client.  If this is not set, clients for all APIs under the input-dir
#     will be generated (under the --output-pkg-root).
#
#   --output-dir <string>
#     The root directory under which to emit code.  Each aspect of client
#     generation will make one or more subdirectories.
#
#   --output-pkg <string>
#     The Go package path (import path) of the --output-dir.  Each aspect of
#     client generation will make one or more sub-packages.
#
#   --boilerplate <string = path_to_kube_codegen_boilerplate>
#     An optional override for the header file to insert into generated files.
#
#   --clientset-name <string = "clientset">
#     An optional override for the leaf name of the generated "clientset" directory.
#
#   --versioned-name <string = "versioned">
#     An optional override for the leaf name of the generated
#     "<clientset>/versioned" directory.
#
#   --with-applyconfig
#     Enables generation of applyconfiguration files.
#
#   --applyconfig-name <string = "applyconfiguration">
#     An optional override for the leaf name of the generated "applyconfiguration" directory.
#
#   --with-watch
#     Enables generation of listers and informers for APIs which support WATCH.
#
#   --listers-name <string = "listers">
#     An optional override for the leaf name of the generated "listers" directory.
#
#   --informers-name <string = "informers">
#     An optional override for the leaf name of the generated "informers" directory.
#
function kube::codegen::gen_client() {
    local in_dir=""
    local one_input_api=""
    local out_dir=""
    local out_pkg=""
    local clientset_subdir="clientset"
    local clientset_versioned_name="versioned"
    local applyconfig="false"
    local applyconfig_subdir="applyconfiguration"
    local watchable="false"
    local listers_subdir="listers"
    local informers_subdir="informers"
    local boilerplate="${KUBE_CODEGEN_ROOT}/hack/boilerplate.go.txt"
    local v="${KUBE_VERBOSE:-0}"

    while [ "$#" -gt 0 ]; do
        case "$1" in
            "--one-input-api")
                one_input_api="/$2"
                shift 2
                ;;
            "--output-dir")
                out_dir="$2"
                shift 2
                ;;
            "--output-pkg")
                out_pkg="$2"
                shift 2
                ;;
            "--boilerplate")
                boilerplate="$2"
                shift 2
                ;;
            "--clientset-name")
                clientset_subdir="$2"
                shift 2
                ;;
            "--versioned-name")
                clientset_versioned_name="$2"
                shift 2
                ;;
            "--with-applyconfig")
                applyconfig="true"
                shift
                ;;
            "--applyconfig-name")
                applyconfig_subdir="$2"
                shift 2
                ;;
            "--with-watch")
                watchable="true"
                shift
                ;;
            "--listers-name")
                listers_subdir="$2"
                shift 2
                ;;
            "--informers-name")
                informers_subdir="$2"
                shift 2
                ;;
            *)
                if [[ "$1" =~ ^-- ]]; then
                    echo "unknown argument: $1" >&2
                    return 1
                fi
                if [ -n "$in_dir" ]; then
                    echo "too many arguments: $1 (already have $in_dir)" >&2
                    return 1
                fi
                in_dir="$1"
                shift
                ;;
        esac
    done

    if [ -z "${in_dir}" ]; then
        echo "input-dir argument is required" >&2
        return 1
    fi
    if [ -z "${out_dir}" ]; then
        echo "--output-dir is required" >&2
        return 1
    fi
    if [ -z "${out_pkg}" ]; then
        echo "--output-pkg-root is required" >&2
    fi

    mkdir -p "${out_dir}"

    (
        # To support running this from anywhere, first cd into this directory,
        # and then install with forced module mode on and fully qualified name.
        cd "${KUBE_CODEGEN_ROOT}"
        BINS=(
            applyconfiguration-gen
            client-gen
            informer-gen
            lister-gen
        )
        # shellcheck disable=2046 # printf word-splitting is intentional
        GO111MODULE=on go install $(printf "k8s.io/code-generator/cmd/%s " "${BINS[@]}")
    )
    # Go installs in $GOBIN if defined, and $GOPATH/bin otherwise
    gobin="${GOBIN:-$(go env GOPATH)/bin}"

    local group_versions=()
    local input_pkgs=()
    while read -r dir; do
        pkg="$(cd "${dir}" && GO111MODULE=on go list -find .)"
        leaf="$(basename "${dir}")"
        if grep -E -q '^v[0-9]+((alpha|beta)[0-9]+)?$' <<< "${leaf}"; then
            input_pkgs+=("${pkg}")

            dir2="$(dirname "${dir}")"
            leaf2="$(basename "${dir2}")"
            group_versions+=("${leaf2}/${leaf}")
        fi
    done < <(
        ( kube::codegen::internal::git_grep -l --null \
            -e '+genclient' \
            ":(glob)${in_dir}${one_input_api}"/'**/*.go' \
            || true \
        ) | while read -r -d $'\0' F; do dirname $F; done \
          | LC_ALL=C sort -u
    )

    if [ "${#group_versions[@]}" == 0 ]; then
        return 0
    fi

    applyconfig_pkg="" # set this for later use, iff enabled
    if [ "${applyconfig}" == "true" ]; then
        applyconfig_pkg="${out_pkg}/${applyconfig_subdir}"

        echo "Generating applyconfig code for ${#input_pkgs[@]} targets"

        ( kube::codegen::internal::git_grep -l --null \
            -e '^// Code generated by applyconfiguration-gen. DO NOT EDIT.$' \
            ":(glob)${out_dir}/${applyconfig_subdir}"/'**/*.go' \
            || true \
        ) | xargs -0 rm -f

        local inputs=()
        for arg in "${input_pkgs[@]}"; do
            inputs+=("--input-dirs" "$arg")
        done
        "${gobin}/applyconfiguration-gen" \
            -v "${v}" \
            --go-header-file "${boilerplate}" \
            --output-base "${out_dir}/${applyconfig_subdir}" \
            --output-package "${applyconfig_pkg}" \
            "${inputs[@]}"
    fi

    echo "Generating client code for ${#group_versions[@]} targets"

    ( kube::codegen::internal::git_grep -l --null \
        -e '^// Code generated by client-gen. DO NOT EDIT.$' \
        ":(glob)${out_dir}/${clientset_subdir}"/'**/*.go' \
        || true \
    ) | xargs -0 rm -f

    local inputs=()
    for arg in "${group_versions[@]}"; do
        inputs+=("--input" "$arg")
    done
     "${gobin}/client-gen" \
        -v "${v}" \
        --go-header-file "${boilerplate}" \
        --output-base "${out_dir}/${clientset_subdir}" \
        --output-package "${out_pkg}/${clientset_subdir}" \
        --clientset-name "${clientset_versioned_name}" \
        --apply-configuration-package "${applyconfig_pkg}" \
        --input-base "$(cd $in_dir; pwd -P)" `# must be absolute path or Go import path"` \
        "${inputs[@]}"

    if [ "${watchable}" == "true" ]; then
        echo "Generating lister code for ${#input_pkgs[@]} targets"

        ( kube::codegen::internal::git_grep -l --null \
            -e '^// Code generated by lister-gen. DO NOT EDIT.$' \
            ":(glob)${out_dir}/${listers_subdir}"/'**/*.go' \
            || true \
        ) | xargs -0 rm -f

        local inputs=()
        for arg in "${input_pkgs[@]}"; do
            inputs+=("--input-dirs" "$arg")
        done
        "${gobin}/lister-gen" \
            -v "${v}" \
            --go-header-file "${boilerplate}" \
            --output-base "${out_dir}/${listers_subdir}" \
            "${inputs[@]}"

        echo "Generating informer code for ${#input_pkgs[@]} targets"

        ( kube::codegen::internal::git_grep -l --null \
            -e '^// Code generated by informer-gen. DO NOT EDIT.$' \
            ":(glob)${out_dir}/${informers_subdir}"/'**/*.go' \
            || true \
        ) | xargs -0 rm -f

        local inputs=()
        for arg in "${input_pkgs[@]}"; do
            inputs+=("--input-dirs" "$arg")
        done
        "${gobin}/informer-gen" \
            -v "${v}" \
            --go-header-file "${boilerplate}" \
            --output-base "${out_dir}/${informers_subdir}" \
            --output-package "${out_pkg}/${informers_subdir}" \
            --versioned-clientset-package "${out_pkg}/${clientset_subdir}/${clientset_versioned_name}" \
            --listers-package "${out_pkg}/${listers_subdir}" \
            "${inputs[@]}"
    fi
}
