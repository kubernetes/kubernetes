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
# Args:
#   --input-pkg-root <string>
#     The root package under which to search for files which request code to be
#     generated.  This must be Go package syntax, e.g.  "k8s.io/foo/bar".
#
#   --output-base <string>
#     The root directory under which to emit code.  The concatenation of
#     <output-base> + <input-pkg-root> must be valid.
#
#   --boilerplate <string = path_to_kube_codegen_boilerplate>
#     An optional override for the header file to insert into generated files.
#
function kube::codegen::gen_helpers() {
    local in_pkg_root=""
    local out_base="" # gengo needs the output dir must be $out_base/$out_pkg_root
    local boilerplate="${KUBE_CODEGEN_ROOT}/hack/boilerplate.go.txt"
    local v="${KUBE_VERBOSE:-0}"

    while [ "$#" -gt 0 ]; do
        case "$1" in
            "--input-pkg-root")
                in_pkg_root="$2"
                shift 2
                ;;
            "--output-base")
                out_base="$2"
                shift 2
                ;;
            "--boilerplate")
                boilerplate="$2"
                shift 2
                ;;
            *)
                echo "unknown argument: $1" >&2
                return 1
                ;;
        esac
    done

    if [ -z "${in_pkg_root}" ]; then
        echo "--input-pkg-root is required" >&2
        return 1
    fi
    if [ -z "${out_base}" ]; then
        echo "--output-base is required" >&2
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

    # These tools all assume out-dir == in-dir.
    root="${out_base}/${in_pkg_root}"
    mkdir -p "${root}"
    root="$(cd "${root}" && pwd -P)"

    # Deepcopy
    #
    local input_pkgs=()
    while read -r file; do
        dir="$(dirname "${file}")"
        pkg="$(cd "${dir}" && GO111MODULE=on go list -find .)"
        input_pkgs+=("${pkg}")
    done < <(
        ( kube::codegen::internal::git_grep -l \
            -e '+k8s:deepcopy-gen=' \
            ":(glob)${root}"/'**/*.go' \
            || true \
        ) | LC_ALL=C sort -u
    )

    if [ "${#input_pkgs[@]}" != 0 ]; then
        echo "Generating deepcopy code for ${#input_pkgs[@]} targets"

        kube::codegen::internal::git_find -z \
            ":(glob)${root}"/'**/zz_generated.deepcopy.go' \
            | xargs -0 rm -f

        local inputs=()
        for arg in "${input_pkgs[@]}"; do
            inputs+=("--input-dirs" "$arg")
        done
        "${gobin}/deepcopy-gen" \
            -v "${v}" \
            -O zz_generated.deepcopy \
            --go-header-file "${boilerplate}" \
            --output-base "${out_base}" \
            "${inputs[@]}"
    fi

    # Defaults
    #
    local input_pkgs=()
    while read -r file; do
        dir="$(dirname "${file}")"
        pkg="$(cd "${dir}" && GO111MODULE=on go list -find .)"
        input_pkgs+=("${pkg}")
    done < <(
        ( kube::codegen::internal::git_grep -l \
            -e '+k8s:defaulter-gen=' \
            ":(glob)${root}"/'**/*.go' \
            || true \
        ) | LC_ALL=C sort -u
    )

    if [ "${#input_pkgs[@]}" != 0 ]; then
        echo "Generating defaulter code for ${#input_pkgs[@]} targets"

        kube::codegen::internal::git_find -z \
            ":(glob)${root}"/'**/zz_generated.defaults.go' \
            | xargs -0 rm -f

        local inputs=()
        for arg in "${input_pkgs[@]}"; do
            inputs+=("--input-dirs" "$arg")
        done
        "${gobin}/defaulter-gen" \
            -v "${v}" \
            -O zz_generated.defaults \
            --go-header-file "${boilerplate}" \
            --output-base "${out_base}" \
            "${inputs[@]}"
    fi

    # Conversions
    #
    local input_pkgs=()
    while read -r file; do
        dir="$(dirname "${file}")"
        pkg="$(cd "${dir}" && GO111MODULE=on go list -find .)"
        input_pkgs+=("${pkg}")
    done < <(
        ( kube::codegen::internal::git_grep -l \
            -e '+k8s:conversion-gen=' \
            ":(glob)${root}"/'**/*.go' \
            || true \
        ) | LC_ALL=C sort -u
    )

    if [ "${#input_pkgs[@]}" != 0 ]; then
        echo "Generating conversion code for ${#input_pkgs[@]} targets"

        kube::codegen::internal::git_find -z \
            ":(glob)${root}"/'**/zz_generated.conversion.go' \
            | xargs -0 rm -f

        local inputs=()
        for arg in "${input_pkgs[@]}"; do
            inputs+=("--input-dirs" "$arg")
        done
        "${gobin}/conversion-gen" \
            -v "${v}" \
            -O zz_generated.conversion \
            --go-header-file "${boilerplate}" \
            --output-base "${out_base}" \
            "${inputs[@]}"
    fi
}

# Generate openapi code
#
# Args:
#   --input-pkg-root <string>
#     The root package under which to search for files which request openapi to
#     be generated.  This must be Go package syntax, e.g.  "k8s.io/foo/bar".
#
#   --output-pkg-root <string>
#     The root package under which generated directories and files
#     will be placed.  This must be go package syntax, e.g. "k8s.io/foo/bar".
#
#   --output-base <string>
#     The root directory under which to emit code.  The concatenation of
#     <output-base> + <input-pkg-root> must be valid.
#
#   --openapi-name <string = "openapi">
#     An optional override for the leaf name of the generated directory.
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
    local in_pkg_root=""
    local out_pkg_root=""
    local out_base="" # gengo needs the output dir must be $out_base/$out_pkg_root
    local openapi_subdir="openapi"
    local extra_pkgs=()
    local report="/dev/null"
    local update_report=""
    local boilerplate="${KUBE_CODEGEN_ROOT}/hack/boilerplate.go.txt"
    local v="${KUBE_VERBOSE:-0}"

    while [ "$#" -gt 0 ]; do
        case "$1" in
            "--input-pkg-root")
                in_pkg_root="$2"
                shift 2
                ;;
            "--output-pkg-root")
                out_pkg_root="$2"
                shift 2
                ;;
            "--output-base")
                out_base="$2"
                shift 2
                ;;
            "--openapi-name")
                openapi_subdir="$2"
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
                echo "unknown argument: $1" >&2
                return 1
                ;;
        esac
    done

    if [ -z "${in_pkg_root}" ]; then
        echo "--input-pkg-root is required" >&2
        return 1
    fi
    if [ -z "${out_pkg_root}" ]; then
        echo "--output-pkg-root is required" >&2
        return 1
    fi
    if [ -z "${out_base}" ]; then
        echo "--output-base is required" >&2
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

    # These tools all assume out-dir == in-dir.
    root="${out_base}/${in_pkg_root}"
    mkdir -p "${root}"
    root="$(cd "${root}" && pwd -P)"

    local input_pkgs=( "${extra_pkgs[@]:+"${extra_pkgs[@]}"}")
    while read -r file; do
        dir="$(dirname "${file}")"
        pkg="$(cd "${dir}" && GO111MODULE=on go list -find .)"
        input_pkgs+=("${pkg}")
    done < <(
        ( kube::codegen::internal::git_grep -l \
            -e '+k8s:openapi-gen=' \
            ":(glob)${root}"/'**/*.go' \
            || true \
        ) | LC_ALL=C sort -u
    )

    if [ "${#input_pkgs[@]}" != 0 ]; then
        echo "Generating openapi code for ${#input_pkgs[@]} targets"

        kube::codegen::internal::git_find -z \
            ":(glob)${root}"/'**/zz_generated.openapi.go' \
            | xargs -0 rm -f

        local inputs=()
        for arg in "${input_pkgs[@]}"; do
            inputs+=("--input-dirs" "$arg")
        done
        "${gobin}/openapi-gen" \
            -v "${v}" \
            -O zz_generated.openapi \
            --go-header-file "${boilerplate}" \
            --output-base "${out_base}" \
            --output-package "${out_pkg_root}/${openapi_subdir}" \
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
# Args:
#   --input-pkg-root <string>
#     The root package under which to search for types.go files which request
#     clients to be generated.  This must be Go package syntax, e.g.
#     "k8s.io/foo/bar".
#
#   --output-pkg-root <string>
#     The root package into which generated directories and files will be
#     placed.  This must be Go package syntax, e.g. "k8s.io/foo/bar".
#
#   --output-base <string>
#     The root directory under which to emit code.  The concatenation of
#     <output-base> + <output-pkg-root> must be valid.
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
    local in_pkg_root=""
    local out_pkg_root=""
    local out_base="" # gengo needs the output dir must be $out_base/$out_pkg_root
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
            "--input-pkg-root")
                in_pkg_root="$2"
                shift 2
                ;;
            "--output-pkg-root")
                out_pkg_root="$2"
                shift 2
                ;;
            "--output-base")
                out_base="$2"
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
                echo "unknown argument: $1" >&2
                return 1
                ;;
        esac
    done

    if [ -z "${in_pkg_root}" ]; then
        echo "--input-pkg-root is required" >&2
        return 1
    fi
    if [ -z "${out_pkg_root}" ]; then
        echo "--output-pkg-root is required" >&2
        return 1
    fi
    if [ -z "${out_base}" ]; then
        echo "--output-base is required" >&2
        return 1
    fi

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

    in_root="${out_base}/${in_pkg_root}"
    mkdir -p "${in_root}"
    in_root="$(cd "${in_root}" && pwd -P)"
    out_root="${out_base}/${out_pkg_root}"
    mkdir -p "${out_root}"
    out_root="$(cd "${out_root}" && pwd -P)"

    local group_versions=()
    local input_pkgs=()
    while read -r file; do
        dir="$(dirname "${file}")"
        pkg="$(cd "${dir}" && GO111MODULE=on go list -find .)"
        leaf="$(basename "${dir}")"
        if grep -E -q '^v[0-9]+((alpha|beta)[0-9]+)?$' <<< "${leaf}"; then
            input_pkgs+=("${pkg}")

            dir2="$(dirname "${dir}")"
            leaf2="$(basename "${dir2}")"
            group_versions+=("${leaf2}/${leaf}")
        fi
    done < <(
        ( kube::codegen::internal::git_grep -l \
            -e '+genclient' \
            ":(glob)${in_root}"/'**/types.go' \
            || true \
        ) | LC_ALL=C sort -u
    )

    if [ "${#group_versions[@]}" == 0 ]; then
        return 0
    fi

    applyconfig_pkg="" # set this for later use, iff enabled
    if [ "${applyconfig}" == "true" ]; then
        applyconfig_pkg="${out_pkg_root}/${applyconfig_subdir}"

        echo "Generating applyconfig code for ${#input_pkgs[@]} targets"

        ( kube::codegen::internal::git_grep -l --null \
            -e '^// Code generated by applyconfiguration-gen. DO NOT EDIT.$' \
            ":(glob)${out_root}/${applyconfig_subdir}"/'**/*.go' \
            || true \
        ) | xargs -0 rm -f

        local inputs=()
        for arg in "${input_pkgs[@]}"; do
            inputs+=("--input-dirs" "$arg")
        done
        "${gobin}/applyconfiguration-gen" \
            -v "${v}" \
            --go-header-file "${boilerplate}" \
            --output-base "${out_base}" \
            --output-package "${out_pkg_root}/${applyconfig_subdir}" \
            "${inputs[@]}"
    fi

    echo "Generating client code for ${#group_versions[@]} targets"

    ( kube::codegen::internal::git_grep -l --null \
        -e '^// Code generated by client-gen. DO NOT EDIT.$' \
        ":(glob)${out_root}/${clientset_subdir}"/'**/*.go' \
        || true \
    ) | xargs -0 rm -f

    local inputs=()
    for arg in "${group_versions[@]}"; do
        inputs+=("--input" "$arg")
    done
    "${gobin}/client-gen" \
        -v "${v}" \
        --go-header-file "${boilerplate}" \
        --clientset-name "${clientset_versioned_name}" \
        --input-base "${in_pkg_root}" \
        --output-base "${out_base}" \
        --output-package "${out_pkg_root}/${clientset_subdir}" \
        --apply-configuration-package "${applyconfig_pkg}" \
        "${inputs[@]}"

    if [ "${watchable}" == "true" ]; then
        echo "Generating lister code for ${#input_pkgs[@]} targets"

        ( kube::codegen::internal::git_grep -l --null \
            -e '^// Code generated by lister-gen. DO NOT EDIT.$' \
            ":(glob)${out_root}/${listers_subdir}"/'**/*.go' \
            || true \
        ) | xargs -0 rm -f

        local inputs=()
        for arg in "${input_pkgs[@]}"; do
            inputs+=("--input-dirs" "$arg")
        done
        "${gobin}/lister-gen" \
            -v "${v}" \
            --go-header-file "${boilerplate}" \
            --output-base "${out_base}" \
            --output-package "${out_pkg_root}/${listers_subdir}" \
            "${inputs[@]}"

        echo "Generating informer code for ${#input_pkgs[@]} targets"

        ( kube::codegen::internal::git_grep -l --null \
            -e '^// Code generated by informer-gen. DO NOT EDIT.$' \
            ":(glob)${out_root}/${informers_subdir}"/'**/*.go' \
            || true \
        ) | xargs -0 rm -f

        local inputs=()
        for arg in "${input_pkgs[@]}"; do
            inputs+=("--input-dirs" "$arg")
        done
        "${gobin}/informer-gen" \
            -v "${v}" \
            --go-header-file "${boilerplate}" \
            --output-base "${out_base}" \
            --output-package "${out_pkg_root}/${informers_subdir}" \
            --versioned-clientset-package "${out_pkg_root}/${clientset_subdir}/${clientset_versioned_name}" \
            --listers-package "${out_pkg_root}/${listers_subdir}" \
            "${inputs[@]}"
    fi
}
