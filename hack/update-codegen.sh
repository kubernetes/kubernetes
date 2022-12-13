#!/usr/bin/env bash
# Copyright 2014 The Kubernetes Authors.
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

# shellcheck disable=2046 # printf word-splitting is intentional

set -o errexit
set -o nounset
set -o pipefail

# This tool wants a different default than usual.
KUBE_VERBOSE="${KUBE_VERBOSE:-1}"

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"
cd "${KUBE_ROOT}"

kube::golang::setup_env

DBG_CODEGEN="${DBG_CODEGEN:-0}"
GENERATED_FILE_PREFIX="${GENERATED_FILE_PREFIX:-zz_generated.}"
UPDATE_API_KNOWN_VIOLATIONS="${UPDATE_API_KNOWN_VIOLATIONS:-}"

OUT_DIR="_output"
BIN_DIR="${OUT_DIR}/bin"
PRJ_SRC_PATH="k8s.io/kubernetes"
BOILERPLATE_FILENAME="vendor/k8s.io/code-generator/hack/boilerplate.go.txt"
APPLYCONFIG_PKG="k8s.io/client-go/applyconfigurations"

if [[ "${DBG_CODEGEN}" == 1 ]]; then
    kube::log::status "DBG: starting generated_files"
fi

# This is a partial 'find' command.  The caller is expected to pass the
# remaining arguments.
#
# Example:
#   kfind -type f -name foobar.go
function kfind() {
    # We want to include the "special" vendor directories which are actually
    # part of the Kubernetes source tree (./staging/*) but we need them to be
    # named as their ./vendor/* equivalents.  Also, we do not want all of
    # ./vendor nor ./hack/tools/vendor nor even all of ./vendor/k8s.io.
    find -H .                      \
        \(                         \
        -not \(                    \
            \(                     \
                -name '_*' -o      \
                -name '.[^.]*' -o  \
                \(                 \
                  -name 'vendor'   \
                  -type d          \
                \) -o              \
                \(                 \
                  -name 'testdata' \
                  -type d          \
                \)                 \
            \) -prune              \
        \)                         \
        \)                         \
        "$@"                       \
        | sed 's|^./staging/src|vendor|'
}

function find_all_go_dirs() {
    kfind -type f -name \*.go  \
        | sed 's|/[^/]*$||'    \
        | sed 's|^./||'        \
        | LC_ALL=C sort -u
}

# This variable holds a list of every directory that contains Go files in this
# project.  Other rules and variables can use this as a starting point to
# reduce filesystem accesses.
if [[ "${DBG_CODEGEN}" == 1 ]]; then
    kube::log::status "DBG: finding all *.go dirs"
fi
ALL_GO_DIRS=()
kube::util::read-array ALL_GO_DIRS < <(find_all_go_dirs)
if [[ "${DBG_CODEGEN}" == 1 ]]; then
    kube::log::status "DBG: found ${#ALL_GO_DIRS[@]} *.go dirs"
fi

# Generate a list of all files that have a `+k8s:` comment-tag.  This will be
# used to derive lists of files/dirs for generation tools.
if [[ "${DBG_CODEGEN}" == 1 ]]; then
    kube::log::status "DBG: finding all +k8s: tags"
fi
ALL_K8S_TAG_FILES=()
kube::util::read-array ALL_K8S_TAG_FILES < <(
    find "${ALL_GO_DIRS[@]}" -maxdepth 1 -type f -name \*.go -print0 \
        | xargs -0 grep --color=never -l '^// *+k8s:')
if [[ "${DBG_CODEGEN}" == 1 ]]; then
    kube::log::status "DBG: found ${#ALL_K8S_TAG_FILES[@]} +k8s: tagged files"
fi

#
# Code generation logic.
#

# prerelease-lifecycle generation
#
# Any package that wants prerelease-lifecycle functions generated must include a
# comment-tag in column 0 of one file of the form:
#     // +k8s:prerelease-lifecycle-gen=true
function codegen::prerelease() {
    # Build the tool.
    hack/make-rules/build.sh k8s.io/code-generator/cmd/prerelease-lifecycle-gen

    # The result file, in each pkg, of prerelease-lifecycle generation.
    local output_base="${GENERATED_FILE_PREFIX}prerelease-lifecycle"

    # The tool used to generate prerelease-lifecycle code.
    local gen_prerelease_bin="${BIN_DIR}/prerelease-lifecycle-gen"

    # Find all the directories that request prerelease-lifecycle generation.
    if [[ "${DBG_CODEGEN}" == 1 ]]; then
        kube::log::status "DBG: finding all +k8s:prerelease-lifecycle-gen tags"
    fi
    local tag_dirs=()
    kube::util::read-array tag_dirs < <( \
        grep --color=never -l '+k8s:prerelease-lifecycle-gen=true' "${ALL_K8S_TAG_FILES[@]}" \
            | xargs -n1 dirname \
            | LC_ALL=C sort -u)
    if [[ "${DBG_CODEGEN}" == 1 ]]; then
        kube::log::status "DBG: found ${#tag_dirs[@]} +k8s:prerelease-lifecycle-gen tagged dirs"
    fi

    local tag_pkgs=()
    for dir in "${tag_dirs[@]}"; do
        tag_pkgs+=("${PRJ_SRC_PATH}/$dir")
    done

    kube::log::status "Generating prerelease-lifecycle code for ${#tag_pkgs[@]} targets"
    if [[ "${DBG_CODEGEN}" == 1 ]]; then
        kube::log::status "DBG: running ${gen_prerelease_bin} for:"
        for dir in "${tag_dirs[@]}"; do
            kube::log::status "DBG:     $dir"
        done
    fi

    ./hack/run-in-gopath.sh "${gen_prerelease_bin}" \
        --v "${KUBE_VERBOSE}" \
        --logtostderr \
        -h "${BOILERPLATE_FILENAME}" \
        -O "${output_base}" \
        $(printf -- " -i %s" "${tag_pkgs[@]}") \
        "$@"

    if [[ "${DBG_CODEGEN}" == 1 ]]; then
        kube::log::status "Generated prerelease-lifecycle code"
    fi
}

# Deep-copy generation
#
# Any package that wants deep-copy functions generated must include a
# comment-tag in column 0 of one file of the form:
#     // +k8s:deepcopy-gen=<VALUE>
#
# The <VALUE> may be one of:
#     generate: generate deep-copy functions into the package
#     register: generate deep-copy functions and register them with a
#               scheme
function codegen::deepcopy() {
    # Build the tool.
    hack/make-rules/build.sh k8s.io/code-generator/cmd/deepcopy-gen

    # The result file, in each pkg, of deep-copy generation.
    local output_base="${GENERATED_FILE_PREFIX}deepcopy"

    # The tool used to generate deep copies.
    local gen_deepcopy_bin="${BIN_DIR}/deepcopy-gen"

    # Find all the directories that request deep-copy generation.
    if [[ "${DBG_CODEGEN}" == 1 ]]; then
        kube::log::status "DBG: finding all +k8s:deepcopy-gen tags"
    fi
    local tag_dirs=()
    kube::util::read-array tag_dirs < <( \
        grep --color=never -l '+k8s:deepcopy-gen=' "${ALL_K8S_TAG_FILES[@]}" \
            | xargs -n1 dirname \
            | LC_ALL=C sort -u)
    if [[ "${DBG_CODEGEN}" == 1 ]]; then
        kube::log::status "DBG: found ${#tag_dirs[@]} +k8s:deepcopy-gen tagged dirs"
    fi

    local tag_pkgs=()
    for dir in "${tag_dirs[@]}"; do
        tag_pkgs+=("${PRJ_SRC_PATH}/$dir")
    done

    kube::log::status "Generating deepcopy code for ${#tag_pkgs[@]} targets"
    if [[ "${DBG_CODEGEN}" == 1 ]]; then
        kube::log::status "DBG: running ${gen_deepcopy_bin} for:"
        for dir in "${tag_dirs[@]}"; do
            kube::log::status "DBG:     $dir"
        done
    fi

    ./hack/run-in-gopath.sh "${gen_deepcopy_bin}" \
        --v "${KUBE_VERBOSE}" \
        --logtostderr \
        -h "${BOILERPLATE_FILENAME}" \
        -O "${output_base}" \
        --bounding-dirs "${PRJ_SRC_PATH},k8s.io/api" \
        $(printf -- " -i %s" "${tag_pkgs[@]}") \
        "$@"

    if [[ "${DBG_CODEGEN}" == 1 ]]; then
        kube::log::status "Generated deepcopy code"
    fi
}

# Defaulter generation
#
# Any package that wants defaulter functions generated must include a
# comment-tag in column 0 of one file of the form:
#     // +k8s:defaulter-gen=<VALUE>
#
# The <VALUE> depends on context:
#     on types:
#       true:  always generate a defaulter for this type
#       false: never generate a defaulter for this type
#     on functions:
#       covers: if the function name matches SetDefault_NAME, instructs
#               the generator not to recurse
#     on packages:
#       FIELDNAME: any object with a field of this name is a candidate
#                  for having a defaulter generated
function codegen::defaults() {
    # Build the tool.
    hack/make-rules/build.sh k8s.io/code-generator/cmd/defaulter-gen

    # The result file, in each pkg, of defaulter generation.
    local output_base="${GENERATED_FILE_PREFIX}defaults"

    # The tool used to generate defaulters.
    local gen_defaulter_bin="${BIN_DIR}/defaulter-gen"

    # All directories that request any form of defaulter generation.
    if [[ "${DBG_CODEGEN}" == 1 ]]; then
        kube::log::status "DBG: finding all +k8s:defaulter-gen tags"
    fi
    local tag_dirs=()
    kube::util::read-array tag_dirs < <( \
        grep --color=never -l '+k8s:defaulter-gen=' "${ALL_K8S_TAG_FILES[@]}" \
            | xargs -n1 dirname \
            | LC_ALL=C sort -u)
    if [[ "${DBG_CODEGEN}" == 1 ]]; then
        kube::log::status "DBG: found ${#tag_dirs[@]} +k8s:defaulter-gen tagged dirs"
    fi

    local tag_pkgs=()
    for dir in "${tag_dirs[@]}"; do
        tag_pkgs+=("${PRJ_SRC_PATH}/$dir")
    done

    kube::log::status "Generating defaulter code for ${#tag_pkgs[@]} targets"
    if [[ "${DBG_CODEGEN}" == 1 ]]; then
        kube::log::status "DBG: running ${gen_defaulter_bin} for:"
        for dir in "${tag_dirs[@]}"; do
            kube::log::status "DBG:     $dir"
        done
    fi

    ./hack/run-in-gopath.sh "${gen_defaulter_bin}" \
        --v "${KUBE_VERBOSE}" \
        --logtostderr \
        -h "${BOILERPLATE_FILENAME}" \
        -O "${output_base}" \
        $(printf -- " --extra-peer-dirs %s" "${tag_pkgs[@]}") \
        $(printf -- " -i %s" "${tag_pkgs[@]}") \
        "$@"

    if [[ "${DBG_CODEGEN}" == 1 ]]; then
        kube::log::status "Generated defaulter code"
    fi
}

# Conversion generation

# Any package that wants conversion functions generated into it must
# include one or more comment-tags in its `doc.go` file, of the form:
#     // +k8s:conversion-gen=<INTERNAL_TYPES_DIR>
#
# The INTERNAL_TYPES_DIR is a project-local path to another directory
# which should be considered when evaluating peer types for
# conversions.  An optional additional comment of the form
#     // +k8s:conversion-gen-external-types=<EXTERNAL_TYPES_DIR>
#
# identifies where to find the external types; if there is no such
# comment then the external types are sought in the package where the
# `k8s:conversion` tag is found.
#
# Conversions, in both directions, are generated for every type name
# that is defined in both an internal types package and the external
# types package.
#
# TODO: it might be better in the long term to make peer-types explicit in the
# IDL.
function codegen::conversions() {
    # Build the tool.
    hack/make-rules/build.sh k8s.io/code-generator/cmd/conversion-gen

    # The result file, in each pkg, of conversion generation.
    local output_base="${GENERATED_FILE_PREFIX}conversion"

    # The tool used to generate conversions.
    local gen_conversion_bin="${BIN_DIR}/conversion-gen"

    # All directories that request any form of conversion generation.
    if [[ "${DBG_CODEGEN}" == 1 ]]; then
        kube::log::status "DBG: finding all +k8s:conversion-gen tags"
    fi
    local tag_dirs=()
    kube::util::read-array tag_dirs < <(\
        grep --color=never -l '^// *+k8s:conversion-gen=' "${ALL_K8S_TAG_FILES[@]}" \
            | xargs -n1 dirname \
            | LC_ALL=C sort -u)
    if [[ "${DBG_CODEGEN}" == 1 ]]; then
        kube::log::status "DBG: found ${#tag_dirs[@]} +k8s:conversion-gen tagged dirs"
    fi

    local tag_pkgs=()
    for dir in "${tag_dirs[@]}"; do
        tag_pkgs+=("${PRJ_SRC_PATH}/$dir")
    done

    local extra_peer_pkgs=(
        k8s.io/kubernetes/pkg/apis/core
        k8s.io/kubernetes/pkg/apis/core/v1
        k8s.io/api/core/v1
    )

    kube::log::status "Generating conversion code for ${#tag_pkgs[@]} targets"
    if [[ "${DBG_CODEGEN}" == 1 ]]; then
        kube::log::status "DBG: running ${gen_conversion_bin} for:"
        for dir in "${tag_dirs[@]}"; do
            kube::log::status "DBG:     $dir"
        done
    fi

    ./hack/run-in-gopath.sh "${gen_conversion_bin}" \
        --v "${KUBE_VERBOSE}" \
        --logtostderr \
        -h "${BOILERPLATE_FILENAME}" \
        -O "${output_base}" \
        $(printf -- " --extra-peer-dirs %s" "${extra_peer_pkgs[@]}") \
        $(printf -- " --extra-dirs %s" "${tag_pkgs[@]}") \
        $(printf -- " -i %s" "${tag_pkgs[@]}") \
        "$@"

    if [[ "${DBG_CODEGEN}" == 1 ]]; then
        kube::log::status "Generated conversion code"
    fi
}

# $@: directories to exclude
# example:
#    k8s_tag_files_except foo bat/qux
function k8s_tag_files_except() {
    for f in "${ALL_K8S_TAG_FILES[@]}"; do
        local excl=""
        for x in "$@"; do
            if [[ "$f" =~ "$x"/.* ]]; then
                excl="true"
                break
            fi
        done
        if [[ "${excl}" != true ]]; then
            echo "$f"
        fi
    done
}

# $@: directories to exclude
# example:
#    k8s_tag_files_matching foo bat/qux
function k8s_tag_files_matching() {
    for f in "${ALL_K8S_TAG_FILES[@]}"; do
        for x in "$@"; do
            if [[ "$f" =~ "${x}"/.* ]]; then
                echo "$f"
                break
            fi
        done
    done
}

# $1: the name of a scalar variable to read
# example:
#    FOO_VAR="foo value"
#    BAR_VAR="bar value"
#    x=FOO
#    indirect "${x}_VAR" # -> "foo value\n"
function indirect() {
    # This is a trick to get bash to indirectly read a variable.
    # Thanks StackOverflow!
    local var="$1"
    echo "${!var}"
}

# $1: the name of an array variable to read
#    FOO_ARR=(a b c)
#    BAR_ARR=(1 2 3)
#    x=FOO
#    indirect_array "${x}_ARR" # -> "a\nb\nc\n"
function indirect_array() {
    # This is a trick to get bash to indirectly read an array.
    # Thanks StackOverflow!
    local arrayname="$1"
    # shellcheck disable=SC1087 # intentional
    local tmp="$arrayname[@]"
    printf -- "%s\n" "${!tmp}"
}

# OpenAPI generation
#
# Any package that wants open-api functions generated must include a
# comment-tag in column 0 of one file of the form:
#     // +k8s:openapi-gen=true
function codegen::openapi() {
    # Build the tool.
    hack/make-rules/build.sh k8s.io/kube-openapi/cmd/openapi-gen

    # The result file, in each pkg, of open-api generation.
    local output_base="${GENERATED_FILE_PREFIX}openapi"

    # The tool used to generate open apis.
    local gen_openapi_bin="${BIN_DIR}/openapi-gen"

    # Standard dirs which all targets need.
    local apimachinery_dirs=(
        vendor/k8s.io/apimachinery/pkg/apis/meta/v1
        vendor/k8s.io/apimachinery/pkg/runtime
        vendor/k8s.io/apimachinery/pkg/version
    )

    # These should probably be configured by tags in code-files somewhere.
    local targets=(
        KUBE
        AGGREGATOR
        APIEXTENSIONS
        CODEGEN
        SAMPLEAPISERVER
    )

    # shellcheck disable=SC2034 # used indirectly
    local KUBE_output_dir="pkg/generated/openapi"
    # shellcheck disable=SC2034 # used indirectly
    local KUBE_known_violations_file="api/api-rules/violation_exceptions.list"
    # shellcheck disable=SC2034 # used indirectly
    local KUBE_tag_files=()
    kube::util::read-array KUBE_tag_files < <(
        k8s_tag_files_except \
            vendor/k8s.io/code-generator \
            vendor/k8s.io/sample-apiserver
        )

    # shellcheck disable=SC2034 # used indirectly
    local AGGREGATOR_output_dir="staging/src/k8s.io/kube-aggregator/pkg/generated/openapi"
    # shellcheck disable=SC2034 # used indirectly
    local AGGREGATOR_known_violations_file="api/api-rules/aggregator_violation_exceptions.list"
    # shellcheck disable=SC2034 # used indirectly
    local AGGREGATOR_tag_files=()
    kube::util::read-array AGGREGATOR_tag_files < <(
        k8s_tag_files_matching \
            vendor/k8s.io/kube-aggregator \
            "${apimachinery_dirs[@]}"
        )

    # shellcheck disable=SC2034 # used indirectly
    local APIEXTENSIONS_output_dir="staging/src/k8s.io/apiextensions-apiserver/pkg/generated/openapi"
    # shellcheck disable=SC2034 # used indirectly
    local APIEXTENSIONS_known_violations_file="api/api-rules/apiextensions_violation_exceptions.list"
    # shellcheck disable=SC2034 # used indirectly
    local APIEXTENSIONS_tag_files=()
    kube::util::read-array APIEXTENSIONS_tag_files < <(
        k8s_tag_files_matching \
            vendor/k8s.io/apiextensions \
            vendor/k8s.io/api/autoscaling/v1 \
            "${apimachinery_dirs[@]}"
        )

    # shellcheck disable=SC2034 # used indirectly
    local CODEGEN_output_dir="staging/src/k8s.io/code-generator/examples/apiserver/openapi"
    # shellcheck disable=SC2034 # used indirectly
    local CODEGEN_known_violations_file="api/api-rules/codegen_violation_exceptions.list"
    # shellcheck disable=SC2034 # used indirectly
    local CODEGEN_tag_files=()
    kube::util::read-array CODEGEN_tag_files < <(
        k8s_tag_files_matching \
            vendor/k8s.io/code-generator \
            "${apimachinery_dirs[@]}"
        )

    # shellcheck disable=SC2034 # used indirectly
    local SAMPLEAPISERVER_output_dir="staging/src/k8s.io/sample-apiserver/pkg/generated/openapi"
    # shellcheck disable=SC2034 # used indirectly
    local SAMPLEAPISERVER_known_violations_file="api/api-rules/sample_apiserver_violation_exceptions.list"
    # shellcheck disable=SC2034 # used indirectly
    local SAMPLEAPISERVER_tag_files=()
    kube::util::read-array SAMPLEAPISERVER_tag_files < <(
        k8s_tag_files_matching \
            vendor/k8s.io/sample-apiserver \
            "${apimachinery_dirs[@]}"
        )

    for prefix in "${targets[@]}"; do
        local report_file="${OUT_DIR}/${prefix}_violations.report"
        # When UPDATE_API_KNOWN_VIOLATIONS is set to be true, let the generator to write
        # updated API violations to the known API violation exceptions list.
        if [[ "${UPDATE_API_KNOWN_VIOLATIONS}" == true ]]; then
            report_file=$(indirect "${prefix}_known_violations_file")
        fi

        # 2 lines because shellcheck
        local output_dir
        output_dir=$(indirect "${prefix}_output_dir")

        if [[ "${DBG_CODEGEN}" == 1 ]]; then
            kube::log::status "DBG: finding all +k8s:openapi-gen tags for ${prefix}"
        fi

        local tag_dirs=()
        kube::util::read-array tag_dirs < <(
            grep --color=never -l '+k8s:openapi-gen=' $(indirect_array "${prefix}_tag_files") \
                | xargs -n1 dirname \
                | LC_ALL=C sort -u
            )

        if [[ "${DBG_CODEGEN}" == 1 ]]; then
            kube::log::status "DBG: found ${#tag_dirs[@]} +k8s:openapi-gen tagged dirs for ${prefix}"
        fi

        local tag_pkgs=()
        for dir in "${tag_dirs[@]}"; do
            tag_pkgs+=("${PRJ_SRC_PATH}/$dir")
        done

        kube::log::status "Generating openapi code for ${prefix}"
        if [[ "${DBG_CODEGEN}" == 1 ]]; then
            kube::log::status "DBG: running ${gen_openapi_bin} for:"
            for dir in "${tag_dirs[@]}"; do
                kube::log::status "DBG:     $dir"
            done
        fi

        ./hack/run-in-gopath.sh ${gen_openapi_bin} \
            --v "${KUBE_VERBOSE}" \
            --logtostderr \
            -h "${BOILERPLATE_FILENAME}" \
            -O "${output_base}" \
            -p "${PRJ_SRC_PATH}/${output_dir}" \
            -r "${report_file}" \
            $(printf -- " -i %s" "${tag_pkgs[@]}") \
            "$@"

        touch "${report_file}"
        # 2 lines because shellcheck
        local known_filename
        known_filename=$(indirect "${prefix}_known_violations_file")
        if ! diff -u "${known_filename}" "${report_file}"; then
            echo -e "ERROR:"
            echo -e "\t'${prefix}' API rule check failed - reported violations differ from known violations"
            echo -e "\tPlease read api/api-rules/README.md to resolve the failure in ${known_filename}"
        fi

        if [[ "${DBG_CODEGEN}" == 1 ]]; then
            kube::log::status "Generated openapi code"
        fi
    done # for each prefix
}

function codegen::applyconfigs() {
    GO111MODULE=on GOPROXY=off go install \
        k8s.io/kubernetes/pkg/generated/openapi/cmd/models-schema \
        k8s.io/code-generator/cmd/applyconfiguration-gen

    local modelsschema
    modelsschema=$(kube::util::find-binary "models-schema")
    local applyconfigurationgen
    applyconfigurationgen=$(kube::util::find-binary "applyconfiguration-gen")

    # because client-gen doesn't do policy/v1alpha1, we have to skip it too
    local ext_apis=()
    kube::util::read-array ext_apis < <(
      cd "${KUBE_ROOT}/staging/src"
      find k8s.io/api -name types.go -print0 | xargs -0 -n1 dirname | sort | grep -v pkg.apis.policy.v1alpha1
    )
    ext_apis+=("k8s.io/apimachinery/pkg/apis/meta/v1")

    kube::log::status "Generating apply-config code for ${#ext_apis[@]} targets"
    if [[ "${DBG_CODEGEN}" == 1 ]]; then
        kube::log::status "DBG: running ${applyconfigurationgen} for:"
        for api in "${ext_apis[@]}"; do
            kube::log::status "DBG:     $api"
        done
    fi

    "${applyconfigurationgen}" \
        --openapi-schema <("${modelsschema}") \
        --go-header-file "${BOILERPLATE_FILENAME}" \
        --output-base "${KUBE_ROOT}/vendor" \
        --output-package "${APPLYCONFIG_PKG}" \
        $(printf -- " --input-dirs %s" "${ext_apis[@]}") \
        "$@"

    if [[ "${DBG_CODEGEN}" == 1 ]]; then
        kube::log::status "Generated apply-config code"
    fi
}

function codegen::clients() {
    GO111MODULE=on GOPROXY=off go install \
        k8s.io/code-generator/cmd/client-gen

    local clientgen
    clientgen=$(kube::util::find-binary "client-gen")

    IFS=" " read -r -a group_versions <<< "${KUBE_AVAILABLE_GROUP_VERSIONS}"
    local gv_dirs=()
    for gv in "${group_versions[@]}"; do
        # add items, but strip off any leading apis/ you find to match command expectations
        local api_dir
        api_dir=$(kube::util::group-version-to-pkg-path "${gv}")
        local nopkg_dir=${api_dir#pkg/}
        nopkg_dir=${nopkg_dir#vendor/k8s.io/api/}
        local pkg_dir=${nopkg_dir#apis/}

        # skip groups that aren't being served, clients for these don't matter
        if [[ " ${KUBE_NONSERVER_GROUP_VERSIONS} " == *" ${gv} "* ]]; then
          continue
        fi

        gv_dirs+=("${pkg_dir}")
    done

    kube::log::status "Generating client code for ${#gv_dirs[@]} targets"
    if [[ "${DBG_CODEGEN}" == 1 ]]; then
        kube::log::status "DBG: running ${clientgen} for:"
        for dir in "${gv_dirs[@]}"; do
            kube::log::status "DBG:     $dir"
        done
    fi

    "${clientgen}" \
        --go-header-file "${BOILERPLATE_FILENAME}" \
        --output-base "${KUBE_ROOT}/vendor" \
        --output-package="k8s.io/client-go" \
        --clientset-name="kubernetes" \
        --input-base="k8s.io/api" \
        --apply-configuration-package "${APPLYCONFIG_PKG}" \
        $(printf -- " --input %s" "${gv_dirs[@]}") \
        "$@"

    if [[ "${DBG_CODEGEN}" == 1 ]]; then
        kube::log::status "Generated client code"
    fi
}

function codegen::listers() {
    GO111MODULE=on GOPROXY=off go install k8s.io/code-generator/cmd/lister-gen

    local listergen
    listergen=$(kube::util::find-binary "lister-gen")

    local ext_apis=()
    kube::util::read-array ext_apis < <(
        cd "${KUBE_ROOT}/staging/src"
        find k8s.io/api -name types.go -print0 | xargs -0 -n1 dirname | sort
    )

    kube::log::status "Generating lister code for ${#ext_apis[@]} targets"
    if [[ "${DBG_CODEGEN}" == 1 ]]; then
        kube::log::status "DBG: running ${listergen} for:"
        for api in "${ext_apis[@]}"; do
            kube::log::status "DBG:     $api"
        done
    fi

    "${listergen}" \
        --go-header-file "${BOILERPLATE_FILENAME}" \
        --output-base "${KUBE_ROOT}/vendor" \
        --output-package "k8s.io/client-go/listers" \
        $(printf -- " --input-dirs %s" "${ext_apis[@]}") \
        "$@"

    if [[ "${DBG_CODEGEN}" == 1 ]]; then
        kube::log::status "Generated lister code"
    fi
}

function codegen::informers() {
    GO111MODULE=on GOPROXY=off go install \
        k8s.io/code-generator/cmd/informer-gen

    local informergen
    informergen=$(kube::util::find-binary "informer-gen")

    # because client-gen doesn't do policy/v1alpha1, we have to skip it too
    local ext_apis=()
    kube::util::read-array ext_apis < <(
        cd "${KUBE_ROOT}/staging/src"
        find k8s.io/api -name types.go -print0 | xargs -0 -n1 dirname | sort | grep -v pkg.apis.policy.v1alpha1
    )

    kube::log::status "Generating informer code for ${#ext_apis[@]} targets"
    if [[ "${DBG_CODEGEN}" == 1 ]]; then
        kube::log::status "DBG: running ${informergen} for:"
        for api in "${ext_apis[@]}"; do
            kube::log::status "DBG:     $api"
        done
    fi

    "${informergen}" \
        --go-header-file "${BOILERPLATE_FILENAME}" \
        --output-base "${KUBE_ROOT}/vendor" \
        --output-package "k8s.io/client-go/informers" \
        --single-directory \
        --versioned-clientset-package k8s.io/client-go/kubernetes \
        --listers-package k8s.io/client-go/listers \
        $(printf -- " --input-dirs %s" "${ext_apis[@]}") \
        "$@"

    if [[ "${DBG_CODEGEN}" == 1 ]]; then
        kube::log::status "Generated informer code"
    fi
}

function codegen::subprojects() {
    # Call generation on sub-projects.
    # TODO(thockin): make these take a list of codegens and flags
    local subs=(
        vendor/k8s.io/code-generator/hack/update-codegen.sh
        vendor/k8s.io/kube-aggregator/hack/update-codegen.sh
        vendor/k8s.io/sample-apiserver/hack/update-codegen.sh
        vendor/k8s.io/sample-controller/hack/update-codegen.sh
        vendor/k8s.io/apiextensions-apiserver/hack/update-codegen.sh
        vendor/k8s.io/metrics/hack/update-codegen.sh
        vendor/k8s.io/apiextensions-apiserver/examples/client-go/hack/update-codegen.sh
    )

    for s in "${subs[@]}"; do 
        CODEGEN_PKG=./vendor/k8s.io/code-generator "$s"
    done
}


#
# main
#

function list_codegens() {
    (
        shopt -s extdebug
        declare -F \
            | cut -f3 -d' ' \
            | grep "^codegen::" \
            | while read -r fn; do declare -F "$fn"; done \
            | sort -n -k2 \
            | cut -f1 -d' ' \
            | sed 's/^codegen:://'
    )
}

# shellcheck disable=SC2207 # safe, no functions have spaces
all_codegens=($(list_codegens))

function print_codegens() {
    echo "available codegens:"
    for g in "${all_codegens[@]}"; do
        echo "    $g"
    done
}

# Validate and accumulate flags to pass thru and codegens to run if args are
# specified.
flags_to_pass=()
codegens_to_run=()
for arg; do
    # Use -? to list known codegens.
    if [[ "${arg}" == "-?" ]]; then
        print_codegens
        exit 0
    fi
    if [[ "${arg}" =~ ^- ]]; then
        flags_to_pass+=("${arg}")
        continue
    fi
    # Make sure each non-flag arg matches at least one codegen.
    nmatches=0
    for t in "${all_codegens[@]}"; do
        if [[ "$t" =~ ${arg} ]]; then
            nmatches=$((nmatches+1))
            # Don't run codegens twice, just keep the first match.
            # shellcheck disable=SC2076 # we want literal matching
            if [[ " ${codegens_to_run[*]} " =~ " $t " ]]; then
                continue
            fi
            codegens_to_run+=("$t")
            continue
        fi
    done
    if [[ ${nmatches} == 0 ]]; then
        echo "ERROR: no codegens match pattern '${arg}'"
        echo
        print_codegens
        exit 1
    fi
    codegens_to_run+=("${matches[@]}")
done

# If no codegens were specified, run them all.
if [[ "${#codegens_to_run[@]}" == 0 ]]; then
    codegens_to_run=("${all_codegens[@]}")
fi

for g in "${codegens_to_run[@]}"; do
    "codegen::${g}" "${flags_to_pass[@]}"
done
