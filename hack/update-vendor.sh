#!/usr/bin/env bash

# Copyright 2019 The Kubernetes Authors.
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

# Go tools really don't like it if you have a symlink in `pwd`.
cd "$(pwd -P)"

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

# Explicitly opt into go modules, even though we're inside a GOPATH directory
export GO111MODULE=on
# Explicitly set GOFLAGS to ignore vendor, since GOFLAGS=-mod=vendor breaks dependency resolution while rebuilding vendor
export GOFLAGS=-mod=mod
# Ensure sort order doesn't depend on locale
export LANG=C
export LC_ALL=C
# Detect problematic GOPROXY settings that prevent lookup of dependencies
if [[ "${GOPROXY:-}" == "off" ]]; then
  kube::log::error "Cannot run hack/update-vendor.sh with \$GOPROXY=off"
  exit 1
fi

kube::golang::verify_go_version
kube::util::require-jq

TMP_DIR="${TMP_DIR:-$(mktemp -d /tmp/update-vendor.XXXX)}"
LOG_FILE="${LOG_FILE:-${TMP_DIR}/update-vendor.log}"
kube::log::status "logfile at ${LOG_FILE}"

function finish {
  ret=$?
  if [[ ${ret} != 0 ]]; then
    echo "An error has occurred. Please see more details in ${LOG_FILE}"
  fi
  exit ${ret}
}
trap finish EXIT

if [ -z "${BASH_XTRACEFD:-}" ]; then
  exec 19> "${LOG_FILE}"
  export BASH_XTRACEFD="19"
  set -x
fi

# ensure_require_replace_directives_for_all_dependencies:
# - ensures all existing 'require' directives have an associated 'replace' directive pinning a version
# - adds explicit 'require' directives for all transitive dependencies
# - adds explicit 'replace' directives for all require directives (existing 'replace' directives take precedence)
function ensure_require_replace_directives_for_all_dependencies() {
  local local_tmp_dir
  local_tmp_dir=$(mktemp -d "${TMP_DIR}/pin_replace.XXXX")

  # collect 'require' directives that actually specify a version
  local require_filter='(.Version != null) and (.Version != "v0.0.0") and (.Version != "v0.0.0-00010101000000-000000000000")'
  # collect 'replace' directives that unconditionally pin versions (old=new@version)
  local replace_filter='(.Old.Version == null) and (.New.Version != null)'

  # Capture local require/replace directives before running any go commands that can modify the go.mod file
  local require_json="${local_tmp_dir}/require.json"
  local replace_json="${local_tmp_dir}/replace.json"
  go mod edit -json \
      | jq -r ".Require // [] | sort | .[] | select(${require_filter})" \
      > "${require_json}"
  go mod edit -json \
      | jq -r ".Replace // [] | sort | .[] | select(${replace_filter})" \
      > "${replace_json}"

  # 1a. Ensure replace directives have an explicit require directive
  jq -r '"-require \(.Old.Path)@\(.New.Version)"' < "${replace_json}" \
      | xargs -L 100 go mod edit -fmt
  # 1b. Ensure require directives have a corresponding replace directive pinning a version
  jq -r '"-replace \(.Path)=\(.Path)@\(.Version)"' < "${require_json}" \
      | xargs -L 100 go mod edit -fmt
  jq -r '"-replace \(.Old.Path)=\(.New.Path)@\(.New.Version)"' < "${replace_json}" \
      | xargs -L 100 go mod edit -fmt

  # 2. Propagate root replace/require directives into staging modules, in case we are downgrading, so they don't bump the root required version back up
  for repo in $(kube::util::list_staging_repos); do
    pushd "staging/src/k8s.io/${repo}" >/dev/null 2>&1
      jq -r '"-require \(.Path)@\(.Version)"' < "${require_json}" \
          | xargs -L 100 go mod edit -fmt
      jq -r '"-replace \(.Path)=\(.Path)@\(.Version)"' < "${require_json}" \
          | xargs -L 100 go mod edit -fmt
      jq -r '"-replace \(.Old.Path)=\(.New.Path)@\(.New.Version)"' < "${replace_json}" \
          | xargs -L 100 go mod edit -fmt
    popd >/dev/null 2>&1
  done

  # 3. Add explicit require directives for indirect dependencies
  go list -m -json all \
      | jq -r 'select(.Main != true) | select(.Indirect == true) | "-require \(.Path)@\(.Version)"' \
      | xargs -L 100 go mod edit -fmt

  # 4. Add explicit replace directives pinning dependencies that aren't pinned yet
  go list -m -json all \
      | jq -r 'select(.Main != true) | select(.Replace == null)  | "-replace \(.Path)=\(.Path)@\(.Version)"' \
      | xargs -L 100 go mod edit -fmt
}

function group_directives() {
  local local_tmp_dir
  local_tmp_dir=$(mktemp -d "${TMP_DIR}/group_replace.XXXX")
  local go_mod_require_direct="${local_tmp_dir}/go.mod.require_direct.tmp"
  local go_mod_require_indirect="${local_tmp_dir}/go.mod.require_indirect.tmp"
  local go_mod_replace="${local_tmp_dir}/go.mod.replace.tmp"
  local go_mod_other="${local_tmp_dir}/go.mod.other.tmp"
  # separate replace and non-replace directives
  awk "
     # print lines between 'require (' ... ')' lines
     /^require [(]/          { inrequire=1; next                            }
     inrequire && /^[)]/     { inrequire=0; next                            }
     inrequire && /\/\/ indirect/ { print > \"${go_mod_require_indirect}\"; next }
     inrequire               { print > \"${go_mod_require_direct}\";   next }

     # print lines between 'replace (' ... ')' lines
     /^replace [(]/      { inreplace=1; next                   }
     inreplace && /^[)]/ { inreplace=0; next                   }
     inreplace           { print > \"${go_mod_replace}\"; next }

     # print ungrouped replace directives with the replace directive trimmed
     /^replace [^(]/ { sub(/^replace /,\"\"); print > \"${go_mod_replace}\"; next }

     # print ungrouped require directives with the require directive trimmed
     /^require [^(].*\/\/ indirect/ { sub(/^require /,\"\"); print > \"${go_mod_require_indirect}\"; next }
     /^require [^(]/ { sub(/^require /,\"\"); print > \"${go_mod_require_direct}\"; next }

     # otherwise print to the other file
     { print > \"${go_mod_other}\" }
  " < go.mod
  {
    cat "${go_mod_other}";
    echo "require (";
    cat "${go_mod_require_direct}";
    echo ")";
    echo "require (";
    cat "${go_mod_require_indirect}";
    echo ")";
    echo "replace (";
    cat "${go_mod_replace}";
    echo ")";
  } > go.mod

  go mod edit -fmt
}

function add_generated_comments() {
  local local_tmp_dir
  local_tmp_dir=$(mktemp -d "${TMP_DIR}/add_generated_comments.XXXX")
  local go_mod_nocomments="${local_tmp_dir}/go.mod.nocomments.tmp"

  # drop comments before the module directive
  awk "
     BEGIN           { dropcomments=1 }
     /^module /      { dropcomments=0 }
     dropcomments && /^\/\// { next }
     { print }
  " < go.mod > "${go_mod_nocomments}"

  # Add the specified comments
  local comments="${1}"
  {
    echo "${comments}"
    echo ""
    cat "${go_mod_nocomments}"
   } > go.mod

  # Format
  go mod edit -fmt
}


# Phase 1: ensure go.mod files for staging modules and main module

for repo in $(kube::util::list_staging_repos); do
  pushd "staging/src/k8s.io/${repo}" >/dev/null 2>&1
    if [[ ! -f go.mod ]]; then
      kube::log::status "go.mod: initialize ${repo}"
      rm -f Godeps/Godeps.json # remove before initializing, staging Godeps are not authoritative
      go mod init "k8s.io/${repo}"
      go mod edit -fmt
    fi
  popd >/dev/null 2>&1
done

if [[ ! -f go.mod ]]; then
  kube::log::status "go.mod: initialize k8s.io/kubernetes"
  go mod init "k8s.io/kubernetes"
  rm -f Godeps/Godeps.json # remove after initializing
fi


# Phase 2: ensure staging repo require/replace directives

kube::log::status "go.mod: update staging references"
# Prune
go mod edit -json \
    | jq -r '.Require[]? | select(.Version == "v0.0.0")                 | "-droprequire \(.Path)"' \
    | xargs -L 100 go mod edit -fmt
go mod edit -json \
    | jq -r '.Replace[]? | select(.New.Path | startswith("./staging/")) | "-dropreplace \(.Old.Path)"' \
    | xargs -L 100 go mod edit -fmt
# Readd
kube::util::list_staging_repos \
    | while read -r X; do echo "-require k8s.io/${X}@v0.0.0"; done \
    | xargs -L 100 go mod edit -fmt
kube::util::list_staging_repos \
    | while read -r X; do echo "-replace k8s.io/${X}=./staging/src/k8s.io/${X}"; done \
    | xargs -L 100 go mod edit -fmt


# Phase 3: capture required (minimum) versions from all modules, and replaced (pinned) versions from the root module

# pin referenced versions
ensure_require_replace_directives_for_all_dependencies
# resolves/expands references in the root go.mod (if needed)
go mod tidy >>"${LOG_FILE}" 2>&1
# pin expanded versions
ensure_require_replace_directives_for_all_dependencies
# group require/replace directives
group_directives

# Phase 4: copy root go.mod to staging dirs and rewrite

kube::log::status "go.mod: propagate to staging modules"
for repo in $(kube::util::list_staging_repos); do
  pushd "staging/src/k8s.io/${repo}" >/dev/null 2>&1
    echo "=== propagating to ${repo}" >> "${LOG_FILE}"
    # copy root go.mod, changing module name
    sed "s#module k8s.io/kubernetes#module k8s.io/${repo}#" \
        < "${KUBE_ROOT}/go.mod" \
        > "${KUBE_ROOT}/staging/src/k8s.io/${repo}/go.mod"
    # remove `require` directives for staging components (will get re-added as needed by `go list`)
    kube::util::list_staging_repos \
        | while read -r X; do echo "-droprequire k8s.io/${X}"; done \
        | xargs -L 100 go mod edit
    # rewrite `replace` directives for staging components to point to peer directories
    kube::util::list_staging_repos \
        | while read -r X; do echo "-replace k8s.io/${X}=../${X}"; done \
        | xargs -L 100 go mod edit
  popd >/dev/null 2>&1
done


# Phase 5: sort and tidy staging components

kube::log::status "go.mod: sorting staging modules"
# tidy staging repos in reverse dependency order.
# the content of dependencies' go.mod files affects what `go mod tidy` chooses to record in a go.mod file.
tidy_unordered="${TMP_DIR}/tidy_unordered.txt"
kube::util::list_staging_repos \
    | xargs -I {} echo "k8s.io/{}" > "${tidy_unordered}"
rm -f "${TMP_DIR}/tidy_deps.txt"
# SC2094 checks that you do not read and write to the same file in a pipeline.
# We do read from ${tidy_unordered} in the pipeline and mention it within the
# pipeline (but only ready it again) so we disable the lint to assure shellcheck
# that :this-is-fine:
# shellcheck disable=SC2094
while IFS= read -r repo; do
  # record existence of the repo to ensure modules with no peer relationships still get included in the order
  echo "${repo} ${repo}" >> "${TMP_DIR}/tidy_deps.txt"

  pushd "${KUBE_ROOT}/staging/src/${repo}" >/dev/null 2>&1
    # save the original go.mod, since go list doesn't just add missing entries, it also removes specific required versions from it
    tmp_go_mod="${TMP_DIR}/tidy_${repo/\//_}_go.mod.original"
    tmp_go_deps="${TMP_DIR}/tidy_${repo/\//_}_deps.txt"
    cp go.mod "${tmp_go_mod}"

    {
      echo "=== sorting ${repo}"
      # 'go list' calculates direct imports and updates go.mod so that go list -m lists our module dependencies
      echo "=== computing imports for ${repo}"
      go list all
      echo "=== computing tools imports for ${repo}"
      go list -tags=tools all
    } >> "${LOG_FILE}" 2>&1

    # capture module dependencies
    go list -m -f '{{if not .Main}}{{.Path}}{{end}}' all > "${tmp_go_deps}"

    # restore the original go.mod file
    cp "${tmp_go_mod}" go.mod

    # list all module dependencies
    for dep in $(join "${tidy_unordered}" "${tmp_go_deps}"); do
      # record the relationship (put dep first, because we want to sort leaves first)
      echo "${dep} ${repo}" >> "${TMP_DIR}/tidy_deps.txt"
      # switch the required version to an explicit v0.0.0 (rather than an unknown v0.0.0-00010101000000-000000000000)
      go mod edit -require "${dep}@v0.0.0"
    done
  popd >/dev/null 2>&1
done < "${tidy_unordered}"

kube::log::status "go.mod: tidying"
for repo in $(tsort "${TMP_DIR}/tidy_deps.txt"); do
  pushd "${KUBE_ROOT}/staging/src/${repo}" >/dev/null 2>&1
    echo "=== tidying ${repo}" >> "${LOG_FILE}"

    # prune replace directives that pin to the naturally selected version.
    # do this before tidying, since tidy removes unused modules that
    # don't provide any relevant packages, which forgets which version of the
    # unused transitive dependency we had a require directive for,
    # and prevents pruning the matching replace directive after tidying.
    go list -m -json all |
      jq -r 'select(.Replace != null) |
             select(.Path == .Replace.Path) |
             select(.Version == .Replace.Version) |
             "-dropreplace \(.Replace.Path)"' |
    xargs -L 100 go mod edit -fmt

    go mod tidy -v >>"${LOG_FILE}" 2>&1

    # disallow transitive dependencies on k8s.io/kubernetes
    loopback_deps=()
    kube::util::read-array loopback_deps < <(go list all 2>/dev/null | grep k8s.io/kubernetes/ || true)
    if [[ -n ${loopback_deps[*]:+"${loopback_deps[*]}"} ]]; then
      kube::log::error "Disallowed ${repo} -> k8s.io/kubernetes dependencies exist via the following imports:
$(go mod why "${loopback_deps[@]}")"
      exit 1
    fi

    # prune unused pinned replace directives
    comm -23 \
      <(go mod edit -json | jq -r '.Replace[] | .Old.Path' | sort) \
      <(go list -m -json all | jq -r .Path | sort) |
    while read -r X; do echo "-dropreplace=${X}"; done |
    xargs -L 100 go mod edit -fmt

    # prune replace directives that pin to the naturally selected version
    go list -m -json all |
      jq -r 'select(.Replace != null) |
             select(.Path == .Replace.Path) |
             select(.Version == .Replace.Version) |
             "-dropreplace \(.Replace.Path)"' |
    xargs -L 100 go mod edit -fmt

    # group require/replace directives
    group_directives

  popd >/dev/null 2>&1
done
echo "=== tidying root" >> "${LOG_FILE}"
go mod tidy >>"${LOG_FILE}" 2>&1

# prune unused pinned non-local replace directives
comm -23 \
  <(go mod edit -json | jq -r '.Replace[] | select(.New.Path | startswith("./") | not) | .Old.Path' | sort) \
  <(go list -m -json all | jq -r .Path | sort) |
while read -r X; do echo "-dropreplace=${X}"; done |
xargs -L 100 go mod edit -fmt

# disallow transitive dependencies on k8s.io/kubernetes
loopback_deps=()
kube::util::read-array loopback_deps < <(go mod graph | grep ' k8s.io/kubernetes' || true)
if [[ -n ${loopback_deps[*]:+"${loopback_deps[*]}"} ]]; then
  kube::log::error "Disallowed transitive k8s.io/kubernetes dependencies exist via the following imports:"
  kube::log::error "${loopback_deps[@]}"
  exit 1
fi

# Phase 6: add generated comments to go.mod files
kube::log::status "go.mod: adding generated comments"
add_generated_comments "
// This is a generated file. Do not edit directly.
// Ensure you've carefully read
// https://git.k8s.io/community/contributors/devel/sig-architecture/vendor.md
// Run hack/pin-dependency.sh to change pinned dependency versions.
// Run hack/update-vendor.sh to update go.mod files and the vendor directory.
"
for repo in $(kube::util::list_staging_repos); do
  pushd "staging/src/k8s.io/${repo}" >/dev/null 2>&1
    add_generated_comments "// This is a generated file. Do not edit directly."
  popd >/dev/null 2>&1
done


# Phase 7: update internal modules
kube::log::status "vendor: updating internal modules"
hack/update-internal-modules.sh >>"${LOG_FILE}" 2>&1


# Phase 8: rebuild vendor directory
kube::log::status "vendor: running 'go mod vendor'"
go mod vendor >>"${LOG_FILE}" 2>&1

# create a symlink in vendor directory pointing to the staging components.
# This lets other packages and tools use the local staging components as if they were vendored.
for repo in $(kube::util::list_staging_repos); do
  rm -fr "${KUBE_ROOT}/vendor/k8s.io/${repo}"
  ln -s "../../staging/src/k8s.io/${repo}" "${KUBE_ROOT}/vendor/k8s.io/${repo}"
done

kube::log::status "vendor: updating vendor/LICENSES"
hack/update-vendor-licenses.sh >>"${LOG_FILE}" 2>&1

kube::log::status "vendor: creating OWNERS file"
rm -f "vendor/OWNERS"
cat <<__EOF__ > "vendor/OWNERS"
# See the OWNERS docs at https://go.k8s.io/owners

approvers:
- dep-approvers
reviewers:
- dep-reviewers
__EOF__

kube::log::status "NOTE: don't forget to handle vendor/* files that were added or removed"
