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

# Get all the default Go environment.
kube::golang::setup_env

# Turn off workspaces until we are ready for them later
export GOWORK=off
# Explicitly opt into go modules
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

kube::util::require-jq

TMP_DIR="${TMP_DIR:-$(mktemp -d /tmp/update-vendor.XXXX)}"
LOG_FILE="${LOG_FILE:-${TMP_DIR}/update-vendor.log}"
kube::log::status "logfile at ${LOG_FILE}"

# Set up some FDs for this script to use, while capturing everything else to
# the log. NOTHING ELSE should write to $LOG_FILE directly.
exec 11>&1            # Real stdout, use this explicitly
exec 22>&2            # Real stderr, use this explicitly
exec 1>"${LOG_FILE}"  # Automatic stdout
exec 2>&1             # Automatic stderr
set -x                # Trace this script to stderr
go env                # For the log

function finish {
  ret=$?
  if [[ ${ret} != 0 ]]; then
    echo "An error has occurred. Please see more details in ${LOG_FILE}" >&22
  fi
  exit ${ret}
}
trap finish EXIT

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

  # Propagate root replace/require directives into staging modules, in case we are downgrading, so they don't bump the root required version back up
  for repo in $(kube::util::list_staging_repos); do
    (
      cd "staging/src/k8s.io/${repo}"
      jq -r '"-require \(.Path)@\(.Version)"' < "${require_json}" \
          | xargs -L 100 go mod edit -fmt
      jq -r '"-replace \(.Old.Path)=\(.New.Path)@\(.New.Version)"' < "${replace_json}" \
          | xargs -L 100 go mod edit -fmt
    )
  done

  # tidy to ensure require directives are added for indirect dependencies
  go mod tidy
}

function print_go_mod_section() {
  local directive="$1"
  local file="$2"

  if [ -s "${file}" ]; then
      echo "${directive} ("
      cat "$file"
      echo ")"
  fi
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
    print_go_mod_section "require" "${go_mod_require_direct}"
    print_go_mod_section "require" "${go_mod_require_indirect}"
    print_go_mod_section "replace" "${go_mod_replace}"
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
  (
    cd "staging/src/k8s.io/${repo}"

    if [[ ! -f go.mod ]]; then
      kube::log::status "go.mod: initialize ${repo}" >&11
      rm -f Godeps/Godeps.json # remove before initializing, staging Godeps are not authoritative
      go mod init "k8s.io/${repo}"
      go mod edit -fmt
    fi
  )
done

if [[ ! -f go.mod ]]; then
  kube::log::status "go.mod: initialize k8s.io/kubernetes" >&11
  go mod init "k8s.io/kubernetes"
  rm -f Godeps/Godeps.json # remove after initializing
fi


# Phase 2: ensure staging repo require/replace directives

kube::log::status "go.mod: update staging references" >&11
# Prune
go mod edit -json \
    | jq -r '.Require[]? | select(.Version == "v0.0.0")                 | "-droprequire \(.Path)"' \
    | xargs -L 100 go mod edit -fmt
go mod edit -json \
    | jq -r '.Replace[]? | select(.New.Path | startswith("./staging/")) | "-dropreplace \(.Old.Path)"' \
    | xargs -L 100 go mod edit -fmt
# Re-add
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
go mod tidy
# pin expanded versions
ensure_require_replace_directives_for_all_dependencies
# group require/replace directives
group_directives

# Phase 4: copy root go.mod to staging dirs and rewrite

kube::log::status "go.mod: propagate to staging modules" >&11
for repo in $(kube::util::list_staging_repos); do
  (
    cd "staging/src/k8s.io/${repo}"

    echo "=== propagating to ${repo}"
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
  )
done


# Phase 5: sort and tidy staging components

kube::log::status "go.mod: sorting staging modules" >&11
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

  (
    cd "${KUBE_ROOT}/staging/src/${repo}"

    # save the original go.mod, since go list doesn't just add missing entries, it also removes specific required versions from it
    tmp_go_mod="${TMP_DIR}/tidy_${repo/\//_}_go.mod.original"
    tmp_go_deps="${TMP_DIR}/tidy_${repo/\//_}_deps.txt"
    cp go.mod "${tmp_go_mod}"

    echo "=== sorting ${repo}"
    # 'go list' calculates direct imports and updates go.mod so that go list -m lists our module dependencies
    echo "=== computing imports for ${repo}"
    go list all
    # ignore errors related to importing `package main` packages, but catch
    # other errors (https://github.com/golang/go/issues/59186)
    errs=()
    kube::util::read-array errs < <(
        go list -e -tags=tools -json all | jq -r '.Error.Err | select( . != null )' \
            | grep -v "is a program, not an importable package"
    )
    if (( "${#errs[@]}" != 0 )); then
        for err in "${errs[@]}"; do
            echo "${err}" >&2
        done
        exit 1
    fi

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
  )
done < "${tidy_unordered}"

kube::log::status "go.mod: tidying" >&11
for repo in $(tsort "${TMP_DIR}/tidy_deps.txt"); do
  (
    cd "${KUBE_ROOT}/staging/src/${repo}"
    echo "=== tidying ${repo}"

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

    go mod tidy -v

    # disallow transitive dependencies on k8s.io/kubernetes
    loopback_deps=()
    kube::util::read-array loopback_deps < <(go list all 2>/dev/null | grep k8s.io/kubernetes/ || true)
    if (( "${#loopback_deps[@]}" > 0 )); then
      kube::log::error "${#loopback_deps[@]} disallowed ${repo} -> k8s.io/kubernetes dependencies exist via the following imports: $(go mod why "${loopback_deps[@]}")" >&22 2>&1
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
  )
done
echo "=== tidying root"
go mod tidy

# prune unused pinned non-local replace directives
comm -23 \
  <(go mod edit -json | jq -r '.Replace[] | select(.New.Path | startswith("./") | not) | .Old.Path' | sort) \
  <(go list -m -json all | jq -r .Path | sort) |
while read -r X; do echo "-dropreplace=${X}"; done |
xargs -L 100 go mod edit -fmt

# disallow transitive dependencies on k8s.io/kubernetes
loopback_deps=()
kube::util::read-array loopback_deps < <(go mod graph | grep ' k8s.io/kubernetes' || true)
if (( "${#loopback_deps[@]}" > 0 )); then
  kube::log::error "${#loopback_deps[@]} disallowed transitive k8s.io/kubernetes dependencies exist via the following imports:" >&22 2>&1
  kube::log::error "${loopback_deps[@]}" >&22 2>&1
  exit 1
fi

# Phase 6: add generated comments to go.mod files
kube::log::status "go.mod: adding generated comments" >&11
add_generated_comments "
// This is a generated file. Do not edit directly.
// Ensure you've carefully read
// https://git.k8s.io/community/contributors/devel/sig-architecture/vendor.md
// Run hack/pin-dependency.sh to change pinned dependency versions.
// Run hack/update-vendor.sh to update go.mod files and the vendor directory.
"
for repo in $(kube::util::list_staging_repos); do
  (
    cd "staging/src/k8s.io/${repo}"
    add_generated_comments "// This is a generated file. Do not edit directly."
  )
done


# Phase 7: update internal modules
kube::log::status "vendor: updating internal modules" >&11
hack/update-internal-modules.sh


# Phase 8: rebuild vendor directory
(
  kube::log::status "vendor: running 'go work vendor'" >&11
  unset GOWORK
  unset GOFLAGS
  go work vendor
)

kube::log::status "vendor: updating vendor/LICENSES" >&11
hack/update-vendor-licenses.sh

kube::log::status "vendor: creating OWNERS file" >&11
rm -f "vendor/OWNERS"
cat <<__EOF__ > "vendor/OWNERS"
# See the OWNERS docs at https://go.k8s.io/owners

options:
  # make root approval non-recursive
  no_parent_owners: true
approvers:
- dep-approvers
reviewers:
- dep-reviewers
__EOF__

kube::log::status "NOTE: don't forget to handle vendor/* and LICENSE/* files that were added or removed" >&11
