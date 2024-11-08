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

function add_staging_replace_directives() {
  local path_to_staging_k8s_io="$1"
  # Prune
  go mod edit -json \
      | jq -r '.Require[]? | select(.Version == "v0.0.0")                 | "-droprequire \(.Path)"' \
      | xargs -L 100 go mod edit -fmt
  go mod edit -json \
      | jq -r '.Replace[]? | select(.New.Path | startswith("'"${path_to_staging_k8s_io}"'")) | "-dropreplace \(.Old.Path)"' \
      | xargs -L 100 go mod edit -fmt
  # Re-add
  kube::util::list_staging_repos \
      | while read -r X; do echo "-require k8s.io/${X}@v0.0.0"; done \
      | xargs -L 100 go mod edit -fmt
  kube::util::list_staging_repos \
      | while read -r X; do echo "-replace k8s.io/${X}=${path_to_staging_k8s_io}/${X}"; done \
      | xargs -L 100 go mod edit -fmt
}

# === Capture go / godebug directives from root go.mod
go_directive_value=$(grep '^go 1.' go.mod | awk '{print $2}' || true)
if [[ -z "${go_directive_value}" ]]; then
  kube::log::error "root go.mod must have 'go 1.x.y' directive" >&22 2>&1
  exit 1
fi
godebug_directive_value=$(grep 'godebug default=go' go.mod | awk '{print $2}' || true)
if [[ -z "${godebug_directive_value}" ]]; then
  kube::log::error "root go.mod must have 'godebug default=go1.x' directive" >&22 2>&1
  exit 1
fi

# === Ensure staging go.mod files exist
for repo in $(kube::util::list_staging_repos); do
  (
    cd "staging/src/k8s.io/${repo}"

    if [[ ! -f go.mod ]]; then
      kube::log::status "go.mod: initialize ${repo}" >&11
      go mod init "k8s.io/${repo}"
    fi
    go mod edit -go "${go_directive_value}" -godebug "${godebug_directive_value}"
  )
done

# === Ensure root and staging go.mod files refer to each other using v0.0.0 and local path replaces
kube::log::status "go.mod: update staging module references" >&11
add_staging_replace_directives "./staging/src/k8s.io"
for repo in $(kube::util::list_staging_repos); do
  (
    cd "staging/src/k8s.io/${repo}"
    add_staging_replace_directives ".."
  )
done

# === Ensure all root and staging modules are included in go.work
kube::log::status "go.mod: go work use" >&11
(
  cd "${KUBE_ROOT}"
  unset GOWORK
  unset GOFLAGS
  if [[ ! -f go.work ]]; then
    kube::log::status "go.work: initialize" >&11
    go work init
  fi
  # Prune use directives
  go work edit -json \
      | jq -r '.Use[]? | "-dropuse \(.DiskPath)"' \
      | xargs -L 100 go work edit -fmt
  # Ensure go and godebug directives
  go work edit -go "${go_directive_value}" -godebug "${godebug_directive_value}"
  # Re-add use directives
  go work use .
  for repo in $(kube::util::list_staging_repos); do
    go work use "./staging/src/k8s.io/${repo}"
  done
)

# === Propagate MVS across all root / staging modules (calculated by `go work`) back into root / staging modules
kube::log::status "go.mod: go work sync" >&11
(
  cd "${KUBE_ROOT}"
  unset GOWORK
  unset GOFLAGS
  go work sync
)

# === Tidy
kube::log::status "go.mod: tidy" >&11
for repo in $(kube::util::list_staging_repos); do
  (
    echo "=== tidying k8s.io/${repo}"
    cd "staging/src/k8s.io/${repo}"
    go mod tidy -v
    group_directives
  )
done
echo "=== tidying root"
go mod tidy -v
group_directives

# === Prune unused replace directives, format modules
kube::log::status "go.mod: prune" >&11
for repo in $(kube::util::list_staging_repos); do
  (
    echo "=== pruning k8s.io/${repo}"
    cd "staging/src/k8s.io/${repo}"

    # drop all unused replace directives
    comm -23 \
      <(go mod edit -json | jq -r '.Replace[] | .Old.Path' | sort) \
      <(go list -m -json all | jq -r 'select(.Main | not) | .Path' | sort) |
    while read -r X; do echo "-dropreplace=${X}"; done |
    xargs -L 100 go mod edit -fmt

    group_directives
  )
done

echo "=== pruning root"
# drop unused replace directives other than to local paths
comm -23 \
  <(go mod edit -json | jq -r '.Replace[] | select(.New.Path | startswith("./") | not) | .Old.Path' | sort) \
  <(go list -m -json all | jq -r 'select(.Main | not) | .Path' | sort) |
while read -r X; do echo "-dropreplace=${X}"; done |
xargs -L 100 go mod edit -fmt

group_directives

# === Add generated comments to go.mod files
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

# === Update internal modules
kube::log::status "vendor: updating internal modules" >&11
hack/update-internal-modules.sh


# === Rebuild vendor directory
(
  kube::log::status "vendor: running 'go work vendor'" >&11
  unset GOWORK
  unset GOFLAGS
  # rebuild go.work.sum
  rm -f go.work.sum
  go mod download
  # rebuild vendor
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

# === Disallow transitive dependencies on k8s.io/kubernetes
kube::log::status "go.mod: prevent staging --> k8s.io/kubernetes dep" >&11
for repo in $(kube::util::list_staging_repos); do
  (
    echo "=== checking k8s.io/${repo}"
    cd "staging/src/k8s.io/${repo}"
    loopback_deps=()
    kube::util::read-array loopback_deps < <(go list all 2>/dev/null | grep k8s.io/kubernetes/ || true)
    if (( "${#loopback_deps[@]}" > 0 )); then
      kube::log::error "${#loopback_deps[@]} disallowed ${repo} -> k8s.io/kubernetes dependencies exist via the following imports: $(go mod why "${loopback_deps[@]}")" >&22 2>&1
      exit 1
    fi
  )
done

kube::log::status "go.mod: prevent k8s.io/kubernetes --> * --> k8s.io/kubernetes dep" >&11
loopback_deps=()
kube::util::read-array loopback_deps < <(go mod graph | grep ' k8s.io/kubernetes' || true)
if (( "${#loopback_deps[@]}" > 0 )); then
  kube::log::error "${#loopback_deps[@]} disallowed transitive k8s.io/kubernetes dependencies exist via the following imports:" >&22 2>&1
  kube::log::error "${loopback_deps[@]}" >&22 2>&1
  exit 1
fi

kube::log::status "NOTE: don't forget to handle vendor/* and LICENSE/* files that were added or removed" >&11
