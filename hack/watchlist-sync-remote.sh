#!/usr/bin/env bash
# Copyright 2026 The Kubernetes Authors.
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

set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
kubernetes_dir=$(cd "$script_dir/.." && pwd)
perf_tests_dir=${PERF_TESTS_DIR:-"$kubernetes_dir/../perf-tests"}
perf_tests_repo=${WATCHLIST_PERF_TESTS_REPO:-"https://github.com/kubernetes/perf-tests.git"}
perf_tests_ref=${WATCHLIST_PERF_TESTS_REF:-"refs/pull/4484/head"}
kubernetes_ref=${WATCHLIST_KUBERNETES_REF:-HEAD}
remote=${WATCHLIST_REMOTE:-"lukasz.szaszkiewicz@34.118.97.146"}
ssh_key=${WATCHLIST_SSH_KEY:-"$HOME/.ssh/id_rsa"}
remote_root=${WATCHLIST_REMOTE_ROOT:-"/tmp/watchlist-build-current"}
remote_perf_tests_dir=${WATCHLIST_REMOTE_PERF_TESTS_DIR:-"/home/lukasz.szaszkiewicz/perf-tests"}
ssh_opts=(-i "$ssh_key" -o StrictHostKeyChecking=no)

die() { echo "error: $*" >&2; exit 1; }

[[ -d "$perf_tests_dir/util-images/request-benchmark" ]] || die "missing perf-tests at $perf_tests_dir"
[[ -d "$kubernetes_dir/staging/src/k8s.io/client-go" ]] || die "missing staged client-go"
command -v tar >/dev/null || die "tar is required"
command -v git >/dev/null || die "git is required to resolve source refs"

kubernetes_commit=$(git -C "$kubernetes_dir" rev-parse --verify "$kubernetes_ref^{commit}" 2>/dev/null) \
  || die "cannot resolve Kubernetes source ref: $kubernetes_ref"

# Keep the benchmark source pinned to the requested perf-tests ref. The local
# checkout may contain unrelated work or may be on a different branch.
perf_tests_commit=$(git -C "$perf_tests_dir" rev-parse --verify "$perf_tests_ref^{commit}" 2>/dev/null || true)
if [[ -z "$perf_tests_commit" ]]; then
  git -C "$perf_tests_dir" fetch --quiet "$perf_tests_repo" "$perf_tests_ref"
  perf_tests_commit=$(git -C "$perf_tests_dir" rev-parse --verify FETCH_HEAD^{commit})
fi
request_benchmark_source=$(mktemp -d "${TMPDIR:-/tmp}/watchlist-request-benchmark.XXXXXX")
staged_modules_source=$(mktemp -d "${TMPDIR:-/tmp}/watchlist-staged-modules.XXXXXX")
cleanup() { rm -rf "$request_benchmark_source" "$staged_modules_source"; }
trap cleanup EXIT
git -C "$perf_tests_dir" archive "$perf_tests_commit" -- util-images/request-benchmark \
  | tar --no-xattrs --no-acls -xf - -C "$request_benchmark_source"
git -C "$kubernetes_dir" archive "$kubernetes_commit" -- \
  staging/src/k8s.io/client-go staging/src/k8s.io/api \
  staging/src/k8s.io/apimachinery staging/src/k8s.io/streaming \
  | tar --no-xattrs --no-acls -xf - -C "$staged_modules_source"

ssh "${ssh_opts[@]}" "$remote" "mkdir -p '$remote_root/perf-tests/util-images' '$remote_root/perf-tests/clusterloader2/testing/list' '$remote_root/kubernetes/staging/src/k8s.io'"
ssh "${ssh_opts[@]}" "$remote" "test -d '$remote_perf_tests_dir/clusterloader2/testing/list' || { echo 'missing remote perf-tests checkout: $remote_perf_tests_dir' >&2; exit 1; }"

copy_path() {
  local source=$1 destination=$2
  if [[ -d "$source" ]]; then
    tar --no-xattrs --no-acls -C "$source" -cf - . | ssh "${ssh_opts[@]}" "$remote" \
      "mkdir -p '$destination' && tar --no-xattrs --no-acls -xf - -C '$destination'"
  else
    tar --no-xattrs --no-acls -C "$(dirname "$source")" -cf - "$(basename "$source")" | ssh "${ssh_opts[@]}" "$remote" \
      "mkdir -p '$destination' && tar --no-xattrs --no-acls -xf - -C '$destination'"
  fi
}

copy_path "$request_benchmark_source/util-images/request-benchmark" \
  "$remote_root/perf-tests/util-images/request-benchmark"

for module in client-go api apimachinery streaming; do
  copy_path "$staged_modules_source/staging/src/k8s.io/$module" \
    "$remote_root/kubernetes/staging/src/k8s.io/$module"
done

if [[ -f "$perf_tests_dir/clusterloader2/testing/list/watchlist-profile-remote.sh" ]]; then
  copy_path "$perf_tests_dir/clusterloader2/testing/list/watchlist-profile-remote.sh" \
    "$remote_root/perf-tests/clusterloader2/testing/list"
  copy_path "$perf_tests_dir/clusterloader2/testing/list/watchlist-profile-remote.sh" \
    "$remote_perf_tests_dir/clusterloader2/testing/list"
else
  echo "Using existing remote watchlist-profile-remote.sh"
fi

ssh "${ssh_opts[@]}" "$remote" "cd '$remote_root' && sha256sum perf-tests/util-images/request-benchmark/*.go kubernetes/staging/src/k8s.io/client-go/*.go"
echo "Synced request-benchmark from perf-tests $perf_tests_commit and staged client-go modules from Kubernetes $kubernetes_commit to $remote:$remote_root"
