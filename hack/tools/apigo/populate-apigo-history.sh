#!/usr/bin/env bash

# Copyright The Kubernetes Authors.
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

KUBE_ROOT=$(dirname "${BASH_SOURCE[0]}")/../../..
source "${KUBE_ROOT}/hack/lib/init.sh"
cd "${KUBE_ROOT}"

kube::golang::setup_env

OUT_DIR="${KUBE_ROOT}/_output"
EXTRACTOR_CMD="./hack/tools/apigo/main.go"
EXTRACTOR_BIN="${OUT_DIR}/apigo"
WORKTREE_DIR="${OUT_DIR}/k8s-history-wt"

mkdir -p "${OUT_DIR}"
echo "Building apigo into ${EXTRACTOR_BIN}..."
go build -o "${EXTRACTOR_BIN}" "${EXTRACTOR_CMD}"

MODULES=(
    "staging/src/k8s.io/client-go"
    "staging/src/k8s.io/apimachinery"
)

git worktree prune
if git worktree list | grep -q "${WORKTREE_DIR}"; then
    git worktree remove --force "${WORKTREE_DIR}" 2>/dev/null || true
fi
rm -rf "${WORKTREE_DIR}"

for module in "${MODULES[@]}"; do
    rm -rf "${KUBE_ROOT}/${module}/apigo"
    mkdir -p "${KUBE_ROOT}/${module}/apigo"
done

# Fetch upstream to ensure we have all remote release branches
KUBE_REMOTE=$(git remote -v | grep 'kubernetes/kubernetes\.git' | awk '{print $1}' | head -n 1)
if [ -z "${KUBE_REMOTE}" ]; then
    # Fallback if the URL format is different
    if git remote | grep -q '^upstream$'; then
        KUBE_REMOTE="upstream"
    else
        KUBE_REMOTE="origin"
    fi
fi

echo "Using remote: ${KUBE_REMOTE} for historical branches."
git fetch "${KUBE_REMOTE}"

# STARTING FROM 1.30 since it is when go workspaces were added: (regex matches 30 through 39, and anything 40+)
BRANCHES=$(git branch -r | grep -E "${KUBE_REMOTE}/release-1\.(3[0-9]|[4-9][0-9]+)$" | sed "s|.*${KUBE_REMOTE}/||" | sort -V)

for branch in $BRANCHES; do
    MINOR_VER="v${branch#release-}"
    
    echo "=================================================="
    echo "Processing Branch: ${branch} -> ${MINOR_VER}.txt"
    echo "=================================================="

    if ! git worktree add --detach "${WORKTREE_DIR}" "${KUBE_REMOTE}/${branch}" > /dev/null 2>&1; then
        echo "  -> Failed to checkout ${branch}. Skipping."
        continue
    fi

    for module_dir in "${MODULES[@]}"; do
        wt_module_path="${WORKTREE_DIR}/${module_dir}"
        workspace_api_dir="${KUBE_ROOT}/${module_dir}/apigo"

        echo "  -> Extracting API delta for ${module_dir}..."
        
        # We still use || true here because we *expect* it to warn about the v1alpha2 removals
        "${EXTRACTOR_BIN}" -update -api-dir="${workspace_api_dir}" "${wt_module_path}" > /dev/null 2>&1 || true

        if [ -f "${workspace_api_dir}/next.txt" ]; then
            mv "${workspace_api_dir}/next.txt" "${workspace_api_dir}/${MINOR_VER}.txt"
        fi
    done

    git worktree remove --force "${WORKTREE_DIR}" > /dev/null 2>&1
done

echo "=================================================="
echo "Finalizing Setup & Syncing Historical Exceptions..."
for module_dir in "${MODULES[@]}"; do
    workspace_api_dir="${KUBE_ROOT}/${module_dir}/apigo"
    
    touch "${workspace_api_dir}/next.txt"
    touch "${workspace_api_dir}/except.txt"
    
    # NEW: Run the sync command against the current master branch workspace
    "${EXTRACTOR_BIN}" -sync-exceptions -api-dir="${workspace_api_dir}" "${KUBE_ROOT}/${module_dir}"
    
    echo "Setup complete for ${module_dir}"
done

echo ""
echo "Historical population complete! Check the apigo/ directories."