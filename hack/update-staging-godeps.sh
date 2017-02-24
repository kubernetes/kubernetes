#!/bin/bash

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

# updates the godeps.json file in the staging folders to allow clean vendoring
# based on kubernetes levels.
# TODO this does not address client-go, since it takes a different approach to vendoring
# TODO client-go should probably be made consistent

set -o errexit
set -o nounset
set -o pipefail


KUBE_ROOT=$(dirname "${BASH_SOURCE}")/..
source "${KUBE_ROOT}/hack/lib/init.sh"

kube::golang::setup_env

ORIGINAL_GOPATH="${GOPATH}"

godepBinDir=${TMPDIR:-/tmp/}/kube-godep-bin
mkdir -p "${godepBinDir}"
godep=${godepBinDir}/bin/godep
pushd "${godepBinDir}" 2>&1 > /dev/null
	# Build the godep tool
	GOPATH="${godepBinDir}"
	rm -rf *
	go get -u github.com/tools/godep 2>/dev/null
	export GODEP="${GOPATH}/bin/godep"
	pin-godep() {
		pushd "${GOPATH}/src/github.com/tools/godep" > /dev/null
			git checkout "$1"
			"${GODEP}" go install
		popd > /dev/null
	}
	# Use to following if we ever need to pin godep to a specific version again
	pin-godep 'v74'
	"${godep}" version
popd 2>&1 > /dev/null

# keep the godep restore path reasonably stable to avoid unnecessary restores
godepRestoreDir=${TMPDIR:-/tmp/}/kube-godep-restore

TARGET_DIR=${TARGET_DIR:-${KUBE_ROOT}/staging}
echo "working in ${TARGET_DIR}"

SKIP_RESTORE=${SKIP_RESTORE:-}
if [ "${SKIP_RESTORE}" != "true" ]; then
	echo "starting godep restore"
	mkdir -p "${godepRestoreDir}"

	# add the vendor folder so that we don't redownload things during restore
	GOPATH="${godepRestoreDir}:${KUBE_ROOT}/staging:${ORIGINAL_GOPATH}"
	# restore from kubernetes godeps to ensure we get the correct levels
	# you get errors about the staging repos not using a known version control system
	${godep} restore > ${godepRestoreDir}/godep-restore.log
	echo "finished godep restore"
fi

echo "forcing fake godep folders to match the current state of master in tmp"
for stagingRepo in $(ls ${KUBE_ROOT}/staging/src/k8s.io); do
	echo "    creating ${stagingRepo}"
	rm -rf ${godepRestoreDir}/src/k8s.io/${stagingRepo}
	cp -a ${KUBE_ROOT}/staging/src/k8s.io/${stagingRepo} ${godepRestoreDir}/src/k8s.io

	# we need a git commit in the godep folder, otherwise godep won't save
	pushd ${godepRestoreDir}/src/k8s.io/${stagingRepo}
	git init > /dev/null
	# we need this so later commands work, but nothing should ever actually include these commits
	# these are local only, not global
	git config user.email "you@example.com"
	git config user.name "Your Name"
	git add . > /dev/null
	git commit -qm "fake commit"
	popd
done

function updateGodepManifest() {
	local repo=${1}

	echo "starting ${repo} save"
	mkdir -p ${TARGET_DIR}/src/k8s.io
	# if target_dir isn't the same as source dir, you need copy
	test "${KUBE_ROOT}/staging" = "${TARGET_DIR}" || cp -a ${KUBE_ROOT}/staging/src/${repo} ${TARGET_DIR}/src/k8s.io
	# remove the current Godeps.json so we always rebuild it
	rm -rf ${TARGET_DIR}/src/${repo}/Godeps
	GOPATH="${godepRestoreDir}:${TARGET_DIR}"
	pushd ${TARGET_DIR}/src/${repo}
	${godep} save ./...

	# now remove all the go files.	We'll re-run a restore, go get, godep save cycle in the sync scripts
	# to get the commits for other staging k8s.io repos anyway, so we don't need the added files
	rm -rf vendor

	echo "rewriting Godeps.json to remove commits that don't really exist because we haven't pushed the prereqs yet"
	GOPATH="${ORIGINAL_GOPATH}"
	go run "${KUBE_ROOT}/staging/godeps-json-updater.go" --godeps-file="${TARGET_DIR}/src/${repo}/Godeps/Godeps.json" --client-go-import-path="${repo}"

	popd
	echo "finished ${repo} save"
}

# move into staging and save the dependencies for everything in order
for stagingRepo in $(ls ${KUBE_ROOT}/staging/src/k8s.io); do
	# we have to skip client-go because it does unusual manipulation of its godeps
	if [ "${stagingRepo}" == "client-go" ]; then
		continue
	fi

	updateGodepManifest "k8s.io/${stagingRepo}"
done
